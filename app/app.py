# pharmacy_deserts/app/app.py
"""
Pharmacy Desert Explorer - Streamlit Application

This app provides fast startup by only loading data needed for the selected mode:
- GLM Only: Loads pre-computed GLM results (fast, no network calls)
- Math Only / Blended: Loads full dataset bundle (slower, includes ACS API call)
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to sys.path so we can import our modules
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.state import (
    load_math_dataset_bundle,
    load_glm_results,
    load_latlon_lookup,
    load_pharmacist_data_only,
    get_glm_model_info,
)
from app.auth import login_form, logout_button, is_authenticated
from app.config import get_config
from models.scoring import score_candidates, average_scores
from models.ai_scores import read_ifae_csv
from viz.map_viz import render_top10_map


def main():
    st.set_page_config(
        page_title="Pharmacy Desert Explorer", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    # Check authentication first (before loading any data)
    config = get_config()
    if config.require_auth and not is_authenticated():
        login_form()
        st.stop()

    # Session state initialization
    if 'prev_top10_zips' not in st.session_state:
        st.session_state.prev_top10_zips = []
    if 'prev_weights' not in st.session_state:
        st.session_state.prev_weights = {}

    # =========================================================================
    # HEADER & MODE SELECTION (before any data loading)
    # =========================================================================
    st.title("Pharmacy Desert Explorer")
    st.markdown("""
    ### Hybrid GLM + Mathematical Approach
    1) **Math model** (weighted, adjustable)  •  2) **GLM+Hybrid model** (Poisson GLM + GBDT/XGBoost residuals)  
    **Final Ranking** blends both for robust, research-grade results.
    """)
    st.divider()

    # GLM Model Status (lightweight check)
    st.sidebar.header("🧠 ML Model (GLM+Expected-Access)")
    results_exist, last_modified = get_glm_model_info()
    if results_exist:
        st.sidebar.info(f"📊 Model trained: {last_modified.strftime('%m/%d/%y %H:%M')}")
    else:
        st.sidebar.warning("⚠️ No trained model found. Run training script first.")
    st.sidebar.divider()

    # Scoring mode selection - THIS MUST COME BEFORE DATA LOADING
    st.sidebar.header("Scoring Mode")
    scoring_mode = st.sidebar.radio(
        "Choose ranking method:", 
        ["GLM Only", "Blended (Math + GLM)", "Math Only"], 
        index=0,
        help="GLM Only: Fast startup, pure ML model | Blended: 50% Math + 50% GLM | Math Only: Full dataset + adjustable weights"
    )
    st.sidebar.divider()

    # Refresh Data button and logout
    st.sidebar.header("🔄 Data Management")
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload all data"):
        st.cache_data.clear()
        st.rerun()
    
    # Show logout button if authenticated
    logout_button()
    
    if scoring_mode == "GLM Only":
        st.sidebar.caption("💡 GLM Only mode: Fast startup, no network calls")
    else:
        st.sidebar.caption("⚠️ Math/Blended mode: Loads full dataset (slower, uses Census API)")
    st.sidebar.divider()

    # =========================================================================
    # CONDITIONAL DATA LOADING
    # =========================================================================
    
    if scoring_mode == "GLM Only":
        # =====================================================================
        # GLM ONLY MODE - Lightweight loading, no network calls
        # =====================================================================
        if not results_exist:
            st.error("GLM results not found. Please run the training script first or switch to Math Only mode.")
            st.stop()
        
        # Load GLM results (cached)
        ranked = load_glm_results()
        
        if ranked.empty:
            st.error("Failed to load GLM results.")
            st.stop()
        
        # Check if we need to merge lat/lon
        if 'lat' not in ranked.columns or ranked['lat'].isna().all():
            latlon_df = load_latlon_lookup()
            if not latlon_df.empty:
                ranked = ranked.merge(latlon_df[['zip', 'lat', 'lon']], on='zip', how='left')
        
        # Load pharmacist data (needed for map popups)
        pharmacist_data = load_pharmacist_data_only()
        
        # Set up variables for GLM mode
        df = None  # Not used in GLM Only mode
        sort_col = 'final_score'
        
        st.success(f"✅ GLM Model loaded - {len(ranked):,} ZIPs analyzed (fast mode)")
        st.sidebar.success(f"✅ GLM Model (PURE) - {len(ranked):,} ZIPs")
        
        # Disable sliders/filters in GLM mode
        w_health = w_income = w_scarcity = w_drive_time = w_pop = w_edu = w_heat = 0.0
        gate_goodrx = False
        
    else:
        # =====================================================================
        # MATH / BLENDED MODE - Full dataset loading
        # =====================================================================
        try:
            df, pharmacist_data = load_math_dataset_bundle()
            
            pharm_count = len(pharmacist_data) if not pharmacist_data.empty else 0
            unique_pharm_zips = pharmacist_data['Short_ZIP'].nunique() if not pharmacist_data.empty else 0
            st.success(f"Data loaded successfully! Analyzing {len(df):,} ZIP codes | {pharm_count:,} pharmacist records from {unique_pharm_zips} ZIPs")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # =====================================================================
        # MATH MODEL WEIGHTS (only active in Math/Blended modes)
        # =====================================================================
        st.sidebar.header("Mathematical Model Weights")
        st.sidebar.markdown("*Adjust priorities to see ranking changes:*")

        st.sidebar.markdown("**METRICS:**")
        
        # PRIMARY: Demographic vulnerability
        w_health = st.sidebar.slider("Health burden", 0.0, 1.0, 0.25, 0.05)
        w_income = st.sidebar.slider("Income (low → worse)", 0.0, 1.0, 0.20, 0.05)

        # SECONDARY: Pharmacy access
        w_scarcity = st.sidebar.slider("Scarcity (fewer pharmacies)", 0.0, 1.0, 0.20, 0.05)
        w_drive_time = st.sidebar.slider(
            "🚗 Drive Time (minutes)", 0.0, 1.0, 0.15, 0.05,
            help="Higher = prioritize longer drive times"
        ) if 'zip_drive_time' in df.columns else 0.0

        # TERTIARY: Context
        w_pop = st.sidebar.slider("Population density", 0.0, 1.0, 0.10, 0.05)
        w_edu = st.sidebar.slider(
            "Education (low attainment)", 0.0, 1.0, 0.07, 0.05
        ) if 'edu_hs_or_lower_pct' in df.columns else 0.0
        w_heat = st.sidebar.slider(
            "Heat vulnerability (HHI)", 0.0, 1.0, 0.03, 0.05
        ) if 'heat_hhb' in df.columns else 0.0

        # GoodRx hard gate controls
        st.sidebar.header("GoodRx Desert Gate")
        gate_goodrx = st.sidebar.checkbox(
            "Hard gate to GoodRx-defined deserts", 
            value=False,
            help="When ON, only ZIPs that are GoodRx drive-time deserts are kept. ⚠️ May introduce geographic bias."
        )
        min_cov = st.sidebar.slider("Minimum crosswalk coverage (HUD)", 0.0, 1.0, 0.60, 0.05)
        thr_goodrx = st.sidebar.slider("Desert severity threshold (advanced)", 0.0, 1.0, 0.50, 0.05)

        # Normalize weights
        total = w_drive_time + w_scarcity + w_health + w_income + w_pop + w_heat + w_edu
        if total > 0:
            w_drive_time, w_scarcity, w_health, w_income, w_pop, w_heat, w_edu = [
                w / total for w in [w_drive_time, w_scarcity, w_health, w_income, w_pop, w_heat, w_edu]
            ]

        st.sidebar.markdown("---")
        st.sidebar.caption("**Normalized Weights:**")
        if w_drive_time > 0:
            st.sidebar.caption(f"🚗 Drive Time: {w_drive_time:.2%}")
        for name, val in [("Scarcity", w_scarcity), ("Health", w_health), ("Income", w_income),
                          ("Population", w_pop), ("Heat", w_heat), ("Education", w_edu)]:
            if val > 0:
                st.sidebar.caption(f"{name}: {val:.2%}")

        # Hard gate (before scoring)
        if "zip_desert_share" in df.columns:
            df["zip_desert_flag_user"] = (df["zip_desert_share"] >= thr_goodrx).astype("Int64")
        else:
            df["zip_desert_flag_user"] = pd.NA

        # Apply GoodRx gate
        if gate_goodrx:
            flag_col = "zip_desert_flag_user" if "zip_desert_flag_user" in df.columns else "zip_desert_flag"
            if flag_col not in df.columns:
                st.warning("GoodRx gate requested, but downscaled fields not found.")
            else:
                mask = (df[flag_col] == 1) & (df.get("zip_alloc_coverage", 0).fillna(0) >= min_cov)
                kept = df.loc[mask].copy()
                dropped = len(df) - len(kept)
                st.info(f"GoodRx hard gate active → kept {len(kept):,} ZIPs, filtered out {dropped:,}.")
                if kept.empty:
                    st.warning("No ZIPs pass the GoodRx gate. Lower coverage/threshold.")
                    st.stop()
                df = kept
            st.caption(f"🔒 GoodRx hard gate ON · min HUD coverage ≥ {min_cov:.0%} · threshold ≥ {thr_goodrx:.0%}")

        # Population/Urban filters
        st.sidebar.header("🏙️ Target Area Filters")
        st.sidebar.markdown("*Focus on semi-urban communities:*")

        min_population = st.sidebar.slider(
            "Minimum population", 0, 50000, 5000, 1000,
            help="Exclude very small ZIPs"
        )
        min_density = st.sidebar.slider(
            "Minimum density (people/km²)", 0, 1000, 100, 50,
            help="100-400 = semi-urban sweet spot"
        )
        max_density = st.sidebar.slider(
            "Maximum density (people/km²)", 0, 10000, 5000, 500,
            help="Exclude extremely dense urban cores if desired. 0 = no max"
        )

        # Apply filters
        filters_applied = []
        df_before_filters = len(df)

        if min_population > 0:
            df = df[df.get('population', df['pop_density'] * 100).fillna(0) >= min_population]
            filters_applied.append(f"pop ≥ {min_population:,}")
        if min_density > 0:
            df = df[df['pop_density'].fillna(0) >= min_density]
            filters_applied.append(f"density ≥ {min_density}")
        if max_density > 0:
            df = df[df['pop_density'].fillna(999999) <= max_density]
            filters_applied.append(f"density ≤ {max_density}")

        if filters_applied:
            filtered_count = df_before_filters - len(df)
            st.info(f"🏙️ Target area filters → kept {len(df):,} ZIPs, filtered {filtered_count:,}")
            st.caption(f"Active: {' | '.join(filters_applied)}")

        if df.empty:
            st.warning("No ZIPs pass all filters. Relax filter criteria.")
            st.stop()

        # =====================================================================
        # SCORING (Math Only or Blended)
        # =====================================================================
        ranked = score_candidates(
            df, w_scarcity, w_health, w_income, w_pop,
            w_heat=w_heat, w_edu=w_edu, w_drive_time=w_drive_time
        )
        ranked['score'] = pd.to_numeric(ranked['score'], errors='coerce')
        ranked = ranked.dropna(subset=['score'])
        math_df = ranked[['zip', 'score']].rename(columns={'score': 'score_math'}).copy()

        if scoring_mode == "Math Only":
            ranked['final_score'] = ranked['score']
            ranked['ai_score'] = np.nan
            sort_col = 'final_score'
            st.sidebar.success("Using Mathematical Model Only")
        else:  # Blended
            ai_df = read_ifae_csv("results/national_ifae_rank.csv")
            combo = average_scores(math_df, ai_df, normalize=True)
            ranked = ranked.merge(combo, on='zip', how='left')
            sort_col = 'final_score'
            st.sidebar.success("Using Blended Approach")

    # =========================================================================
    # COMMON POST-PROCESSING (all modes)
    # =========================================================================
    ranked[sort_col] = pd.to_numeric(ranked[sort_col], errors='coerce')
    ranked = ranked.dropna(subset=[sort_col])
    ranked['desert_flag'] = ranked['desert_flag'].astype(int)

    # Sort by score only (don't prioritize desert_flag)
    ranked = ranked.sort_values(sort_col, ascending=False, na_position='last').reset_index(drop=True)

    # =========================================================================
    # UI OUTPUT
    # =========================================================================
    st.write("### Top Pharmacy Desert Candidates")
    mode_labels = {
        'Math Only': '🔢 Mathematical Model',
        'GLM Only': '🧠 GLM+Hybrid Model (Poisson + GBDT/XGBoost)',
        'Blended (Math + GLM)': '⚖️ Hybrid: Math + GLM'
    }
    st.caption(f"**Active Mode:** {mode_labels[scoring_mode]}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total ZIPs Analyzed", f"{len(ranked):,}")
    with col2:
        st.metric("Zero-Pharmacy Deserts", f"{(ranked['desert_flag'] == 1).sum():,}")
    with col3:
        st.metric("Avg Final Score", f"{ranked[sort_col].mean():.3f}")
    with col4:
        if 'ai_score' in ranked.columns and ranked['ai_score'].notna().any():
            ai_coverage = ranked['ai_score'].notna().sum()
            st.metric("GLM Coverage", f"{100 * ai_coverage / len(ranked):.1f}%")
        else:
            st.metric("GLM Coverage", "0%")

    # Weight distribution chart (Math/Blended modes only)
    if scoring_mode != "GLM Only":
        st.divider()
        st.write("#### Weight Distribution")
        import plotly.graph_objects as go
        names, values = [], []
        if w_drive_time > 0:
            names.append('🚗 Drive Time')
            values.append(w_drive_time)
        names += ['Scarcity', 'Health', 'Income', 'Population']
        values += [w_scarcity, w_health, w_income, w_pop]
        if w_heat > 0:
            names.append('Heat')
            values.append(w_heat)
        if w_edu > 0:
            names.append('Education')
            values.append(w_edu)
        fig = go.Figure(data=[go.Bar(
            x=names, y=values, 
            text=[f'{v:.1%}' for v in values], 
            textposition='auto'
        )])
        fig.update_layout(
            title="Current Weight Distribution (Normalized)", 
            yaxis_title="Weight", 
            height=300, 
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        if scoring_mode == "Blended (Math + GLM)":
            st.info("In Blended mode, Math weights affect 50% of the final score")

    # Columns to show in results table
    show_cols = ['zip', 'population', 'n_pharmacies', 'pharm_per_10k', 'pop_density', 'median_income', 'health_burden']
    if 'zip_drive_time' in ranked.columns:
        show_cols.append('zip_drive_time')
    for c in ["zip_desert_share", "zip_desert_flag", "zip_desert_flag_user", "zip_desert_pop_pct", 
              "zip_alloc_coverage", "zip_alloc_method"]:
        if c in ranked.columns:
            show_cols.append(c)
    for c in ['heat_hhb', 'edu_hs_or_lower_pct']:
        if c in ranked.columns:
            show_cols.append(c)
    for c in ['scarcity', 'pop_norm', 'income_inv', 'health_n', 'score_math', 'ai_score', 'final_score']:
        if c in ranked.columns:
            show_cols.append(c)
    show_cols += ['desert_flag']
    show_cols = [c for c in show_cols if c in ranked.columns]

    with st.expander("Understanding the Scores", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scores**: `score_math` (weighted), `ai_score` (IFAE), `final_score` (blend), `desert_flag` (pharmacy need indicator).")
            st.markdown("**Key Metrics**: `pharm_per_10k` (pharmacies per 10K people), `population` (total ZIP population)")
        with c2:
            st.markdown("**Features**: `scarcity`(population-adjusted), `health_n`, `income_inv`, `pop_norm`, optional: `drive_time_norm`, `heat_norm`, `edu_low_norm`.")
            st.markdown("**Desert Flag**: ZIPs with <5 pharmacies per 10K people are flagged as high-need areas.")

    st.dataframe(ranked[show_cols].head(50), use_container_width=True, height=400)

    # Interactive Map
    st.write("### Top 10 ZIPs on Interactive Map")
    map_key = f"map_{scoring_mode}_{len(ranked)}"
    if map_key not in st.session_state:
        st.session_state[map_key] = True

    render_top10_map(ranked.head(10).copy(), pharmacist_df=pharmacist_data)

    # Export Results
    st.write("### Export Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Full Results CSV", 
            ranked.to_csv(index=False), 
            "pharmacy_desert_candidates_full.csv", 
            "text/csv"
        )
    with col2:
        st.download_button(
            "Download Top 100 CSV", 
            ranked.head(100).to_csv(index=False), 
            "pharmacy_desert_top100.csv", 
            "text/csv"
        )

    st.divider()
    st.caption("🏥 Pharmacy Desert Explorer | Hybrid AI + Mathematical Approach")
    st.caption("Built with Streamlit | Data refreshed on page load")


if __name__ == "__main__":
    main()
