# pharmacy_deserts/app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to sys.path so we can import our modules
# This is needed because streamlit runs app.py as a top-level script
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_processing import (
    read_financial_data, read_health_data, read_pharmacy_data, read_population_data,
    read_hhi_excel, read_ifae_csv, read_population_labels,
    read_education_data_acs, read_hud_zip_county_crosswalk, read_county_desert_csv,
    downscale_county_to_zip, preprocess, score_candidates, average_scores,
    load_all_pharmacist_data, run_glm_training
)
from viz.map_viz import render_top10_map

def main():
    st.set_page_config(page_title="Pharmacy Desert Explorer", layout="wide", initial_sidebar_state="expanded")

    if 'prev_top10_zips' not in st.session_state: st.session_state.prev_top10_zips = []
    if 'prev_weights'   not in st.session_state: st.session_state.prev_weights = {}

    st.title("Pharmacy Desert Explorer")
    st.markdown("""
    ### Hybrid GLM + Mathematical Approach
    1) **Math model** (weighted, adjustable)  •  2) **GLM+Hybrid model** (Poisson GLM + GBDT/XGBoost residuals)  
    **Final Ranking** blends both for robust, research-grade results.
    """)
    st.divider()

    with st.spinner("Loading data... (this only happens once per session)"):
        try:
            financial_data = read_financial_data('data/financial_data.csv')
            health_data    = read_health_data('data/health_data.csv')
            pharmacy_data  = read_pharmacy_data('data/pharmacy_data.csv')
            population_data= read_population_data('data/population_data.csv')
            hhi           = read_hhi_excel('data/HHI_data.xlsx')
            education     = read_education_data_acs(year=2023)
            hud_xwalk     = read_hud_zip_county_crosswalk("data/zip_county_cross.xlsx")
            county_df     = read_county_desert_csv("data/driving-time-desert.csv")
            zip_desert_df = downscale_county_to_zip(county_df, hud_xwalk, tiny_cutoff=0.01, min_coverage=0.60, threshold=0.50)
            
            # Load pharmacist data
            pharmacist_data = load_all_pharmacist_data('data')

            df = preprocess(financial_data, health_data, pharmacy_data, population_data, hhi=hhi)
            df = df.merge(education[["zip","edu_hs_or_lower_pct"]], on='zip', how='left')
            df = df.merge(zip_desert_df, on='zip', how='left')
            
            pharm_count = len(pharmacist_data) if not pharmacist_data.empty else 0
            unique_pharm_zips = pharmacist_data['Short_ZIP'].nunique() if not pharmacist_data.empty else 0
            st.success(f"Data loaded successfully! Analyzing {len(df):,} ZIP codes | {pharm_count:,} pharmacist records from {unique_pharm_zips} ZIPs")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    # GLM Training Controls
    st.sidebar.header("🧠 ML Model (GLM+Expected-Access)")
    
    col_train1, col_train2 = st.sidebar.columns([2, 1])
    with col_train1:
        if st.button("🔄 Retrain Model", help="Run GLM training with OOF per-state calibration", use_container_width=True):
            with st.spinner("Training GLM model... This may take several minutes."):
                success = run_glm_training(force_retrain=True)
                if success:
                    st.success("✅ Model retrained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Training failed. Check console for details.")
    with col_train2:
        results_exist = Path("results/national_ifae_rank.csv").exists()
        if results_exist:
            st.markdown("✅")
        else:
            st.markdown("❌")
    
    if not results_exist:
        st.sidebar.warning("⚠️ No trained model found. Click 'Retrain Model' to generate GLM scores.")
    
    st.sidebar.divider()
    
    # Sidebar: mode
    st.sidebar.header("Scoring Mode")
    scoring_mode = st.sidebar.radio("Choose ranking method:", ["GLM Only", "Blended (Math + GLM)", "Math Only"], index=0,
                                    help="GLM Only: Pure ML model (recommended for diverse results) | Blended: 50% Math + 50% GLM | Math Only: Show slider impact")
    st.sidebar.divider()

    # Weights
    st.sidebar.header("Mathematical Model Weights")
    slider_disabled = (scoring_mode == "GLM Only")
    if slider_disabled: st.sidebar.info("Sliders disabled in GLM-only mode")
    else: st.sidebar.markdown("*Adjust priorities to see ranking changes:*")

    st.sidebar.markdown("**METRICS:**")
    # In app.py, change the default values on lines 71-80:

    # PRIMARY: Demographic vulnerability (not just access)
    w_health   = st.sidebar.slider("Health burden", 0.0, 1.0, 0.25, 0.05, disabled=slider_disabled)
    w_income   = st.sidebar.slider("Income (low → worse)", 0.0, 1.0, 0.20, 0.05, disabled=slider_disabled)

    # SECONDARY: Pharmacy access
    w_scarcity = st.sidebar.slider("Scarcity (fewer pharmacies)", 0.0, 1.0, 0.20, 0.05, disabled=slider_disabled)
    w_drive_time = st.sidebar.slider("🚗 Drive Time (minutes)", 0.0, 1.0, 0.15, 0.05,
                                    help="Higher = prioritize longer drive times", disabled=slider_disabled) if 'zip_drive_time' in df.columns else 0.0

    # TERTIARY: Context
    w_pop      = st.sidebar.slider("Population density", 0.0, 1.0, 0.10, 0.05, disabled=slider_disabled)
    w_edu      = st.sidebar.slider("Education (low attainment)", 0.0, 1.0, 0.07, 0.05, disabled=slider_disabled) if ('edu_hs_or_lower_pct' in df.columns) else 0.0
    w_heat     = st.sidebar.slider("Heat vulnerability (HHI)", 0.0, 1.0, 0.03, 0.05, disabled=slider_disabled) if ('heat_hhb' in df.columns) else 0.0


    # GoodRx hard gate controls (disabled in GLM mode)
    st.sidebar.header("GoodRx Desert Gate")
    if scoring_mode == "GLM Only":
        st.sidebar.info("⚠️ Disabled in GLM Only mode")
        gate_goodrx = False
    else:
        gate_goodrx = st.sidebar.checkbox("Hard gate to GoodRx-defined deserts", value=False,
                                          help="When ON, only ZIPs that are GoodRx drive-time deserts (via county→ZIP downscale) are kept. ⚠️ May introduce geographic bias.")
    
    gate_disabled = (scoring_mode == "GLM Only")
    min_cov     = st.sidebar.slider("Minimum crosswalk coverage (HUD)", 0.0, 1.0, 0.60, 0.05, disabled=gate_disabled)
    thr_goodrx  = st.sidebar.slider("Desert severity threshold (advanced)", 0.0, 1.0, 0.50, 0.05, disabled=gate_disabled)

    # Normalize weights
    total = w_drive_time + w_scarcity + w_health + w_income + w_pop + w_heat + w_edu
    if total > 0:
        w_drive_time, w_scarcity, w_health, w_income, w_pop, w_heat, w_edu = [
            w/total for w in [w_drive_time, w_scarcity, w_health, w_income, w_pop, w_heat, w_edu]
        ]

    st.sidebar.markdown("---")
    st.sidebar.caption("**Normalized Weights:**")
    if w_drive_time > 0: st.sidebar.caption(f"🚗 Drive Time: {w_drive_time:.2%}")
    for name,val in [("Scarcity",w_scarcity),("Health",w_health),("Income",w_income),("Population",w_pop),
                     ("Heat",w_heat),("Education",w_edu)]:
        if val > 0: st.sidebar.caption(f"{name}: {val:.2%}")

    # Hard gate (before scoring)
    if "zip_desert_share" in df.columns:
        df["zip_desert_flag_user"] = (df["zip_desert_share"] >= thr_goodrx).astype("Int64")
    else:
        df["zip_desert_flag_user"] = pd.NA

    # GoodRx gate only applies to Math/Blended modes (GLM has its own logic)
    if gate_goodrx and scoring_mode != "GLM Only":
        flag_col = "zip_desert_flag_user" if "zip_desert_flag_user" in df.columns else "zip_desert_flag"
        if flag_col not in df.columns:
            st.warning("GoodRx gate requested, but downscaled fields not found.")
        else:
            mask = (df[flag_col] == 1) & (df.get("zip_alloc_coverage", 0).fillna(0) >= min_cov)
            kept = df.loc[mask].copy(); dropped = len(df) - len(kept)
            st.info(f"GoodRx hard gate active → kept {len(kept):,} ZIPs, filtered out {dropped:,}.")
            if kept.empty:
                st.warning("No ZIPs pass the GoodRx gate. Lower coverage/threshold.")
                st.stop()
            df = kept
        st.caption(f"🔒 GoodRx hard gate ON · min HUD coverage ≥ {min_cov:.0%} · threshold ≥ {thr_goodrx:.0%}")

    # Population/Urban filters (disabled in GLM mode)
    st.sidebar.header("🏙️ Target Area Filters")
    if scoring_mode == "GLM Only":
        st.sidebar.info("⚠️ Disabled in GLM Only mode (model uses all data)")
    else:
        st.sidebar.markdown("*Focus on semi-urban communities:*")

    filters_disabled = (scoring_mode == "GLM Only")
    min_population = st.sidebar.slider(
        "Minimum population", 
        0, 50000, 5000, 1000,
        help="Exclude very small ZIPs",
        disabled=filters_disabled
    )

    min_density = st.sidebar.slider(
        "Minimum density (people/km²)", 
        0, 1000, 100, 50,
        help="100-400 = semi-urban sweet spot",
        disabled=filters_disabled
    )

    max_density = st.sidebar.slider(
        "Maximum density (people/km²)",
        0, 10000, 5000, 500,
        help="Exclude extremely dense urban cores if desired. 0 = no max",
        disabled=filters_disabled
    )

    # Apply filters only for Math/Blended modes (GLM uses its own complete dataset)
    if scoring_mode != "GLM Only":
        filters_applied = []
        df_before_filters = len(df)
        
        if min_population > 0:
            df = df[df.get('population', df['pop_density']*100).fillna(0) >= min_population]
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

    # Load GLM scores FIRST (before Math scoring)
    st.sidebar.markdown("---")
    ai_file_path = "results/national_ifae_rank.csv"
    try:
        ai_df = read_ifae_csv(ai_file_path)
    except Exception as e:
        st.sidebar.error(f"Error loading GLM scores: {e}")
        st.sidebar.info("Using mathematical scores only")
        ai_df = pd.DataFrame(columns=['zip','ai_score'])

    # In GLM Only mode, skip ALL app filtering - use model's own logic
    if scoring_mode == "GLM Only":
        # Load full GLM results directly - NO filters applied
        # The GLM model already has its own sophisticated logic built-in
        glm_full = pd.read_csv(ai_file_path, dtype={"ZCTA5": str})
        glm_full["zip"] = glm_full["ZCTA5"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
        glm_full["ai_score"] = pd.to_numeric(glm_full["IFAE_score"], errors="coerce")
        
        # Create ranked from full GLM results - NO FILTERING
        cols_to_load = ['zip', 'ai_score', 'ZCTA5', 'REGION', 'population', 'pharmacies_count', 
                        'median_income', 'poor_health_pct', 'pop_density']
        
        # Check if LON/LAT exist in GLM results
        if 'LON' in glm_full.columns and 'LAT' in glm_full.columns:
            cols_to_load.extend(['LON', 'LAT'])
        
        ranked = glm_full[cols_to_load].copy()
        ranked['final_score'] = ranked['ai_score']
        ranked['score'] = np.nan
        ranked['desert_flag'] = (ranked['pharmacies_count'] < 5).astype(int)
        
        # Rename columns to match app expectations
        if 'median_income' in ranked.columns:
            ranked['income'] = ranked['median_income']
        if 'poor_health_pct' in ranked.columns:
            ranked['health_poor_pct'] = ranked['poor_health_pct']
        
        # Add n_pharmacies for map compatibility (GLM uses pharmacies_count)
        if 'pharmacies_count' in ranked.columns:
            ranked['n_pharmacies'] = ranked['pharmacies_count']
        
        # Get lat/lon from population data (GLM results don't have them)
        # Population data has 'lat' and 'long', map viz expects 'lat' and 'lon'
        if 'lat' in df.columns:
            if 'lon' in df.columns:
                ranked = ranked.merge(df[['zip', 'lat', 'lon']], on='zip', how='left')
            elif 'long' in df.columns:
                # Population data uses 'long' not 'lon'
                ranked = ranked.merge(df[['zip', 'lat', 'long']], on='zip', how='left')
                ranked['lon'] = ranked['long']  # Rename for map compatibility
        elif 'LON' in ranked.columns and 'LAT' in ranked.columns:
            # If GLM had them, rename to lowercase
            ranked['lon'] = pd.to_numeric(ranked['LON'], errors='coerce')
            ranked['lat'] = pd.to_numeric(ranked['LAT'], errors='coerce')
        else:
            # Last resort: try to load from population data directly
            try:
                pop_labels = read_population_labels('data/population_data.csv')
                if 'lat' in pop_labels.columns:
                    lon_col = 'lon' if 'lon' in pop_labels.columns else 'long' if 'long' in pop_labels.columns else None
                    if lon_col:
                        ranked = ranked.merge(pop_labels[['zip', 'lat', lon_col]], on='zip', how='left')
                        if lon_col == 'long':
                            ranked['lon'] = ranked['long']
            except:
                pass  # If this fails, map will show warning about missing lat/lon
        
        # Calculate pharm_per_10k if not present
        if 'pharm_per_10k' not in ranked.columns:
            ranked['pharm_per_10k'] = (ranked['pharmacies_count'] / ranked['population'].clip(lower=1)) * 10000
        
        sort_col = 'final_score'
        st.sidebar.success(f"✅ GLM Model (PURE) - {len(ranked):,} ZIPs analyzed")
    else:
        # Math Only or Blended: Run Math model
        ranked = score_candidates(df, w_scarcity, w_health, w_income, w_pop, w_heat=w_heat, w_edu=w_edu, w_drive_time=w_drive_time)
        ranked['score'] = pd.to_numeric(ranked['score'], errors='coerce')
        ranked = ranked.dropna(subset=['score'])
        math_df = ranked[['zip','score']].rename(columns={'score':'score_math'}).copy()
        
        if scoring_mode == "Math Only":
            ranked['final_score'] = ranked['score']
            ranked['ai_score'] = np.nan
            sort_col = 'final_score'
            st.sidebar.success("Using Mathematical Model Only")
        else:  # Blended
            combo = average_scores(math_df, ai_df, normalize=True)
            ranked = ranked.merge(combo, on='zip', how='left')
            sort_col = 'final_score'
            st.sidebar.success("Using Blended Approach")

    ranked[sort_col] = pd.to_numeric(ranked[sort_col], errors='coerce')
    ranked = ranked.dropna(subset=[sort_col])
    ranked['desert_flag'] = ranked['desert_flag'].astype(int)

    # Sort by score only (don't prioritize desert_flag to avoid geographic clustering)
    # This allows GLM model to find underserved areas even with some pharmacies
    ranked = ranked.sort_values(sort_col, ascending=False, na_position='last').reset_index(drop=True)

    # ----- UI -----
    st.write("### Top Pharmacy Desert Candidates")
    st.caption(f"**Active Mode:** { {'Math Only':'🔢 Mathematical Model','GLM Only':'🧠 GLM+Hybrid Model (Poisson + GBDT/XGBoost)','Blended (Math + GLM)':'⚖️ Hybrid: Math + GLM'}[scoring_mode] }")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total ZIPs Analyzed", f"{len(ranked):,}")
    with col2: st.metric("Zero-Pharmacy Deserts", f"{(ranked['desert_flag']==1).sum():,}")
    with col3: st.metric("Avg Final Score", f"{ranked[sort_col].mean():.3f}")
    with col4:
        if 'ai_score' in ranked.columns and ranked['ai_score'].notna().any():
            ai_coverage = ranked['ai_score'].notna().sum()
            st.metric("GLM Coverage", f"{100*ai_coverage/len(ranked):.1f}%")
        else:
            st.metric("GLM Coverage", "0%")

    if scoring_mode != "GLM Only":
        st.divider(); st.write("#### Weight Distribution")
        import plotly.graph_objects as go
        names, values = [], []
        if w_drive_time > 0: names.append('🚗 Drive Time'); values.append(w_drive_time)
        names += ['Scarcity','Health','Income','Population']; values += [w_scarcity,w_health,w_income,w_pop]
        if w_heat>0: names.append('Heat'); values.append(w_heat)
        if w_edu>0: names.append('Education'); values.append(w_edu)
        fig = go.Figure(data=[go.Bar(x=names, y=values, text=[f'{v:.1%}' for v in values], textposition='auto')])
        fig.update_layout(title="Current Weight Distribution (Normalized)", yaxis_title="Weight", height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if scoring_mode == "Blended (Math + AI)": st.info("In Blended mode, Math weights affect 50% of the final score")

    # Optional Top-10 change tracker (Math Only) — unchanged from your UI, omitted here for brevity

    # Columns to show
    show_cols = ['zip','population','n_pharmacies','pharm_per_10k','pop_density','median_income','health_burden']
    if 'zip_drive_time' in ranked.columns: show_cols.append('zip_drive_time')
    for c in ["zip_desert_share","zip_desert_flag","zip_desert_flag_user","zip_desert_pop_pct","zip_alloc_coverage","zip_alloc_method"]:
        if c in ranked.columns: show_cols.append(c)
    for c in ['heat_hhb','edu_hs_or_lower_pct']: 
        if c in ranked.columns: show_cols.append(c)
    for c in ['scarcity','pop_norm','income_inv','health_n','score_math','ai_score','final_score']:
        if c in ranked.columns: show_cols.append(c)
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

    st.write("### Top 10 ZIPs on Interactive Map")
    render_top10_map(ranked.head(10).copy(), pharmacist_df=pharmacist_data)

    st.write("### Export Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Full Results CSV", ranked.to_csv(index=False), "pharmacy_desert_candidates_full.csv", "text/csv")
    with col2:
        st.download_button("Download Top 100 CSV", ranked.head(100).to_csv(index=False), "pharmacy_desert_top100.csv", "text/csv")

    st.divider()
    st.caption("🏥 Pharmacy Desert Explorer | Hybrid AI + Mathematical Approach")
    st.caption("Built with Streamlit | Data refreshed on page load")

if __name__ == "__main__":
    main()
