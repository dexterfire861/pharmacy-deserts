# pharmacy_deserts/app/app.py
"""
Pharmacy Desert Explorer - Streamlit Application

This app provides fast startup by only loading data needed for the selected mode:
- GLM Only: Loads pre-computed GLM results (fast, no network calls)
- Math Only / Blended: Loads full dataset bundle (slower, includes ACS API call)

Supports flexible scoring configurations from uploaded datasets.
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
    load_smart_dataset_bundle,
    load_glm_results,
    load_latlon_lookup,
    load_pharmacist_data_only,
    get_glm_model_info,
    get_available_datasets,
    get_active_dataset_id,
    set_active_dataset_id,
    is_using_dataset_config,
    get_scoring_config_object,
)
from app.auth import login_form, logout_button, is_authenticated
from app.config import get_config
from models.scoring import score_with_config, average_scores, get_available_weights_for_dataset
from models.schema import (
    ScoringConfig, SCORING_COMPONENTS, COMPONENT_CATEGORIES,
    get_default_scoring_config, ScoreDirection
)
from models.ai_scores import read_ifae_csv
from viz.map_viz import render_top10_map


def render_dynamic_weight_sliders(df: pd.DataFrame, scoring_config: ScoringConfig):
    """
    Render weight sliders dynamically based on the scoring config.
    
    Only shows sliders for components that are mapped in the config
    AND have corresponding columns in the data.
    
    Args:
        df: The loaded DataFrame
        scoring_config: The scoring configuration
    
    Returns:
        Dict of component_name -> weight value
    """
    st.sidebar.header("Model Weights")
    
    weights = {}
    
    # Get available components from the config
    available = get_available_weights_for_dataset(df, scoring_config)
    
    if not available:
        st.sidebar.info("No scoring components mapped for this dataset.")
        st.sidebar.caption("Upload data with scoring mappings to enable weighted scoring.")
        return weights
    
    st.sidebar.markdown("*Adjust priorities:*")
    
    # Group by category
    by_category = {}
    for comp_name, (display_name, default_weight) in available.items():
        comp = SCORING_COMPONENTS.get(comp_name)
        if comp:
            cat = comp.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((comp_name, display_name, default_weight, comp))
    
    # Render sliders by category
    for category, components in by_category.items():
        category_label = COMPONENT_CATEGORIES.get(category, category.title())
        
        with st.sidebar.expander(f"**{category_label}**", expanded=True):
            for comp_name, display_name, default_weight, comp in components:
                # Show direction indicator
                direction = "↑=worse" if comp.direction == ScoreDirection.HIGHER_IS_WORSE else "↑=better"
                
                weight = st.slider(
                    f"{display_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.05,
                    key=f"weight_{comp_name}",
                    help=f"{comp.description}\n\n({direction})"
                )
                weights[comp_name] = weight
    
    # Show normalized weights summary
    total = sum(weights.values())
    if total > 0:
        st.sidebar.markdown("---")
        st.sidebar.caption("**Normalized Weights:**")
        for comp_name, w in weights.items():
            if w > 0:
                display_name = SCORING_COMPONENTS.get(comp_name, {})
                name = display_name.display_name if hasattr(display_name, 'display_name') else comp_name
                st.sidebar.caption(f"{name}: {w/total:.1%}")
    
    return weights


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

    # =========================================================================
    # DATA STATUS (shows what data sources are loaded)
    # =========================================================================
    # Always use the default combined dataset "pharmacy_data"
    DEFAULT_DATASET_ID = "pharmacy_data"
    set_active_dataset_id(DEFAULT_DATASET_ID)
    
    # Show data status in sidebar
    st.sidebar.header("📊 Data Status")
    available_datasets = get_available_datasets()
    
    # Check if we have any uploaded data
    pharmacy_dataset = next((d for d in available_datasets if d.get('dataset_id') == DEFAULT_DATASET_ID), None)
    if pharmacy_dataset:
        source_count = pharmacy_dataset.get('source_count', 0)
        version = pharmacy_dataset.get('version', '')
        description = pharmacy_dataset.get('description', '')
        
        st.sidebar.success(f"✓ {source_count} data source(s) loaded")
        if version:
            st.sidebar.caption(f"Version: `{version}`")
        if description:
            st.sidebar.caption(f"_{description}_")
        st.sidebar.caption("Upload more data via **Upload Data** page")
    else:
        st.sidebar.warning("⚠️ No data uploaded yet")
        st.sidebar.caption("Go to **Upload Data** page to add data files")
    
    st.sidebar.divider()
    
    # Model Registry Section
    st.sidebar.header("🧠 Model Registry")
    
    try:
        from storage.models import get_model_registry
        model_registry = get_model_registry()
        model_versions = model_registry.list_model_versions()
        latest_version = model_registry.get_latest_version()
        
        if model_versions:
            # Show latest model info
            latest_model = next((m for m in model_versions if m['version_id'] == latest_version), None)
            if latest_model:
                trained_at = latest_model.get('trained_at', 'Unknown')
                if isinstance(trained_at, str) and 'T' in trained_at:
                    trained_at = trained_at.split('T')[0]
                row_count = latest_model.get('row_count', 'N/A')
                st.sidebar.success(f"✓ Active: `{latest_version}`")
                st.sidebar.caption(f"Trained: {trained_at} | {row_count:,} rows" if isinstance(row_count, int) else f"Trained: {trained_at}")
            
            # Model selector (for future multi-model support)
            if len(model_versions) > 1:
                with st.sidebar.expander(f"📜 {len(model_versions)} model versions"):
                    for mv in model_versions[:5]:  # Show top 5
                        ver = mv['version_id']
                        is_latest = "✓" if ver == latest_version else ""
                        st.caption(f"{is_latest} {ver}")
        else:
            # Fall back to checking for legacy results
            results_exist, last_modified = get_glm_model_info()
            if results_exist:
                st.sidebar.info(f"📊 Legacy model: {last_modified.strftime('%m/%d/%y %H:%M')}")
                st.sidebar.caption("Upload new data to train versioned model")
            else:
                st.sidebar.warning("⚠️ No trained model found")
                st.sidebar.caption("Upload data to train a model")
                
    except Exception as e:
        # Fallback to legacy check
        results_exist, last_modified = get_glm_model_info()
        if results_exist:
            st.sidebar.info(f"📊 Model trained: {last_modified.strftime('%m/%d/%y %H:%M')}")
        else:
            st.sidebar.warning("⚠️ No trained model found")
    
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
        scoring_config = None
        sort_col = 'final_score'
        
        st.success(f"✅ GLM Model loaded - {len(ranked):,} ZIPs analyzed (fast mode)")
        st.sidebar.success(f"✅ GLM Model (PURE) - {len(ranked):,} ZIPs")
        
        # No weight sliders in GLM mode
        weights = {}
        gate_goodrx = False
        
    else:
        # =====================================================================
        # MATH / BLENDED MODE - Full dataset loading with flexible scoring
        # =====================================================================
        try:
            # Use smart loader that returns (df, pharmacist_data, scoring_config_dict)
            # Pass the active dataset ID as argument for proper caching
            current_dataset_id = get_active_dataset_id()
            df, pharmacist_data, scoring_config_dict = load_smart_dataset_bundle(current_dataset_id)
            
            # Convert scoring config dict back to object
            scoring_config = get_scoring_config_object(scoring_config_dict)
            
            pharm_count = len(pharmacist_data) if not pharmacist_data.empty else 0
            unique_pharm_zips = pharmacist_data['Short_ZIP'].nunique() if not pharmacist_data.empty else 0
            st.success(f"Data loaded successfully! Analyzing {len(df):,} ZIP codes | {pharm_count:,} pharmacist records from {unique_pharm_zips} ZIPs")
            
            # Show scoring config info
            if is_using_dataset_config():
                mapped_count = len(scoring_config.column_mappings)
                st.info(f"🎯 Using custom scoring config with {mapped_count} mapped components")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

        # =====================================================================
        # DYNAMIC WEIGHT SLIDERS (based on scoring config)
        # =====================================================================
        weights = render_dynamic_weight_sliders(df, scoring_config)

        # Optional: GoodRx hard gate controls (only if those columns exist)
        if "zip_desert_share" in df.columns:
            st.sidebar.header("GoodRx Desert Gate")
            gate_goodrx = st.sidebar.checkbox(
                "Hard gate to GoodRx-defined deserts", 
                value=False,
                help="When ON, only ZIPs that are GoodRx drive-time deserts are kept."
            )
            min_cov = st.sidebar.slider("Minimum crosswalk coverage (HUD)", 0.0, 1.0, 0.60, 0.05)
            thr_goodrx = st.sidebar.slider("Desert severity threshold", 0.0, 1.0, 0.50, 0.05)
            
            # Apply desert flag
            df["zip_desert_flag_user"] = (df["zip_desert_share"] >= thr_goodrx).astype("Int64")
            
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
        else:
            gate_goodrx = False

        # Population/Urban filters (if columns exist)
        if 'pop_density' in df.columns or 'population' in df.columns:
            st.sidebar.header("🏙️ Target Area Filters")
            st.sidebar.markdown("*Focus on semi-urban communities:*")

            min_population = st.sidebar.slider(
                "Minimum population", 0, 50000, 5000, 1000,
                help="Exclude very small ZIPs"
            ) if 'population' in df.columns else 0
            
            min_density = st.sidebar.slider(
                "Minimum density (people/km²)", 0, 1000, 100, 50,
                help="100-400 = semi-urban sweet spot"
            ) if 'pop_density' in df.columns else 0
            
            max_density = st.sidebar.slider(
                "Maximum density (people/km²)", 0, 10000, 5000, 500,
                help="Exclude extremely dense urban cores if desired. 0 = no max"
            ) if 'pop_density' in df.columns else 0

            # Apply filters
            filters_applied = []
            df_before_filters = len(df)

            if min_population > 0 and 'population' in df.columns:
                df = df[df['population'].fillna(0) >= min_population]
                filters_applied.append(f"pop ≥ {min_population:,}")
            if min_density > 0 and 'pop_density' in df.columns:
                df = df[df['pop_density'].fillna(0) >= min_density]
                filters_applied.append(f"density ≥ {min_density}")
            if max_density > 0 and 'pop_density' in df.columns:
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
        # SCORING (Math Only or Blended) - USING FLEXIBLE SCORER
        # =====================================================================
        ranked = score_with_config(df, scoring_config, weights)
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
    
    # Ensure desert_flag exists
    if 'desert_flag' not in ranked.columns:
        ranked['desert_flag'] = 0
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
    if scoring_mode != "GLM Only" and weights:
        st.divider()
        st.write("#### Weight Distribution")
        import plotly.graph_objects as go
        
        # Build chart data from weights
        names = []
        values = []
        total_weight = sum(weights.values())
        
        for comp_name, w in weights.items():
            if w > 0:
                comp = SCORING_COMPONENTS.get(comp_name)
                display_name = comp.display_name if comp else comp_name
                names.append(display_name)
                values.append(w / total_weight if total_weight > 0 else 0)
        
        if names:
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

    # Build show columns dynamically based on what exists
    show_cols = ['zip']
    
    # Core metrics
    for c in ['population', 'n_pharmacies', 'pharm_per_10k', 'pop_density']:
        if c in ranked.columns:
            show_cols.append(c)
    
    # Income/health columns (may have different names based on dataset)
    for c in ['median_income', 'income', 'health_burden', 'health_poor_pct']:
        if c in ranked.columns and c not in show_cols:
            show_cols.append(c)
    
    # Optional columns
    optional_cols = [
        'zip_drive_time', 'drive_time',
        'zip_desert_share', 'zip_desert_flag', 'zip_desert_flag_user', 
        'zip_desert_pop_pct', 'zip_alloc_coverage', 'zip_alloc_method',
        'heat_hhb', 'heat_vulnerability',
        'edu_hs_or_lower_pct', 'education_low',
        'age_elderly_pct', 'uninsured_pct', 'vehicle_access'
    ]
    for c in optional_cols:
        if c in ranked.columns and c not in show_cols:
            show_cols.append(c)
    
    # Score columns
    for c in ['score', 'score_math', 'ai_score', 'final_score', 'desert_flag']:
        if c in ranked.columns and c not in show_cols:
            show_cols.append(c)
    
    show_cols = [c for c in show_cols if c in ranked.columns]

    with st.expander("Understanding the Scores", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scores**: `score` (weighted), `ai_score` (GLM), `final_score` (blend), `desert_flag` (pharmacy need indicator).")
            st.markdown("**Key Metrics**: `pharm_per_10k` (pharmacies per 10K people), `population` (total ZIP population)")
        with c2:
            st.markdown("**Scoring**: Components are normalized to [0,1] and weighted. Higher scores = more need for pharmacy access.")
            st.markdown("**Desert Flag**: ZIPs with <2 pharmacies per 10K people are flagged as high-need areas.")

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
