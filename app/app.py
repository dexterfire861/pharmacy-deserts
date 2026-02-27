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
import hashlib
import re
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
    load_pharmacy_data_only,
    get_glm_model_info,
    get_available_datasets,
    get_active_dataset_id,
    set_active_dataset_id,
    is_using_dataset_config,
    get_scoring_config_object,
)
from app.auth import login_form, logout_button, is_authenticated
from app.config import get_config
from models.scoring import (
    score_candidates, score_with_config, score_with_features, average_scores,
    get_available_weights_for_dataset, get_available_features_for_weighting
)
from models.schema import (
    ScoringConfig, SCORING_COMPONENTS, COMPONENT_CATEGORIES,
    get_default_scoring_config, ScoreDirection
)
from models.ai_scores import read_ifae_csv
from viz.map_viz import render_top10_map


def render_feature_weight_sliders(df: pd.DataFrame):
    """
    Render weight sliders for feature columns directly - modular approach.
    
    Shows sliders for all numeric features in the dataset.
    No component mapping needed - weights apply directly to features.
    
    Args:
        df: The loaded DataFrame
    
    Returns:
        Dict of feature_column_name -> weight value
    """
    st.sidebar.header("Feature Weights")
    
    # Get available numeric features (exclude metadata/geo columns)
    available_features = get_available_features_for_weighting(df)
    
    if not available_features:
        st.sidebar.info("No numeric features found for weighting.")
        return {}
    
    st.sidebar.markdown(f"*Adjust weights for {len(available_features)} features:*")
    
    weights = {}
    
    # Initialize weights from session state if available
    if 'feature_weights' not in st.session_state:
        st.session_state.feature_weights = {}
    
    # Render sliders for each feature
    for feature in available_features:
        # Get default weight from session state or use 0.0
        default_weight = st.session_state.feature_weights.get(feature, 0.0)
        
        # Show feature stats
        feature_mean = df[feature].mean()
        feature_min = df[feature].min()
        feature_max = df[feature].max()
        stats_text = f"Range: {feature_min:.1f} - {feature_max:.1f} (mean: {feature_mean:.1f})"
        
        weight = st.slider(
            f"**{feature}**",
            min_value=0.0,
            max_value=1.0,
            value=default_weight,
            step=0.05,
            key=f"feature_weight_{feature}",
            help=f"{stats_text}\n\nHigher weight = more influence on final score"
        )
        weights[feature] = weight
        st.session_state.feature_weights[feature] = weight
    
    # Show normalized weights summary
    total = sum(weights.values())
    if total > 0:
        st.sidebar.markdown("---")
        st.sidebar.caption("**Normalized Weights:**")
        for feature, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if w > 0:
                st.sidebar.caption(f"{feature}: {w/total:.1%}")
    else:
        st.sidebar.warning("⚠️ Set at least one weight > 0 to calculate scores")
    
    return weights


def render_dynamic_weight_sliders(df: pd.DataFrame, scoring_config: ScoringConfig):
    """
    Render weight sliders dynamically based on the scoring config (LEGACY).
    
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


def align_dataset_to_main_scoring_columns(df: pd.DataFrame, scoring_config: ScoringConfig):
    """
    Backfill only the app's fixed scoring inputs from mapped columns.

    This keeps the existing slider/scoring UX unchanged while allowing uploaded
    datasets with different column names to populate those same core inputs.
    """
    if df is None or df.empty or scoring_config is None:
        return df, 0

    component_to_main_col = {
        "pharmacy_count": "n_pharmacies",
        "population": "population",
        "income": "median_income",
        "health_burden": "health_burden",
        "pop_density": "pop_density",
        "education_low": "edu_hs_or_lower_pct",
        "drive_time": "zip_drive_time",
        "heat_vulnerability": "heat_hhb",
        "latitude": "lat",
        "longitude": "lon",
    }
    numeric_targets = {
        "n_pharmacies",
        "population",
        "median_income",
        "health_burden",
        "pop_density",
        "edu_hs_or_lower_pct",
        "zip_drive_time",
        "heat_hhb",
        "lat",
        "lon",
    }

    out = df.copy()
    applied = 0

    for mapping in scoring_config.column_mappings:
        src_col = mapping.source_column
        target_col = component_to_main_col.get(mapping.target_component)

        if not target_col or src_col not in out.columns:
            continue

        if target_col not in out.columns:
            out[target_col] = np.nan

        before_non_null = out[target_col].notna().sum()

        if target_col in numeric_targets:
            source_values = pd.to_numeric(out[src_col], errors="coerce")
            current_values = pd.to_numeric(out[target_col], errors="coerce")
            out[target_col] = current_values.where(current_values.notna(), source_values)
        else:
            out[target_col] = out[target_col].where(out[target_col].notna(), out[src_col])

        after_non_null = out[target_col].notna().sum()
        if after_non_null > before_non_null:
            applied += 1

    if "n_pharmacies" in out.columns:
        out["n_pharmacies"] = (
            pd.to_numeric(out["n_pharmacies"], errors="coerce").fillna(0).astype(int)
        )

    return out, applied


def _is_numeric_feature_candidate(values: pd.Series, min_valid_ratio: float = 0.50) -> bool:
    """Check if a column is numeric enough to be used as a weighted score feature."""
    if pd.api.types.is_numeric_dtype(values):
        return values.notna().any()

    numeric = pd.to_numeric(values, errors="coerce")
    observed = values.notna()
    observed_count = int(observed.sum())
    if observed_count == 0:
        return False
    valid_ratio = float(numeric[observed].notna().mean())
    return valid_ratio >= min_valid_ratio


def _format_feature_label(col_name: str) -> str:
    """Make custom feature column names easier to read in the UI."""
    revenue_kind = _classify_revenue_column(str(col_name or ""))
    if revenue_kind == "with_insurance":
        return "Revenue (With Insurance) ($/year)"
    if revenue_kind == "without_insurance":
        return "Revenue (Without Insurance) ($/year)"
    if revenue_kind == "with_cancer":
        return "Revenue (With Cancer) ($/year)"
    return col_name.replace("__", " • ")


def _normalize_col_key(col_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col_name or "").lower()).strip("_")


def _classify_revenue_column(col_name: str) -> str | None:
    """
    Classify a column as a revenue field.

    Returns one of:
    - with_insurance
    - without_insurance
    - with_cancer
    """
    normalized = _normalize_col_key(col_name)
    if not normalized:
        return None

    has_insurance = ("insurance" in normalized) or ("insurace" in normalized)

    if "revenue_with_insurance" in normalized:
        return "with_insurance"
    if "revenue_without_insurance" in normalized:
        return "without_insurance"
    if "revenue_with_cancer" in normalized:
        return "with_cancer"
    if "revenue_potential" in normalized:
        return "with_insurance"

    if "grand_total" in normalized and "without_cancer" in normalized:
        if has_insurance:
            return "with_insurance"
        return "without_insurance"
    if "grand_total" in normalized and "with_cancer" in normalized:
        return "with_cancer"

    # Fallback for user-renamed columns that still retain insurance keywords.
    if "revenue" in normalized and has_insurance:
        return "with_insurance"
    if "revenue" in normalized and "without" in normalized:
        return "without_insurance"
    return None


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
    
    # Initialize results_exist for GLM mode check
    results_exist = False
    last_modified = None
    latest_model_dataset_version = None
    
    try:
        from storage.models import get_model_registry
        model_registry = get_model_registry()
        model_versions = model_registry.list_model_versions()
        latest_version = model_registry.get_latest_version()
        
        if model_versions:
            results_exist = True
            # Show latest model info
            latest_model = next((m for m in model_versions if m['version_id'] == latest_version), None)
            if latest_model:
                trained_at = latest_model.get('trained_at', 'Unknown')
                if isinstance(trained_at, str) and 'T' in trained_at:
                    trained_at = trained_at.split('T')[0]
                row_count = latest_model.get('row_count', 'N/A')
                latest_model_dataset_version = latest_model.get('dataset_version')
                st.sidebar.success(f"✓ Active: `{latest_version}`")
                st.sidebar.caption(f"Trained: {trained_at} | {row_count:,} rows" if isinstance(row_count, int) else f"Trained: {trained_at}")
                if latest_model_dataset_version:
                    st.sidebar.caption(f"Dataset for model: `{latest_model_dataset_version}`")
            
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

    if (
        pharmacy_dataset
        and latest_model_dataset_version
        and pharmacy_dataset.get("version")
        and latest_model_dataset_version != pharmacy_dataset.get("version")
    ):
        st.sidebar.warning(
            "Model/data version mismatch: GLM uses an older dataset version than Math mode."
        )
    
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
        st.sidebar.caption("Uploaded custom feature columns are viewable in Math/Blended mode.")
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
            st.error("⚠️ GLM results not found.")
            st.info("""
            **To use GLM mode, you need to:**
            1. Upload data via the **Upload Data** page
            2. Train a model (run the training pipeline after upload)
            
            **Or switch to Math/Blended mode** which works with uploaded data directly.
            """)
            if st.button("📤 Go to Upload Data Page", type="primary"):
                st.switch_page("pages/Upload_Data.py")
            st.stop()
        
        # Load GLM results (cached)
        ranked = load_glm_results()
        
        if ranked.empty:
            st.error("Failed to load GLM results.")
            st.stop()
        
        # Merge lat/lon from the unified dataset if not present in GLM results
        if 'lat' not in ranked.columns or ranked['lat'].isna().all():
            latlon_df = load_latlon_lookup()
            if latlon_df.empty:
                # Direct fallback: read lat/lon from unified dataset on disk
                try:
                    import json as _json
                    _latest_path = Path('raw_data/datasets/pharmacy_data/LATEST.json')
                    if _latest_path.exists():
                        _ver = _json.loads(_latest_path.read_text()).get('latest_version')
                        if _ver:
                            _upath = Path(f'raw_data/datasets/pharmacy_data/versions/{_ver}/unified_dataset.csv')
                            if _upath.exists():
                                latlon_df = pd.read_csv(_upath, dtype={'zip': str}, usecols=['zip', 'lat', 'lon'])
                                latlon_df['zip'] = latlon_df['zip'].astype(str).str.zfill(5)
                                latlon_df = latlon_df.dropna(subset=['lat', 'lon']).drop_duplicates(subset=['zip'])
                except Exception:
                    pass
            if not latlon_df.empty:
                ranked = ranked.merge(latlon_df[['zip', 'lat', 'lon']], on='zip', how='left')
        
        # Load pharmacist data (needed for map popups)
        pharmacist_data = load_pharmacist_data_only()
        pharmacy_detail_data = load_pharmacy_data_only()
        
        # Set up variables for GLM mode
        df = None  # Not used in GLM Only mode
        scoring_config = None
        sort_col = 'final_score'
        
        st.success(f"✅ GLM Model loaded - {len(ranked):,} ZIPs analyzed (fast mode)")
        st.sidebar.success(f"✅ GLM Model (PURE) - {len(ranked):,} ZIPs")
        st.info(
            "GLM Only mode shows model-result columns. "
            "To inspect uploaded custom dataset columns, switch to Math Only or Blended mode."
        )
        
        # No weight sliders in GLM mode
        weights = {}
        gate_goodrx = False
        
    else:
        # =====================================================================
        # MATH / BLENDED MODE - Full dataset loading with flexible scoring
        # =====================================================================
        
        # Check if dataset exists before trying to load
        if not pharmacy_dataset:
            st.warning("⚠️ **No data uploaded yet**")
            st.info("""
            **To use Math/Blended mode, you need to upload data first:**
            
            1. Go to the **Upload Data** page
            2. Upload your data files (CSV, Excel, etc.)
            3. Configure column mappings and scoring components
            4. Return here to analyze the data
            
            The platform will automatically use your uploaded dataset.
            """)
            if st.button("📤 Go to Upload Data Page", type="primary"):
                st.switch_page("pages/Upload_Data.py")
            st.stop()
        
        try:
            # Use smart loader that returns (df, pharmacist_data, scoring_config_dict)
            # Pass the active dataset ID as argument for proper caching
            current_dataset_id = get_active_dataset_id()
            dataset_version_hint = pharmacy_dataset.get("version", "") if pharmacy_dataset else ""
            df, pharmacist_data, scoring_config_dict = load_smart_dataset_bundle(
                current_dataset_id, dataset_version_hint
            )
            pharmacy_detail_data = load_pharmacy_data_only()
            
            # Check if dataset is empty
            if df.empty or len(df) == 0:
                st.warning("⚠️ **Dataset is empty**")
                st.info("""
                The dataset exists but contains no data. Please:
                
                1. Go to the **Upload Data** page
                2. Upload data files with ZIP/ZCTA codes
                3. Configure the data mappings
                4. Return here to analyze
                """)
                if st.button("📤 Go to Upload Data Page", type="primary"):
                    st.switch_page("pages/Upload_Data.py")
                st.stop()
            
            # Convert scoring config dict back to object
            scoring_config = get_scoring_config_object(scoring_config_dict)
            df, backfilled_inputs = align_dataset_to_main_scoring_columns(df, scoring_config)
            
            pharm_count = len(pharmacist_data) if not pharmacist_data.empty else 0
            unique_pharm_zips = pharmacist_data['Short_ZIP'].nunique() if not pharmacist_data.empty else 0
            pharmacy_count = len(pharmacy_detail_data) if pharmacy_detail_data is not None and not pharmacy_detail_data.empty else 0
            unique_pharmacy_zips = (
                pharmacy_detail_data["Short_ZIP"].nunique()
                if pharmacy_detail_data is not None and not pharmacy_detail_data.empty and "Short_ZIP" in pharmacy_detail_data.columns
                else 0
            )
            st.success(
                f"Data loaded successfully! Analyzing {len(df):,} ZIP codes | "
                f"{pharm_count:,} pharmacist records from {unique_pharm_zips} ZIPs | "
                f"{pharmacy_count:,} pharmacy records from {unique_pharmacy_zips} ZIPs"
            )
            
            # Show scoring config info
            if is_using_dataset_config():
                mapped_count = len(scoring_config.column_mappings)
                st.info(f"🎯 Using curated scoring mappings ({mapped_count} mapped components)")
                if backfilled_inputs:
                    st.caption(
                        f"Mapped {backfilled_inputs} column(s) into the app's fixed scoring inputs."
                    )
                
        except (ValueError, FileNotFoundError) as e:
            # Dataset doesn't exist or can't be loaded
            st.warning("⚠️ **Could not load dataset**")
            st.info(f"""
            **Error:** {str(e)}
            
            **To fix this:**
            1. Go to the **Upload Data** page
            2. Upload your data files
            3. Ensure all files are properly configured
            4. Return here to analyze
            """)
            if st.button("📤 Go to Upload Data Page", type="primary"):
                st.switch_page("pages/Upload_Data.py")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
            st.info("""
            **If this persists:**
            1. Check that data was uploaded correctly
            2. Verify file formats and column mappings
            3. Try refreshing the data cache
            """)
            st.stop()

        # =====================================================================
        # SCORING WEIGHT SLIDERS
        # =====================================================================
        st.sidebar.header("Model Weights")
        st.sidebar.markdown("*Adjust scoring priorities:*")

        w_scarcity = st.sidebar.slider(
            "Pharmacy Scarcity", 0.0, 1.0, 0.30, 0.05,
            key="w_scarcity",
            help="Weight for 1/(1+n_pharmacies). Fewer pharmacies → higher need.",
        )
        w_health = st.sidebar.slider(
            "Health Burden", 0.0, 1.0, 0.25, 0.05,
            key="w_health",
            help="Weight for poor-health prevalence (PLACES GHLTH).",
        )
        w_income = st.sidebar.slider(
            "Income (inverted)", 0.0, 1.0, 0.20, 0.05,
            key="w_income",
            help="Lower median income → higher need score.",
        )
        w_pop = st.sidebar.slider(
            "Population Density", 0.0, 1.0, 0.10, 0.05,
            key="w_pop",
            help="Denser areas with gaps = more affected people.",
        )

        # Optional components — only show if data is present
        w_edu = 0.0
        if "edu_hs_or_lower_pct" in df.columns:
            w_edu = st.sidebar.slider(
                "Low Education %", 0.0, 1.0, 0.05, 0.05,
                key="w_edu",
                help="% with HS diploma or lower (health-literacy proxy).",
            )

        w_drive_time = 0.0
        if "zip_drive_time" in df.columns and df["zip_drive_time"].notna().any():
            w_drive_time = st.sidebar.slider(
                "Drive Time", 0.0, 1.0, 0.10, 0.05,
                key="w_drive_time",
                help="Average drive time to nearest pharmacy (county-downscaled).",
            )

        w_heat = 0.0
        if "heat_hhb" in df.columns:
            w_heat = st.sidebar.slider(
                "Heat Vulnerability", 0.0, 1.0, 0.05, 0.05,
                key="w_heat",
                help="Heat-health burden index (HHI).",
            )

        # Automatically add uploaded custom numeric features as optional
        # scoring inputs with their own sliders.
        extra_feature_defaults = (
            scoring_config.custom_feature_weights
            if scoring_config and hasattr(scoring_config, "custom_feature_weights")
            else {}
        ) or {}
        core_scoring_columns = {
            "zip", "n_pharmacies", "population", "median_income", "health_burden",
            "pop_density", "edu_hs_or_lower_pct", "zip_drive_time", "heat_hhb",
            "lat", "lon", "score", "final_score", "ai_score", "desert_flag",
            "pharm_per_10k", "zip_desert_flag", "zip_desert_flag_user",
        }

        extra_feature_candidates: list[str] = []
        for col in sorted(extra_feature_defaults.keys()):
            if col in df.columns and col not in core_scoring_columns and _is_numeric_feature_candidate(df[col]):
                extra_feature_candidates.append(col)

        # Fallback for older configs: auto-detect prefixed custom columns.
        for col in sorted(c for c in df.columns if "__" in c):
            if (
                col not in core_scoring_columns
                and col not in extra_feature_candidates
                and _is_numeric_feature_candidate(df[col])
            ):
                extra_feature_candidates.append(col)

        extra_feature_weights: dict[str, float] = {}
        if extra_feature_candidates:
            st.sidebar.markdown("**Additional Uploaded Features**")
            st.sidebar.caption(
                "New numeric custom fields from uploaded datasets can be weighted here."
            )
            dataset_version_for_keys = (
                pharmacy_dataset.get("version", "none") if pharmacy_dataset else "none"
            )
            for col in extra_feature_candidates:
                default_weight = float(extra_feature_defaults.get(col, 0.05))
                default_weight = min(1.0, max(0.0, default_weight))
                slider_key = (
                    "w_extra_"
                    + hashlib.md5(f"{dataset_version_for_keys}:{col}".encode("utf-8")).hexdigest()[:12]
                )
                extra_feature_weights[col] = st.sidebar.slider(
                    _format_feature_label(col),
                    0.0,
                    1.0,
                    default_weight,
                    0.05,
                    key=slider_key,
                    help=f"Additional uploaded feature: `{col}`",
                )

        weights = {
            "scarcity": w_scarcity, "health": w_health, "income": w_income,
            "pop": w_pop, "edu": w_edu, "drive_time": w_drive_time, "heat": w_heat,
        }
        for col, weight in extra_feature_weights.items():
            weights[f"feature::{col}"] = weight
        st.sidebar.divider()

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
        # SCORING (Math Only or Blended)
        # =====================================================================
        ranked = score_candidates(
            df, w_scarcity, w_health, w_income, w_pop,
            w_heat=w_heat, w_edu=w_edu, w_drive_time=w_drive_time,
            extra_feature_weights=extra_feature_weights,
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
        active_weights = {k: v for k, v in weights.items() if v > 0}
        total_weight = sum(active_weights.values())

        if active_weights and total_weight > 0:
            st.divider()
            st.write("#### Weight Distribution")
            import plotly.graph_objects as go

            display_names = {
                "scarcity": "Pharmacy Scarcity", "health": "Health Burden",
                "income": "Income (inv)", "pop": "Pop Density",
                "edu": "Low Education", "drive_time": "Drive Time",
                "heat": "Heat Vulnerability",
            }
            names = []
            for key in active_weights:
                if key.startswith("feature::"):
                    names.append(_format_feature_label(key.split("feature::", 1)[1]))
                else:
                    names.append(display_names.get(key, key))
            values = [v / total_weight for v in active_weights.values()]

            fig = go.Figure(data=[go.Bar(
                x=names, y=values,
                text=[f'{v:.1%}' for v in values],
                textposition='auto',
            )])
            fig.update_layout(
                title="Current Weight Distribution (Normalized)",
                yaxis_title="Weight",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
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

    # Revenue metrics from uploaded health economics files (with/without insurance).
    revenue_by_kind: dict[str, list[str]] = {
        "with_insurance": [],
        "without_insurance": [],
        "with_cancer": [],
    }
    for col in ranked.columns:
        revenue_kind = _classify_revenue_column(col)
        if revenue_kind in revenue_by_kind:
            revenue_by_kind[revenue_kind].append(col)

    for revenue_kind in ["with_insurance", "without_insurance", "with_cancer"]:
        candidates = revenue_by_kind[revenue_kind]
        if not candidates:
            continue
        preferred = sorted(
            candidates,
            key=lambda c: (
                "revenue_potential" not in _normalize_col_key(c),
                "revenue_with_insurance" not in _normalize_col_key(c),
                "revenue_without_insurance" not in _normalize_col_key(c),
                "revenue_with_cancer" not in _normalize_col_key(c),
                len(c),
            ),
        )[0]
        if preferred not in show_cols:
            show_cols.append(preferred)
    
    # Score columns
    for c in ['score', 'score_math', 'ai_score', 'final_score', 'desert_flag']:
        if c in ranked.columns and c not in show_cols:
            show_cols.append(c)
    
    show_cols = [c for c in show_cols if c in ranked.columns]

    # Auto-include a limited set of uploaded custom columns by default.
    auto_extra_limit = 6
    remaining_cols = [c for c in ranked.columns if c not in show_cols]
    auto_custom_cols = [c for c in remaining_cols if "__" in c][:auto_extra_limit]

    hidden_cols = [c for c in ranked.columns if c not in show_cols]
    extra_cols: list[str] = []
    with st.expander("Customize Displayed Columns", expanded=False):
        st.caption(
            f"Showing {len(show_cols)} default columns. "
            f"{len(hidden_cols)} additional columns are available."
        )
        if hidden_cols:
            picker_key = (
                f"extra_table_cols_{scoring_mode}_"
                f"{pharmacy_dataset.get('version', 'none') if pharmacy_dataset else 'none'}"
            )
            extra_cols = st.multiselect(
                "Add columns to the table",
                options=hidden_cols,
                default=[c for c in auto_custom_cols if c in hidden_cols],
                key=picker_key,
                help=(
                    "Use this to display custom uploaded fields (for example, "
                    "columns prefixed like houseprice__* or insurance__*)."
                ),
            )
        else:
            st.caption("No additional columns available for this view.")

    table_cols = show_cols + [c for c in extra_cols if c in ranked.columns and c not in show_cols]

    with st.expander("Understanding the Scores", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scores**: `score` (weighted math), `ai_score` (GLM), `final_score` (blend), `desert_flag` (zero-pharmacy indicator).")
            st.markdown("**Scarcity**: 1/(1+n_pharmacies) — fewer pharmacies = higher scarcity score.")
        with c2:
            st.markdown("**Scoring**: Each component is normalized [0,1], multiplied by its weight, and the sum is rescaled to [0,1]. Higher = more need.")
            st.markdown("**Desert Flag**: ZIPs with **zero** pharmacies are flagged.")

    st.dataframe(ranked[table_cols].head(50), use_container_width=True, height=400)

    # Interactive Map
    st.write("### Top 10 ZIPs on Interactive Map")
    map_key = f"map_{scoring_mode}_{len(ranked)}"
    if map_key not in st.session_state:
        st.session_state[map_key] = True

    render_top10_map(
        ranked.head(10).copy(),
        pharmacist_df=pharmacist_data,
        pharmacy_df=pharmacy_detail_data,
    )

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
