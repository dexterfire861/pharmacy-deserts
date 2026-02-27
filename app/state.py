# pharmacy_deserts/app/state.py
"""
Cached data loading functions for the Streamlit app.
These functions ensure heavy data loading only happens once per session
and only when needed for the selected scoring mode.

Supports both local filesystem (development) and S3 (production) data sources.
Also supports loading from dataset configurations when ACTIVE_DATASET_ID is set.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import io
import logging

logger = logging.getLogger(__name__)

# Add parent directory to sys.path for imports
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.config import get_config
from data.loaders import (
    read_financial_data, read_health_data, read_pharmacy_data, read_population_data,
    read_hhi_excel, read_population_labels,
    read_education_data_acs, read_hud_zip_county_crosswalk, read_county_desert_csv,
    downscale_county_to_zip, load_all_pharmacist_data
)
from data.features import preprocess


def get_active_dataset_id() -> str:
    """
    Get the active dataset ID from environment or session state.
    
    Checks in order:
    1. Streamlit session state (for UI-selected datasets)
    2. ACTIVE_DATASET_ID environment variable
    
    Returns:
        Dataset ID or empty string if none set
    """
    # Check session state first (allows UI override)
    if 'active_dataset_id' in st.session_state and st.session_state.active_dataset_id:
        return st.session_state.active_dataset_id
    
    # Fall back to environment variable
    return os.getenv('ACTIVE_DATASET_ID', '')


def set_active_dataset_id(dataset_id: str):
    """Set the active dataset ID in session state."""
    st.session_state.active_dataset_id = dataset_id


def is_using_dataset_config() -> bool:
    """Check if we should load from dataset config instead of default files."""
    return bool(get_active_dataset_id())


def _get_data_path(relative_path: str) -> str:
    """
    Get the appropriate data path based on environment.
    
    In production with S3 configured, returns S3 URI.
    In development, returns local path.
    """
    config = get_config()
    return config.get_data_path(relative_path)


def _is_s3_environment() -> bool:
    """Check if we're running in S3/production mode."""
    config = get_config()
    return config.is_production and config.aws_s3_bucket is not None


@st.cache_data(show_spinner="Loading math model datasets...")
def load_math_dataset_bundle():
    """
    Load and merge all datasets needed for Math and Blended scoring modes.
    
    Automatically loads from S3 in production or local filesystem in development.
    
    This includes:
    - Financial data (income)
    - Health data (health burden)
    - Pharmacy locations
    - Population density
    - Heat-Health Index (HHI)
    - Education data (ACS API call)
    - HUD ZIP/County crosswalk
    - County desert data
    - Pharmacist data
    
    Returns:
        tuple: (merged_df, pharmacist_data)
    """
    config = get_config()
    
    if _is_s3_environment():
        # Production mode: Load from S3
        from data.s3_loaders import (
            smart_read_csv, smart_read_excel, smart_load_directory
        )
        
        # Load base datasets from S3
        financial_path = config.get_financial_data_path()
        health_path = config.get_health_data_path()
        pharmacy_path = config.get_pharmacy_data_path()
        population_path = config.get_population_data_path()
        hhi_path = config.get_hhi_data_path()
        hud_path = config.get_hud_crosswalk_path()
        county_path = config.get_county_desert_path()
        
        # For S3, we need to use smart loaders that handle S3 paths
        # Note: The existing loaders need to be updated to use smart_read_* functions
        # For now, we download to temp and use existing loaders
        import tempfile
        from data.s3_loaders import download_s3_file_to_memory, parse_s3_path, download_s3_directory
        import io
        
        def load_csv_from_s3_path(s3_path):
            bucket, key = parse_s3_path(s3_path)
            data = download_s3_file_to_memory(bucket, key)
            return pd.read_csv(io.BytesIO(data))
        
        def load_excel_from_s3_path(s3_path, **kwargs):
            bucket, key = parse_s3_path(s3_path)
            data = download_s3_file_to_memory(bucket, key)
            return pd.read_excel(io.BytesIO(data), **kwargs)
        
        # Load financial data
        fin_df = load_csv_from_s3_path(financial_path)
        fin_df = fin_df[['NAME', 'S1901_C01_012E']]
        fin_df['zip'] = fin_df['NAME'].str.extract(r'(\d{5})')
        financial_data = fin_df
        
        # Load health data
        health_df = load_csv_from_s3_path(health_path)
        health_df = health_df[['ZCTA5', 'GHLTH_CrudePrev']]
        health_df['ZCTA5'] = health_df['ZCTA5'].astype(str).str.split('.').str[0].str.zfill(5)
        health_data = health_df
        
        # Load population data
        pop_df = pd.read_csv(
            io.BytesIO(download_s3_file_to_memory(*parse_s3_path(population_path))),
            skiprows=10
        )
        pop_df.columns = [str(c).strip() for c in pop_df.columns]
        lower = {c.lower(): c for c in pop_df.columns}
        population_data = pd.DataFrame({
            "zip": pop_df[lower["zip"]].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
            "population": pd.to_numeric(pop_df[lower["population"]].astype(str).str.replace(",", "", regex=False), errors="coerce"),
            "pop_density": pd.to_numeric(pop_df[lower["density"]].astype(str).str.replace(",", "", regex=False), errors="coerce"),
            "lat": pd.to_numeric(pop_df[lower["lat"]], errors="coerce"),
            "lon": pd.to_numeric(pop_df[lower["long"]], errors="coerce"),
        }).dropna(subset=["zip"]).groupby("zip", as_index=False).agg({
            "population": "sum", "pop_density": "max", "lat": "first", "lon": "first"
        })
        
        # Load HHI data
        hhi_df = load_excel_from_s3_path(hhi_path, dtype={'ZCTA': str})
        hhi_df['zip'] = hhi_df['ZCTA'].astype(str).str.extract(r'(\d{5})')[0].fillna('').str.zfill(5)
        hhi = pd.DataFrame({'zip': hhi_df['zip']})
        if 'HHB_SCORE' in hhi_df.columns:
            hhi['heat_hhb'] = pd.to_numeric(hhi_df['HHB_SCORE'], errors='coerce')
        if 'NBE_SCORE' in hhi_df.columns:
            hhi['nbe_score'] = pd.to_numeric(hhi_df['NBE_SCORE'], errors='coerce')
        if 'OVERALL_SCORE' in hhi_df.columns:
            hhi['hhi_overall'] = pd.to_numeric(hhi_df['OVERALL_SCORE'], errors='coerce')
        hhi = hhi.dropna(subset=['zip']).drop_duplicates(subset=['zip'])
        
        # Load pharmacy data - need to download directory
        bucket, key = parse_s3_path(pharmacy_path)
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='pharmacy_data_')
        local_pharmacy_path = download_s3_directory(bucket, key.rstrip('/') + '/', temp_dir)
        pharmacy_data = read_pharmacy_data(local_pharmacy_path)
        
        # Load HUD crosswalk
        hud_df = load_excel_from_s3_path(hud_path, dtype=str)
        cols = {c.lower(): c for c in hud_df.columns}
        zip_col = cols.get("zip") or cols.get("zipcode")
        county_col = cols.get("county") or cols.get("county_fips")
        weight_col = next((cols[c] for c in ["tot_ratio", "total_ratio", "res_ratio"] if c in cols), None)
        state_col = cols.get("state") or cols.get("stabbr")
        hud_xwalk = pd.DataFrame({
            "zip": hud_df[zip_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
            "county": hud_df[county_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
            "state": hud_df[state_col] if state_col else pd.Series(dtype="object"),
            "weight": pd.to_numeric(hud_df[weight_col], errors="coerce").fillna(0.0)
        }).dropna(subset=["zip", "county"])
        hud_xwalk = hud_xwalk[hud_xwalk["weight"] > 0]
        
        # Load county desert data
        county_df = read_county_desert_csv(county_path) if not county_path.startswith('s3://') else _load_county_desert_from_s3(county_path)
        
        # Load pharmacist data
        pharmacist_bucket, pharmacist_key = parse_s3_path(config.get_data_path(config.data_dir))
        temp_pharmacist_dir = tempfile.mkdtemp(prefix='pharmacist_data_')
        # Just use local loader for pharmacist data - it's in the data_dir
        pharmacist_data = load_all_pharmacist_data(temp_dir)  # Use same temp dir as pharmacy
        
    else:
        # Development mode: Load from local filesystem
        financial_data = read_financial_data('raw_data/financial_data.csv')
        health_data = read_health_data('raw_data/health_data.csv')
        pharmacy_data = read_pharmacy_data('raw_data/Pharmacy_list_ZIP_fixed_final')
        population_data = read_population_data('raw_data/population_data.csv')
        hhi = read_hhi_excel('raw_data/HHI_data.xlsx')
        hud_xwalk = read_hud_zip_county_crosswalk("raw_data/zip_county_cross.xlsx")
        pharmacist_data = load_all_pharmacist_data('raw_data')
    
    # Education data - requires network call to Census API (same for both modes)
    education = read_education_data_acs(year=2023)
    
    # Load county desert data (local)
    if not _is_s3_environment():
        county_df = read_county_desert_csv("raw_data/driving-time-desert.csv")
    else:
        county_df = _load_county_desert_from_s3(config.get_county_desert_path())
    
    # Downscale county to ZIP
    zip_desert_df = downscale_county_to_zip(
        county_df, hud_xwalk, 
        tiny_cutoff=0.01, min_coverage=0.60, threshold=0.50
    )
    
    # Preprocess and merge
    df = preprocess(financial_data, health_data, pharmacy_data, population_data, hhi=hhi)
    df = df.merge(education[["zip", "edu_hs_or_lower_pct"]], on='zip', how='left')
    df = df.merge(zip_desert_df, on='zip', how='left')
    
    return df, pharmacist_data


def _load_county_desert_from_s3(s3_path: str) -> pd.DataFrame:
    """Load county desert data from S3."""
    from data.s3_loaders import parse_s3_path, download_s3_file_to_memory
    import io
    
    bucket, key = parse_s3_path(s3_path)
    data = download_s3_file_to_memory(bucket, key)
    df = pd.read_csv(io.BytesIO(data), dtype=str, low_memory=False)
    
    fips_col = next((c for c in df.columns if "fips" in c.lower()), None)
    if not fips_col:
        raise ValueError("County dataset must include a county FIPS column.")
    df["county"] = df[fips_col].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
    
    flag_col = next((c for c in df.columns if c.lower() in ["desert", "is_desert", "desert_flag", "model1_pharm_desert", "pharm_desert"]), None)
    if flag_col:
        val = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        score_col = next((c for c in df.columns if any(k in c.lower() for k in ["score", "index", "risk", "prob"])), None)
        if not score_col:
            raise ValueError("No desert flag/score column found.")
        raw = pd.to_numeric(df[score_col], errors="coerce")
        val = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else 0.0
    
    drive_time_col = next((c for c in df.columns if "drive_time" in c.lower() and "min" in c.lower()), None)
    pop_pct_col = next((c for c in df.columns if "desert_pop" in c.lower() and "pct" in c.lower()), None)
    
    out = pd.DataFrame({
        "county": df["county"],
        "county_desert": val,
        "drive_time_min": pd.to_numeric(df[drive_time_col], errors="coerce") if drive_time_col else pd.Series(dtype=float),
        "desert_pop_pct": pd.to_numeric(df[pop_pct_col], errors="coerce") if pop_pct_col else pd.Series(dtype=float)
    })
    return out.dropna(subset=["county"]).drop_duplicates(subset=["county"])


@st.cache_data(show_spinner="Loading GLM model results...")
def load_glm_results():
    """
    Load GLM/IFAE results from CSV and return a standardized ranked DataFrame.
    
    Automatically loads from S3 in production or local filesystem in development.
    
    This is a lightweight load for GLM Only mode - no heavy processing,
    no network calls, just reads the pre-computed results.
    
    Returns:
        DataFrame with columns: zip, ai_score, population, pharmacies_count, 
        pop_density, median_income, poor_health_pct, n_pharmacies, pharm_per_10k,
        final_score, desert_flag, and optionally lat/lon if available in results.
    """
    config = get_config()
    
    if _is_s3_environment():
        from data.s3_loaders import parse_s3_path, download_s3_file_to_memory
        import io
        
        s3_path = config.get_glm_results_path()
        try:
            bucket, key = parse_s3_path(s3_path)
            data = download_s3_file_to_memory(bucket, key)
            glm_full = pd.read_csv(io.BytesIO(data), dtype={"ZCTA5": str}, low_memory=False)
        except Exception as e:
            st.warning(f"Could not load GLM results from S3: {e}")
            return pd.DataFrame()
    else:
        ai_file_path = "results/national_ifae_rank.csv"
        if not Path(ai_file_path).exists():
            return pd.DataFrame()
        glm_full = pd.read_csv(ai_file_path, dtype={"ZCTA5": str}, low_memory=False)
    
    glm_full["zip"] = glm_full["ZCTA5"].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5)
    glm_full["ai_score"] = pd.to_numeric(glm_full["IFAE_score"], errors="coerce")
    
    # Select columns to keep
    cols_to_load = ['zip', 'ai_score', 'ZCTA5', 'population', 'pharmacies_count',
                    'median_income', 'poor_health_pct', 'pop_density']
    
    if 'REGION' in glm_full.columns:
        cols_to_load.append('REGION')
    if 'LON' in glm_full.columns and 'LAT' in glm_full.columns:
        cols_to_load.extend(['LON', 'LAT'])
    
    cols_to_load = [c for c in cols_to_load if c in glm_full.columns]
    ranked = glm_full[cols_to_load].copy()
    
    # Standardize the DataFrame
    ranked['final_score'] = ranked['ai_score']
    ranked['score'] = np.nan
    ranked['desert_flag'] = (ranked['pharmacies_count'] < 5).astype(int)
    
    if 'median_income' in ranked.columns:
        ranked['income'] = ranked['median_income']
    if 'poor_health_pct' in ranked.columns:
        ranked['health_poor_pct'] = ranked['poor_health_pct']
    if 'pharmacies_count' in ranked.columns:
        ranked['n_pharmacies'] = ranked['pharmacies_count']
    
    if 'LON' in ranked.columns and 'LAT' in ranked.columns:
        ranked['lon'] = pd.to_numeric(ranked['LON'], errors='coerce')
        ranked['lat'] = pd.to_numeric(ranked['LAT'], errors='coerce')
    
    ranked['pharm_per_10k'] = (
        ranked['pharmacies_count'] / ranked['population'].clip(lower=1)
    ) * 10000
    
    return ranked


@st.cache_data(show_spinner="Loading location data...")
def load_latlon_lookup():
    """
    Load minimal lat/lon lookup table.
    
    Tries population_data.csv first, then falls back to the unified dataset.
    
    Returns:
        DataFrame with columns: zip, lat, lon (empty if no source found)
    """
    config = get_config()
    empty = pd.DataFrame(columns=["zip", "lat", "lon"])

    # --- Attempt 1: population_data.csv (legacy raw file) ---
    try:
        if _is_s3_environment():
            from data.s3_loaders import parse_s3_path, download_s3_file_to_memory
            import io
            
            s3_path = config.get_population_data_path()
            bucket, key = parse_s3_path(s3_path)
            data = download_s3_file_to_memory(bucket, key)
            df = pd.read_csv(io.BytesIO(data), skiprows=10)
        else:
            population_path = Path('raw_data/population_data.csv')
            if population_path.exists():
                df = pd.read_csv(population_path, skiprows=10)
            else:
                df = None

        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            lower = {c.lower(): c for c in df.columns}
            if all(k in lower for k in ["zip", "lat", "long"]):
                out = pd.DataFrame({
                    "zip": df[lower["zip"]].astype(str).str.extract(r"(\d{5})")[0].str.zfill(5),
                    "lat": pd.to_numeric(df[lower["lat"]], errors="coerce"),
                    "lon": pd.to_numeric(df[lower["long"]], errors="coerce"),
                })
                out = out.dropna(subset=["zip"]).drop_duplicates(subset=["zip"])
                if not out.empty:
                    return out
    except Exception as e:
        logger.warning(f"Population file lat/lon load failed: {e}")

    # --- Attempt 2: unified dataset (has lat/lon from preprocessing) ---
    try:
        from storage.datasets import get_storage
        storage = get_storage()
        version_id = storage.get_latest_version("pharmacy_data")
        if version_id:
            raw = storage.download_unified_dataset("pharmacy_data", version_id)
            udf = pd.read_csv(io.BytesIO(raw), dtype={"zip": str}, usecols=lambda c: c in ("zip", "lat", "lon"))
            if {"zip", "lat", "lon"}.issubset(udf.columns):
                udf["zip"] = udf["zip"].astype(str).str.zfill(5)
                udf["lat"] = pd.to_numeric(udf["lat"], errors="coerce")
                udf["lon"] = pd.to_numeric(udf["lon"], errors="coerce")
                out = udf.dropna(subset=["zip", "lat", "lon"]).drop_duplicates(subset=["zip"])
                if not out.empty:
                    return out
    except Exception as e:
        logger.warning(f"Unified dataset lat/lon load failed: {e}")

    return empty


@st.cache_data(show_spinner="Loading pharmacist data...")
def load_pharmacist_data_only():
    """
    Load only pharmacist data for GLM Only mode.
    Lighter than loading the full math dataset bundle.
    
    Returns:
        DataFrame with pharmacist data (empty if not found)
    """
    config = get_config()
    
    try:
        if _is_s3_environment():
            from data.s3_loaders import parse_s3_path, download_s3_directory
            import tempfile
            
            s3_path = config.get_data_path(config.data_dir)
            bucket, key = parse_s3_path(s3_path)
            temp_dir = tempfile.mkdtemp(prefix='pharmacist_data_')
            local_path = download_s3_directory(bucket, key.rstrip('/') + '/', temp_dir)
            return load_all_pharmacist_data(local_path)
        else:
            raw_data_path = Path('raw_data')
            if not raw_data_path.exists():
                logger.warning(f"Raw data directory not found: {raw_data_path}")
                return pd.DataFrame(columns=['Short_ZIP'])
            return load_all_pharmacist_data('raw_data')
    except Exception as e:
        logger.warning(f"Failed to load pharmacist data: {e}")
        return pd.DataFrame(columns=['Short_ZIP'])


def get_glm_model_info():
    """
    Get info about the GLM model file without loading the full results.
    
    Returns:
        tuple: (exists: bool, last_modified: datetime or None)
    """
    from datetime import datetime
    config = get_config()
    
    if _is_s3_environment():
        # In S3 mode, check if file exists in S3
        try:
            from data.s3_loaders import get_s3_client, parse_s3_path
            s3_path = config.get_glm_results_path()
            bucket, key = parse_s3_path(s3_path)
            s3 = get_s3_client()
            response = s3.head_object(Bucket=bucket, Key=key)
            last_modified = response['LastModified']
            return True, last_modified
        except Exception:
            return False, None
    else:
        ai_file_path = Path("results/national_ifae_rank.csv")
        if ai_file_path.exists():
            last_modified = datetime.fromtimestamp(ai_file_path.stat().st_mtime)
            return True, last_modified
        return False, None


# =============================================================================
# DATASET CONFIG-BASED LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading dataset from configuration...")
def load_dataset_from_config_cached(dataset_id: str, version_id: str = None):
    """
    Load a dataset from its configuration file (cached).
    
    This function:
    1. Reads LATEST.json to get the current version (if version_id not specified)
    2. Reads dataset_config.json for the version
    3. Loads each source file and applies its mapping
    4. Merges all sources on 'zcta5' with appropriate suffixes
    5. Returns the scoring config if available
    
    Args:
        dataset_id: Dataset identifier
        version_id: Specific version to load (uses LATEST if not specified)
    
    Returns:
        Tuple of (DataFrame, ScoringConfig dict or None)
    """
    from ingestion.dataset_loader import load_dataset_from_config as _load_from_config
    from ingestion.dataset_loader import get_scoring_config_from_dataset
    
    logger.info(f"Loading dataset from config: {dataset_id}")
    
    # Load the dataset
    df = _load_from_config(dataset_id, version_id=version_id)
    
    # Standardize column name from zcta5 to zip for compatibility
    if 'zcta5' in df.columns and 'zip' not in df.columns:
        df = df.rename(columns={'zcta5': 'zip'})
    
    # Get scoring config
    scoring_config = get_scoring_config_from_dataset(dataset_id, version_id)
    
    # Convert to dict for caching (ScoringConfig objects can't be cached)
    scoring_config_dict = scoring_config.to_dict() if scoring_config else None
    
    return df, scoring_config_dict


@st.cache_data(show_spinner="Loading dataset...")
def load_smart_dataset_bundle(active_dataset_id: str = "", dataset_version_hint: str = ""):
    """
    Load the active dataset for the app.

    If *active_dataset_id* points to an uploaded (unified) dataset, that dataset
    IS the complete data — no merging with a default bundle is needed.

    Falls back to the built-in default dataset when no uploaded data exists.

    Args:
        active_dataset_id: Dataset ID to load (empty string for default only)
        dataset_version_hint: Optional version ID to include in the cache key

    Returns:
        tuple: (df, pharmacist_data, scoring_config_dict or None)
    """
    from models.schema import get_default_scoring_config

    # ------------------------------------------------------------------
    # If an uploaded dataset exists, load it directly (unified CSV)
    # ------------------------------------------------------------------
    if active_dataset_id:
        logger.info(f"Loading uploaded dataset: {active_dataset_id}")
        try:
            version_id = dataset_version_hint or None
            uploaded_df, scoring_config_dict = load_dataset_from_config_cached(
                active_dataset_id, version_id=version_id
            )
            logger.info(f"Uploaded dataset: {len(uploaded_df)} rows, {len(uploaded_df.columns)} columns")

            pharmacist_data = load_pharmacist_data_only()
            return uploaded_df, pharmacist_data, scoring_config_dict
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Dataset '{active_dataset_id}' not found: {e}. Falling back to default.")

    # ------------------------------------------------------------------
    # Fallback: load built-in default dataset from raw_data/
    # ------------------------------------------------------------------
    logger.info("Loading default dataset from raw_data/")
    default_files_exist = Path('raw_data/financial_data.csv').exists()

    if not default_files_exist:
        logger.warning("No default raw_data files found")
        default_config = get_default_scoring_config()
        return pd.DataFrame(columns=['zip']), pd.DataFrame(columns=['Short_ZIP']), default_config.to_dict()

    default_df, default_pharmacist = load_math_dataset_bundle()
    default_config = get_default_scoring_config()
    pharmacist_data = default_pharmacist if not default_pharmacist.empty else load_pharmacist_data_only()
    return default_df, pharmacist_data, default_config.to_dict()


def get_scoring_config_object(scoring_config_dict):
    """
    Convert a scoring config dict back to a ScoringConfig object.
    
    Args:
        scoring_config_dict: Dictionary from cached load
    
    Returns:
        ScoringConfig object or default config if None
    """
    from models.schema import ScoringConfig, get_default_scoring_config
    
    if scoring_config_dict is None:
        return get_default_scoring_config()
    
    return ScoringConfig.from_dict(scoring_config_dict)


def get_available_datasets():
    """
    Get list of available datasets from storage.
    
    Returns:
        List of dataset info dictionaries
    """
    from ingestion.dataset_loader import list_available_datasets
    
    try:
        return list_available_datasets()
    except Exception as e:
        logger.warning(f"Failed to list datasets: {e}")
        return []


def get_dataset_details(dataset_id: str):
    """
    Get detailed info about a specific dataset.
    
    Args:
        dataset_id: Dataset identifier
    
    Returns:
        Dictionary with dataset details
    """
    from ingestion.dataset_loader import get_dataset_info
    
    try:
        return get_dataset_info(dataset_id)
    except Exception as e:
        logger.warning(f"Failed to get dataset info: {e}")
        return None
