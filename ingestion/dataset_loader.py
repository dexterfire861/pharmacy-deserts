"""
Dataset loader that reads configurations and loads data according to saved mappings.
Provides load_dataset_from_config() as the main entry point.
"""
import io
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging

from .normalize import normalize_zip_column, NORMALIZATION_MODES
from .parsers import parse_file, get_file_type

logger = logging.getLogger(__name__)


def apply_file_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply a file mapping configuration to a DataFrame.
    
    This function:
    1. Normalizes the ZIP column to standard 5-digit ZCTA format
    2. Selects only the specified feature columns
    3. Renames columns according to the mapping
    4. Applies cleaning rules (drop nulls)
    
    Args:
        df: Raw DataFrame loaded from file
        mapping: Mapping configuration dict with keys:
            - zip_column: Column containing ZIP codes
            - normalization_mode: How to normalize ZIPs
            - feature_columns: List of columns to keep
            - column_renames: Dict mapping original names to new names
            - cleaning_rules: Dict with cleaning configuration
    
    Returns:
        Transformed DataFrame with 'zcta5' as the key column
    """
    result = df.copy()
    
    # 1. Normalize ZIP column to zcta5
    zip_column = mapping.get('zip_column')
    norm_mode = mapping.get('normalization_mode', 'already_5_digit')
    
    if zip_column and zip_column in result.columns:
        result['zcta5'] = normalize_zip_column(result, zip_column, norm_mode)
    else:
        raise ValueError(f"ZIP column '{zip_column}' not found in DataFrame. "
                        f"Available columns: {list(result.columns)}")
    
    # 2. Select feature columns (plus zcta5 and geo columns)
    feature_columns = mapping.get('feature_columns', [])
    
    # Always include common geo columns (lat/lon) if they exist, even if not in feature_columns
    # This ensures lat/lon are available for mapping even if user didn't select them as features
    geo_column_names = ['latitude', 'longitude', 'lat', 'lon', 'long', 'lng']
    geo_columns = [col for col in result.columns if col.lower() in geo_column_names]
    
    columns_to_keep = ['zcta5'] + [c for c in feature_columns if c in result.columns] + geo_columns
    
    # Warn about missing columns
    missing = set(feature_columns) - set(result.columns)
    if missing:
        logger.warning(f"Missing columns in data: {missing}")
    
    result = result[columns_to_keep]
    
    # 3. Apply column renames
    column_renames = mapping.get('column_renames', {})
    if column_renames:
        # Only rename columns that exist
        valid_renames = {k: v for k, v in column_renames.items() if k in result.columns}
        result = result.rename(columns=valid_renames)
    
    # 4. Apply cleaning rules
    cleaning_rules = mapping.get('cleaning_rules', {})
    drop_null_columns = cleaning_rules.get('drop_null_columns', [])
    
    if drop_null_columns:
        # Map original column names to potentially renamed ones
        mapped_drop_cols = []
        for col in drop_null_columns:
            if col == zip_column:
                mapped_drop_cols.append('zcta5')
            elif col in column_renames:
                mapped_drop_cols.append(column_renames[col])
            elif col in result.columns:
                mapped_drop_cols.append(col)
        
        if mapped_drop_cols:
            before_count = len(result)
            result = result.dropna(subset=mapped_drop_cols)
            dropped = before_count - len(result)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NULL values")
    
    # 5. Drop rows where zcta5 is None (failed normalization)
    result = result.dropna(subset=['zcta5'])
    
    # 6. Remove duplicates by zcta5 (keep first occurrence)
    result = result.drop_duplicates(subset=['zcta5'], keep='first')
    
    return result


def load_single_source(
    storage,
    dataset_id: str,
    version_id: str,
    source_mapping: Dict[str, Any]
) -> pd.DataFrame:
    """
    Load a single source file and apply its mapping.
    
    Args:
        storage: Storage backend (Local or S3)
        dataset_id: Dataset identifier
        version_id: Version identifier
        source_mapping: Source configuration from dataset_config.json
    
    Returns:
        Transformed DataFrame
    """
    filename = source_mapping.get('filename')
    file_type = source_mapping.get('file_type')
    sheet_name = source_mapping.get('sheet_name')
    skip_rows = source_mapping.get('skip_rows', 0)
    
    logger.info(f"Loading source: {filename}")
    
    # Download file content
    file_content = storage.download_file(dataset_id, version_id, filename)
    
    # Parse file
    df = parse_file(
        filename=filename,
        file_content=file_content,
        file_type=file_type,
        sheet_name=sheet_name,
        skip_rows=skip_rows
    )
    
    logger.info(f"Parsed {len(df)} rows from {filename}")
    
    # Apply mapping
    transformed = apply_file_mapping(df, source_mapping)
    
    logger.info(f"After transformation: {len(transformed)} rows, columns: {list(transformed.columns)}")
    
    return transformed


def merge_dataframes_on_zcta5(
    dataframes: List[Tuple[str, pd.DataFrame]],
    how: str = 'outer'
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on zcta5 column.
    
    Args:
        dataframes: List of (source_name, DataFrame) tuples
        how: Merge method ('outer', 'inner', 'left', 'right')
    
    Returns:
        Merged DataFrame with suffixes for conflicting column names
    """
    if not dataframes:
        return pd.DataFrame(columns=['zcta5'])
    
    if len(dataframes) == 1:
        return dataframes[0][1]
    
    # Start with the first dataframe
    result = dataframes[0][1].copy()
    first_name = dataframes[0][0]
    
    # Rename columns to include source prefix (except zcta5)
    if len(dataframes) > 1:
        new_cols = {}
        for col in result.columns:
            if col != 'zcta5':
                new_cols[col] = f"{col}"  # Keep original name for first source
        if new_cols:
            result = result.rename(columns=new_cols)
    
    # Merge remaining dataframes
    for source_name, df in dataframes[1:]:
        # Add suffix based on source name for conflicts
        suffix = f"_{Path(source_name).stem}"
        
        # Check for column conflicts (excluding zcta5)
        existing_cols = set(result.columns) - {'zcta5'}
        new_cols = set(df.columns) - {'zcta5'}
        conflicts = existing_cols & new_cols
        
        if conflicts:
            logger.info(f"Column conflicts with {source_name}: {conflicts}")
        
        result = result.merge(
            df,
            on='zcta5',
            how=how,
            suffixes=('', suffix)
        )
    
    return result


def load_dataset_from_config(
    dataset_id: str,
    version_id: Optional[str] = None,
    storage=None,
    include_scoring_config: bool = False
) -> pd.DataFrame:
    """
    Load a dataset from its stored configuration.

    Prefers the unified CSV (single pre-merged file created at upload time).
    Falls back to per-source loading for legacy dataset versions.

    Args:
        dataset_id: Unique identifier for the dataset
        version_id: Specific version to load (uses LATEST if not provided)
        storage: Storage backend (auto-detected if not provided)
        include_scoring_config: If True, returns (df, scoring_config) tuple

    Returns:
        DataFrame with 'zcta5' as the key column.
        If include_scoring_config=True, returns (DataFrame, ScoringConfig or None)
    """
    if storage is None:
        from storage.datasets import get_storage
        storage = get_storage()

    if version_id is None:
        version_id = storage.get_latest_version(dataset_id)
        if version_id is None:
            raise ValueError(f"No versions found for dataset '{dataset_id}'")
        logger.info(f"Using latest version: {version_id}")

    config = storage.get_config(dataset_id, version_id)
    if config is None:
        raise ValueError(f"Configuration not found for {dataset_id}/{version_id}")

    logger.info(f"Loading dataset '{dataset_id}' version '{version_id}'")

    # ------------------------------------------------------------------
    # Try unified CSV first (new format)
    # ------------------------------------------------------------------
    if config.get('unified', False):
        try:
            csv_bytes = storage.download_unified_dataset(dataset_id, version_id)
            merged = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
            for col in ('zcta5', 'zip'):
                if col in merged.columns:
                    merged[col] = merged[col].astype(str).str.zfill(5)
            logger.info(f"Loaded unified dataset: {len(merged)} rows, {len(merged.columns)} columns")

            if include_scoring_config:
                scoring_config = get_scoring_config_from_dataset(dataset_id, version_id, storage)
                return merged, scoring_config
            return merged
        except FileNotFoundError:
            logger.warning("Unified CSV flag set but file not found — falling back to per-source loading")

    # ------------------------------------------------------------------
    # Legacy fallback: load individual sources and merge
    # ------------------------------------------------------------------
    logger.info(f"Loading {len(config.get('sources', []))} source(s) (legacy mode)")

    dataframes = []
    for source in config.get('sources', []):
        filename = source.get('filename', 'unknown')
        try:
            df = load_single_source(storage, dataset_id, version_id, source)
            dataframes.append((filename, df))
        except Exception as e:
            logger.error(f"Failed to load source '{filename}': {e}")
            raise

    if not dataframes:
        raise ValueError("No sources found in dataset configuration")

    merged = merge_dataframes_on_zcta5(dataframes)
    logger.info(f"Final merged dataset: {len(merged)} rows, {len(merged.columns)} columns")

    if include_scoring_config:
        scoring_config = get_scoring_config_from_dataset(dataset_id, version_id, storage)
        return merged, scoring_config
    return merged


def get_scoring_config_from_dataset(
    dataset_id: str,
    version_id: Optional[str] = None,
    storage=None
):
    """
    Get the ScoringConfig from a dataset's configuration.
    
    Args:
        dataset_id: Dataset identifier
        version_id: Version identifier (uses LATEST if not provided)
        storage: Storage backend
    
    Returns:
        ScoringConfig object or None if not configured
    """
    if storage is None:
        from storage.datasets import get_storage
        storage = get_storage()
    
    if version_id is None:
        version_id = storage.get_latest_version(dataset_id)
    
    if version_id is None:
        return None
    
    config = storage.get_config(dataset_id, version_id)
    if config is None:
        return None
    
    # Check for scoring_config in the dataset config
    scoring_config_dict = config.get('scoring_config')
    if scoring_config_dict is None:
        return None
    
    # Import here to avoid circular imports
    from models.schema import ScoringConfig
    
    try:
        return ScoringConfig.from_dict(scoring_config_dict)
    except Exception as e:
        logger.warning(f"Failed to parse scoring config: {e}")
        return None


def get_dataset_info(dataset_id: str, version_id: Optional[str] = None, storage=None) -> Dict[str, Any]:
    """
    Get information about a dataset without loading the full data.
    
    Args:
        dataset_id: Dataset identifier
        version_id: Version identifier (uses LATEST if not provided)
        storage: Storage backend
    
    Returns:
        Dictionary with dataset metadata
    """
    if storage is None:
        from storage.datasets import get_storage
        storage = get_storage()
    
    if version_id is None:
        version_id = storage.get_latest_version(dataset_id)
    
    if version_id is None:
        return {"error": f"Dataset '{dataset_id}' not found"}
    
    config = storage.get_config(dataset_id, version_id)
    if config is None:
        return {"error": f"Configuration not found for {dataset_id}/{version_id}"}
    
    # Summarize sources
    sources_info = []
    for source in config.get('sources', []):
        sources_info.append({
            'filename': source.get('filename'),
            'file_type': source.get('file_type'),
            'features': source.get('feature_columns', []),
            'zip_column': source.get('zip_column'),
        })
    
    # Include scoring config info
    scoring_config = config.get('scoring_config')
    scoring_info = None
    if scoring_config:
        scoring_info = {
            'mapped_components': len(scoring_config.get('column_mappings', [])),
            'has_weight_overrides': bool(scoring_config.get('weight_overrides')),
        }
    
    return {
        'dataset_id': dataset_id,
        'version_id': version_id,
        'created_at': config.get('created_at'),
        'status': config.get('status'),
        'source_count': len(sources_info),
        'sources': sources_info,
        'total_features': sum(len(s['features']) for s in sources_info),
        'scoring_config': scoring_info,
    }


def list_available_datasets(storage=None) -> List[Dict[str, Any]]:
    """
    List all available datasets with basic info.
    
    Args:
        storage: Storage backend
    
    Returns:
        List of dataset info dictionaries
    """
    if storage is None:
        from storage.datasets import get_storage
        storage = get_storage()
    
    datasets = []
    for dataset_id in storage.list_datasets():
        latest = storage.get_latest_version(dataset_id)
        if latest:
            config = storage.get_config(dataset_id, latest)
            has_scoring = bool(config.get('scoring_config')) if config else False
            source_count = 0
            if config:
                source_count = len(config.get('uploaded_files', {})) or len(config.get('sources', []))
            datasets.append({
                'dataset_id': dataset_id,
                'latest_version': latest,
                'version': latest,
                'created_at': config.get('created_at') if config else None,
                'source_count': source_count,
                'has_scoring_config': has_scoring,
                'description': config.get('description', '') if config else '',
            })
    
    return datasets


# Alias for backwards compatibility
apply_mapping_to_dataframe = apply_file_mapping


def get_active_dataset_id() -> Optional[str]:
    """
    Get the active dataset ID from environment configuration.
    
    Returns:
        Dataset ID if configured via ACTIVE_DATASET_ID env var, None otherwise
    """
    from app.config import get_config
    config = get_config()
    return config.active_dataset_id


def load_dataset_with_scoring(
    dataset_id: str,
    version_id: Optional[str] = None,
    storage=None
) -> Tuple[pd.DataFrame, Optional[Any]]:
    """
    Load a dataset along with its scoring configuration.
    
    This is the recommended function for loading datasets that need to be scored.
    
    Args:
        dataset_id: Dataset identifier
        version_id: Version identifier (uses LATEST if not provided)
        storage: Storage backend
    
    Returns:
        Tuple of (DataFrame, ScoringConfig or None)
    """
    return load_dataset_from_config(
        dataset_id, 
        version_id, 
        storage, 
        include_scoring_config=True
    )


def load_dataset_or_default(
    dataset_id: Optional[str] = None,
    fallback_loader=None,
    include_scoring_config: bool = False
) -> Tuple[pd.DataFrame, str, Optional[Any]]:
    """
    Load dataset from config if available, otherwise use fallback.
    
    This is the main entry point for the app data loading. It checks:
    1. If dataset_id is provided, load that dataset
    2. Else if ACTIVE_DATASET_ID env var is set, load that dataset
    3. Else use the fallback_loader function
    
    Args:
        dataset_id: Explicit dataset ID to load (optional)
        fallback_loader: Callable that returns a DataFrame when no dataset config is found
        include_scoring_config: If True, also returns the ScoringConfig
    
    Returns:
        Tuple of (DataFrame, source_description, ScoringConfig or None)
            - DataFrame: The loaded data
            - source_description: String describing where data came from
            - ScoringConfig: The scoring configuration (None if using fallback)
    """
    # Determine which dataset to load
    active_id = dataset_id or get_active_dataset_id()
    
    if active_id:
        try:
            logger.info(f"Loading dataset from config: {active_id}")
            
            if include_scoring_config:
                df, scoring_config = load_dataset_from_config(
                    active_id, include_scoring_config=True
                )
            else:
                df = load_dataset_from_config(active_id)
                scoring_config = None
            
            # Rename zcta5 to zip for compatibility with existing app
            if 'zcta5' in df.columns and 'zip' not in df.columns:
                df = df.rename(columns={'zcta5': 'zip'})
            
            return df, f"dataset:{active_id}", scoring_config
        except Exception as e:
            logger.warning(f"Failed to load dataset '{active_id}': {e}")
            if fallback_loader is None:
                raise
            logger.info("Falling back to default loader")
    
    if fallback_loader is not None:
        df = fallback_loader()
        # Fallback uses the default scoring config
        from models.schema import get_default_scoring_config
        default_config = get_default_scoring_config() if include_scoring_config else None
        return df, "default:local_files", default_config
    
    raise ValueError("No dataset_id provided and no fallback_loader specified")
