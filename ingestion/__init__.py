# pharmacy_deserts/ingestion/__init__.py
"""
Data ingestion utilities for parsing, normalizing, loading datasets, and API connectors.
"""
from .parsers import (
    get_file_type,
    get_excel_sheet_names,
    parse_file,
    parse_csv,
    parse_excel,
    parse_json,
    parse_parquet,
    extract_zip_contents,
    get_column_types,
    get_column_stats,
)
from .normalize import (
    NORMALIZATION_MODES,
    normalize_zip_column,
    normalize_zip_already_5_digit,
    normalize_zip_extract_regex,
    normalize_zip_plus_4,
    detect_zip_column,
    suggest_normalization_mode,
)
from .dataset_loader import (
    load_dataset_from_config,
    load_dataset_or_default,
    load_single_source,
    apply_mapping_to_dataframe,
    merge_dataframes_on_zcta5,
    get_active_dataset_id,
    get_dataset_info,
    list_available_datasets,
)
from .api_connectors import (
    APISourceConfig,
    BaseAPIConnector,
    CensusACSConnector,
    HUDConnector,
    CustomAPIConnector,
    CENSUS_PRESETS,
    US_STATES,
    CONNECTORS,
    get_connector,
    list_connectors,
    list_census_presets,
)

__all__ = [
    # Parsers
    'get_file_type',
    'get_excel_sheet_names',
    'parse_file',
    'parse_csv',
    'parse_excel',
    'parse_json',
    'parse_parquet',
    'extract_zip_contents',
    'get_column_types',
    'get_column_stats',
    # Normalize
    'NORMALIZATION_MODES',
    'normalize_zip_column',
    'normalize_zip_already_5_digit',
    'normalize_zip_extract_regex',
    'normalize_zip_plus_4',
    'detect_zip_column',
    'suggest_normalization_mode',
    # Dataset Loader
    'load_dataset_from_config',
    'load_dataset_or_default',
    'load_single_source',
    'apply_mapping_to_dataframe',
    'merge_dataframes_on_zcta5',
    'get_active_dataset_id',
    'get_dataset_info',
    'list_available_datasets',
    # API Connectors
    'APISourceConfig',
    'BaseAPIConnector',
    'CensusACSConnector',
    'HUDConnector',
    'CustomAPIConnector',
    'CENSUS_PRESETS',
    'US_STATES',
    'CONNECTORS',
    'get_connector',
    'list_connectors',
    'list_census_presets',
]
