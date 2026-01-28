# pharmacy_deserts/data/__init__.py
"""
Data loading and feature engineering modules for pharmacy desert analysis.
"""
from .loaders import (
    read_financial_data,
    read_health_data,
    read_pharmacy_data,
    read_population_data,
    read_population_labels,
    read_hhi_excel,
    read_education_data_acs,
    read_hud_zip_county_crosswalk,
    read_county_desert_csv,
    downscale_county_to_zip,
    load_all_pharmacist_data,
    get_pharmacists_for_zip,
)
from .features import preprocess, norm01
from .s3_loaders import (
    is_s3_path,
    parse_s3_path,
    smart_read_csv,
    smart_read_excel,
    smart_load_directory,
    get_data_path,
    clear_s3_cache,
)

__all__ = [
    # Loaders
    'read_financial_data',
    'read_health_data',
    'read_pharmacy_data',
    'read_population_data',
    'read_population_labels',
    'read_hhi_excel',
    'read_education_data_acs',
    'read_hud_zip_county_crosswalk',
    'read_county_desert_csv',
    'downscale_county_to_zip',
    'load_all_pharmacist_data',
    'get_pharmacists_for_zip',
    # Features
    'preprocess',
    'norm01',
    # S3 Loaders
    'is_s3_path',
    'parse_s3_path',
    'smart_read_csv',
    'smart_read_excel',
    'smart_load_directory',
    'get_data_path',
    'clear_s3_cache',
]

