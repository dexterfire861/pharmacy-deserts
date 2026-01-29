# pharmacy_deserts/app/__init__.py
"""
Streamlit application package for Pharmacy Desert Explorer.
"""
from .config import get_config, load_config, Config
from .auth import (
    login_form,
    logout,
    logout_button,
    is_authenticated,
    check_password,
    require_auth,
)
from .state import (
    load_math_dataset_bundle,
    load_glm_results,
    load_latlon_lookup,
    load_pharmacist_data_only,
    get_glm_model_info,
    # Dataset config loading
    get_active_dataset_id,
    set_active_dataset_id,
    is_using_dataset_config,
    load_dataset_from_config_cached,
    load_smart_dataset_bundle,
    get_scoring_config_object,
    get_available_datasets,
    get_dataset_details,
)

__all__ = [
    # Config
    'get_config',
    'load_config',
    'Config',
    # Auth
    'login_form',
    'logout',
    'logout_button',
    'is_authenticated',
    'check_password',
    'require_auth',
    # State - Default loading
    'load_math_dataset_bundle',
    'load_glm_results',
    'load_latlon_lookup',
    'load_pharmacist_data_only',
    'get_glm_model_info',
    # State - Dataset config loading
    'get_active_dataset_id',
    'set_active_dataset_id',
    'is_using_dataset_config',
    'load_dataset_from_config_cached',
    'load_smart_dataset_bundle',
    'get_scoring_config_object',
    'get_available_datasets',
    'get_dataset_details',
]
