# pharmacy_deserts/storage/__init__.py
"""
Storage utilities for dataset and model management with local and S3 backends.
"""
from .datasets import (
    generate_version_id,
    get_file_hash,
    get_s3_dataset_prefix,
    get_s3_version_prefix,
    build_dataset_config,
    build_file_mapping,
    DatasetStorageLocal,
    DatasetStorageS3,
    get_storage,
)
from .models import (
    ModelInfo,
    build_model_config,
    get_s3_model_prefix,
    get_s3_model_version_prefix,
    ModelRegistryLocal,
    ModelRegistryS3,
    get_model_registry,
)

__all__ = [
    # Dataset storage
    'generate_version_id',
    'get_file_hash',
    'get_s3_dataset_prefix',
    'get_s3_version_prefix',
    'build_dataset_config',
    'build_file_mapping',
    'DatasetStorageLocal',
    'DatasetStorageS3',
    'get_storage',
    # Model registry
    'ModelInfo',
    'build_model_config',
    'get_s3_model_prefix',
    'get_s3_model_version_prefix',
    'ModelRegistryLocal',
    'ModelRegistryS3',
    'get_model_registry',
]

