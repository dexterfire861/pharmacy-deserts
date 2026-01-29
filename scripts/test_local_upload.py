#!/usr/bin/env python3
"""
Test script for local dataset upload and loading workflow.

This script demonstrates how the dataset upload system works locally:
1. Creates a simple test dataset
2. Uploads it with mappings
3. Loads it back using the config

Run from project root:
    python scripts/test_local_upload.py

NOTE: This script forces local storage regardless of .env settings.
"""
import sys
import os
from pathlib import Path

# Force local development mode BEFORE importing anything else
os.environ['ENVIRONMENT'] = 'development'
os.environ.pop('AWS_S3_BUCKET', None)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from storage.datasets import DatasetStorageLocal, build_dataset_config, build_file_mapping, generate_version_id
from ingestion.dataset_loader import load_dataset_from_config, get_dataset_info, list_available_datasets


def create_test_csv():
    """Create a simple test CSV file."""
    data = {
        'ZIP': ['01234', '02345', '03456', '04567', '05678'],
        'population': [10000, 25000, 15000, 30000, 8000],
        'income': [45000, 65000, 55000, 75000, 40000],
        'pharmacy_count': [2, 5, 3, 8, 1],
        'health_index': [0.6, 0.4, 0.5, 0.3, 0.7],
    }
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("LOCAL DATASET UPLOAD TEST")
    print("=" * 60)
    
    # Initialize local storage
    storage = DatasetStorageLocal()
    print(f"\n📁 Storage path: {storage.base_path.absolute()}")
    
    # Create test dataset
    dataset_id = "test_dataset"
    version_id = generate_version_id()
    filename = "test_data.csv"
    
    print(f"\n📦 Creating test dataset: {dataset_id}")
    print(f"   Version: {version_id}")
    
    # Create test data
    test_df = create_test_csv()
    print(f"\n📊 Test data preview:")
    print(test_df.to_string(index=False))
    
    # Convert to CSV bytes
    csv_content = test_df.to_csv(index=False).encode('utf-8')
    
    # Upload file
    print(f"\n⬆️  Uploading file: {filename}")
    file_path = storage.upload_file(dataset_id, version_id, filename, csv_content)
    print(f"   Saved to: {file_path}")
    
    # Create and upload mapping
    mapping = build_file_mapping(
        filename=filename,
        file_type='csv',
        zip_column='ZIP',
        normalization_mode='already_5_digit',
        feature_columns=['population', 'income', 'pharmacy_count', 'health_index'],
        column_renames={
            'pharmacy_count': 'n_pharmacies',
            'health_index': 'health_poor_pct'
        },
        cleaning_rules={'drop_null_columns': ['ZIP']},
        skip_rows=0
    )
    
    print(f"\n📋 Creating mapping config:")
    print(json.dumps(mapping, indent=2))
    
    mapping_path = storage.upload_mapping(dataset_id, version_id, filename, mapping)
    print(f"   Saved to: {mapping_path}")
    
    # Create and upload dataset config
    config = build_dataset_config(
        dataset_id=dataset_id,
        version_id=version_id,
        sources=[mapping]
    )
    
    config_path = storage.upload_config(dataset_id, version_id, config)
    print(f"\n⚙️  Dataset config saved to: {config_path}")
    
    # Update LATEST pointer
    latest_path = storage.update_latest(dataset_id, version_id)
    print(f"📌 LATEST.json updated: {latest_path}")
    
    # Show directory structure
    print("\n" + "=" * 60)
    print("DIRECTORY STRUCTURE")
    print("=" * 60)
    
    base = storage.base_path / dataset_id
    for path in sorted(base.rglob('*')):
        rel = path.relative_to(storage.base_path)
        indent = "  " * (len(rel.parts) - 1)
        if path.is_file():
            size = path.stat().st_size
            print(f"{indent}📄 {path.name} ({size} bytes)")
        else:
            print(f"{indent}📁 {path.name}/")
    
    # Test loading the dataset back
    print("\n" + "=" * 60)
    print("LOADING DATASET BACK")
    print("=" * 60)
    
    # Use the same local storage instance for consistency
    # List available datasets
    datasets = list_available_datasets(storage=storage)
    print(f"\n📦 Available datasets: {len(datasets)}")
    for ds in datasets:
        print(f"   - {ds['dataset_id']} (version: {ds['latest_version']})")
    
    # Get dataset info
    info = get_dataset_info(dataset_id, storage=storage)
    print(f"\n📊 Dataset info:")
    print(json.dumps(info, indent=2, default=str))
    
    # Load dataset
    print(f"\n⬇️  Loading dataset: {dataset_id}")
    loaded_df = load_dataset_from_config(dataset_id, storage=storage)
    
    print(f"\n✅ Loaded DataFrame:")
    print(f"   Shape: {loaded_df.shape}")
    print(f"   Columns: {list(loaded_df.columns)}")
    print(f"\n{loaded_df.to_string(index=False)}")
    
    # Note the transformations
    print("\n" + "=" * 60)
    print("TRANSFORMATIONS APPLIED")
    print("=" * 60)
    print("""
    ✓ ZIP column → 'zcta5' (normalized to 5-digit)
    ✓ Selected features kept: population, income, pharmacy_count, health_index
    ✓ Renamed: pharmacy_count → n_pharmacies
    ✓ Renamed: health_index → health_poor_pct
    ✓ Rows with NULL ZIP dropped
    """)
    
    print("\n" + "=" * 60)
    print("HOW TO USE IN THE APP")
    print("=" * 60)
    print(f"""
    Option 1: Environment variable
        export ACTIVE_DATASET_ID={dataset_id}
        streamlit run app/app.py
    
    Option 2: Add to .env file
        ACTIVE_DATASET_ID={dataset_id}
    
    Option 3: Select in app sidebar
        The Dataset Selector dropdown will show '{dataset_id}'
    """)
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()

