#!/usr/bin/env python3
"""
Reset the pharmacy_data dataset to a clean state.

This script:
1. Backs up existing versions to a backup directory
2. Removes LATEST.json to start fresh
3. Optionally removes all versions (if --clean flag)

Usage:
    python scripts/reset_dataset.py              # Backup and reset
    python scripts/reset_dataset.py --clean      # Backup and remove all versions
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reset pharmacy_data dataset")
    parser.add_argument('--clean', action='store_true', help='Remove all versions (not just backup)')
    args = parser.parse_args()
    
    dataset_path = project_root / "raw_data" / "datasets" / "pharmacy_data"
    versions_path = dataset_path / "versions"
    latest_path = dataset_path / "LATEST.json"
    
    # Create backup directory
    backup_dir = project_root / "raw_data" / "datasets" / "pharmacy_data_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_versions = backup_dir / f"versions_backup_{timestamp}"
    
    print("=" * 60)
    print("Dataset Reset Script")
    print("=" * 60)
    
    # Backup existing versions
    if versions_path.exists() and any(versions_path.iterdir()):
        print(f"\n📦 Backing up existing versions to: {backup_versions}")
        shutil.copytree(versions_path, backup_versions, dirs_exist_ok=True)
        print(f"   ✓ Backed up {len(list(versions_path.iterdir()))} version(s)")
    else:
        print("\n📦 No existing versions to backup")
    
    # Backup LATEST.json
    if latest_path.exists():
        backup_latest = backup_dir / f"LATEST_backup_{timestamp}.json"
        shutil.copy2(latest_path, backup_latest)
        print(f"   ✓ Backed up LATEST.json")
    
    # Remove LATEST.json
    if latest_path.exists():
        latest_path.unlink()
        print(f"\n🗑️  Removed LATEST.json")
    
    # Optionally remove all versions
    if args.clean:
        if versions_path.exists():
            shutil.rmtree(versions_path)
            versions_path.mkdir(parents=True, exist_ok=True)
            print(f"🗑️  Removed all versions")
    
    print("\n" + "=" * 60)
    print("✅ Reset Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Go to Upload Data page in the Streamlit app")
    print("2. Upload your first data file(s)")
    print("3. Each subsequent upload will add to the accumulated dataset")
    print("4. The platform will automatically use the latest version")
    print(f"\nBackup location: {backup_versions}")

if __name__ == "__main__":
    main()
