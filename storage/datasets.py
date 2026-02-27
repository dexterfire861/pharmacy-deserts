"""
Dataset storage and versioning utilities for S3-backed data management.
Handles uploading, versioning, and configuration management for datasets.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import hashlib


def generate_version_id() -> str:
    """Generate a unique version ID with sub-second precision."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def get_file_hash(file_bytes: bytes) -> str:
    """Generate MD5 hash of file contents for integrity checking."""
    return hashlib.md5(file_bytes).hexdigest()


def generate_source_id(filename: str, sheet_name: Optional[str] = None) -> str:
    """
    Generate a stable source ID for a file.
    
    Source IDs are used to identify sources across versions for add/replace logic.
    Format: {filename}_{sheet_name} if sheet_name, else {filename}
    
    Args:
        filename: Original filename
        sheet_name: Excel sheet name (if applicable)
    
    Returns:
        Stable source ID string
    """
    # Normalize filename (lowercase, but keep extension)
    source_id = filename.lower()
    
    # Add sheet name if present (for Excel files)
    if sheet_name:
        # Normalize sheet name: lowercase, replace spaces with underscores
        normalized_sheet = sheet_name.lower().replace(' ', '_')
        source_id = f"{source_id}_{normalized_sheet}"
    
    return source_id


def get_s3_dataset_prefix(dataset_id: str) -> str:
    """Get the S3 prefix for a dataset."""
    return f"raw_data/datasets/{dataset_id}"


def get_s3_version_prefix(dataset_id: str, version_id: str) -> str:
    """Get the S3 prefix for a specific dataset version."""
    return f"raw_data/datasets/{dataset_id}/versions/{version_id}"


def build_dataset_config(
    dataset_id: str,
    version_id: str,
    sources: List[Dict[str, Any]],
    created_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build the dataset configuration JSON structure.
    
    Args:
        dataset_id: Unique identifier for the dataset
        version_id: Version identifier
        sources: List of source file configurations
        created_at: Timestamp string (auto-generated if not provided)
    
    Returns:
        Dictionary representing the dataset configuration
    """
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "created_at": created_at or datetime.utcnow().isoformat(),
        "sources": sources,
        "status": "active"
    }


def build_file_mapping(
    filename: str,
    file_type: str,
    zip_column: str,
    normalization_mode: str,
    feature_columns: List[str],
    column_renames: Dict[str, str],
    cleaning_rules: Dict[str, Any],
    sheet_name: Optional[str] = None,
    skip_rows: int = 0,
    source_id: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build the mapping configuration for a single file.
    
    Args:
        filename: Original filename
        file_type: File type (csv, xlsx, etc.)
        zip_column: Column containing ZIP/ZCTA codes
        normalization_mode: ZIP normalization mode
        feature_columns: List of selected feature columns
        column_renames: Mapping of original to new column names
        cleaning_rules: Cleaning configuration
        sheet_name: Excel sheet name (if applicable)
        skip_rows: Number of rows to skip
        source_id: Optional source ID (auto-generated if not provided)
        metadata_columns: Optional list of columns to exclude from training (auto-detected if None)
    
    Returns:
        Dictionary representing the file mapping
    """
    # Generate source_id if not provided
    if source_id is None:
        source_id = generate_source_id(filename, sheet_name)
    
    # Default to empty list if metadata_columns not provided
    if metadata_columns is None:
        metadata_columns = []
    
    return {
        "source_id": source_id,
        "filename": filename,
        "file_type": file_type,
        "sheet_name": sheet_name,
        "skip_rows": skip_rows,
        "zip_column": zip_column,
        "normalization_mode": normalization_mode,
        "feature_columns": feature_columns,
        "metadata_columns": metadata_columns,
        "column_renames": column_renames,
        "cleaning_rules": cleaning_rules
    }


class DatasetStorageLocal:
    """Local filesystem storage for development."""
    
    def __init__(self, base_path: str = "raw_data/datasets"):
        self.base_path = Path(base_path)
    
    def upload_file(self, dataset_id: str, version_id: str, filename: str, content: bytes) -> str:
        """Upload a file to local storage."""
        path = self.base_path / dataset_id / "versions" / version_id / "files" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)
    
    def upload_unified_dataset(self, dataset_id: str, version_id: str, csv_bytes: bytes) -> str:
        """Upload the unified merged dataset CSV."""
        path = self.base_path / dataset_id / "versions" / version_id / "unified_dataset.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes)
        return str(path)
    
    def download_unified_dataset(self, dataset_id: str, version_id: str) -> bytes:
        """Download the unified merged dataset CSV."""
        path = self.base_path / dataset_id / "versions" / version_id / "unified_dataset.csv"
        if not path.exists():
            raise FileNotFoundError(f"Unified dataset not found: {path}")
        return path.read_bytes()
    
    def download_file(self, dataset_id: str, version_id: str, filename: str) -> bytes:
        """Download a file from local storage."""
        path = self.base_path / dataset_id / "versions" / version_id / "files" / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()
    
    def copy_file_from_version(
        self, 
        dataset_id: str, 
        source_version_id: str, 
        target_version_id: str, 
        filename: str
    ) -> str:
        """Copy a file from one version to another."""
        source_path = self.base_path / dataset_id / "versions" / source_version_id / "files" / filename
        target_path = self.base_path / dataset_id / "versions" / target_version_id / "files" / filename
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file content
        target_path.write_bytes(source_path.read_bytes())
        
        return str(target_path)
    
    def upload_mapping(self, dataset_id: str, version_id: str, filename: str, mapping: Dict) -> str:
        """Upload a mapping JSON file."""
        mapping_filename = f"{Path(filename).stem}_mapping.json"
        path = self.base_path / dataset_id / "versions" / version_id / "mappings" / mapping_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2))
        return str(path)
    
    def upload_config(self, dataset_id: str, version_id: str, config: Dict) -> str:
        """Upload the dataset configuration."""
        path = self.base_path / dataset_id / "versions" / version_id / "dataset_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))
        return str(path)
    
    def update_latest(self, dataset_id: str, version_id: str) -> str:
        """Update LATEST.json to point to the new version."""
        latest = {
            "latest_version": version_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        path = self.base_path / dataset_id / "LATEST.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(latest, indent=2))
        return str(path)
    
    def get_latest_version(self, dataset_id: str) -> Optional[str]:
        """Get the latest version ID for a dataset."""
        path = self.base_path / dataset_id / "LATEST.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("latest_version")
        return None
    
    def get_config(self, dataset_id: str, version_id: str) -> Optional[Dict]:
        """Get the configuration for a specific version."""
        path = self.base_path / dataset_id / "versions" / version_id / "dataset_config.json"
        if path.exists():
            return json.loads(path.read_text())
        return None
    
    def list_datasets(self) -> List[str]:
        """List all dataset IDs."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]
    
    def list_versions(self, dataset_id: str) -> List[str]:
        """List all versions for a dataset."""
        versions_path = self.base_path / dataset_id / "versions"
        if not versions_path.exists():
            return []
        return sorted([v.name for v in versions_path.iterdir() if v.is_dir()], reverse=True)


class DatasetStorageS3:
    """S3 storage for production."""
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        if not bucket:
            raise ValueError("S3 bucket name is required for production storage")
        self.bucket = bucket
        self.region = region
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        self.client = boto3.client('s3', region_name=region)
    
    def upload_file(self, dataset_id: str, version_id: str, filename: str, content: bytes) -> str:
        """Upload a file to S3."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/files/{filename}"
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise IOError(f"Failed to upload file to S3: {error_code} - {e}") from e
        return f"s3://{self.bucket}/{key}"
    
    def upload_unified_dataset(self, dataset_id: str, version_id: str, csv_bytes: bytes) -> str:
        """Upload the unified merged dataset CSV to S3."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/unified_dataset.csv"
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=csv_bytes,
                ContentType='text/csv'
            )
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise IOError(f"Failed to upload unified dataset to S3: {error_code} - {e}") from e
        return f"s3://{self.bucket}/{key}"
    
    def download_unified_dataset(self, dataset_id: str, version_id: str) -> bytes:
        """Download the unified merged dataset CSV from S3."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/unified_dataset.csv"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Unified dataset not found: s3://{self.bucket}/{key}")
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Unified dataset not found: s3://{self.bucket}/{key}")
            raise IOError(f"Failed to download unified dataset: {error_code} - {e}") from e
    
    def download_file(self, dataset_id: str, version_id: str, filename: str) -> bytes:
        """Download a file from S3."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/files/{filename}"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"File not found in S3: s3://{self.bucket}/{key}")
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: s3://{self.bucket}/{key}")
            elif error_code == '403':
                raise PermissionError(f"Access denied to S3 object: s3://{self.bucket}/{key}")
            else:
                raise IOError(f"Failed to download from S3: {error_code} - {e}") from e
        except Exception as e:
            raise IOError(f"Unexpected error downloading from S3: {e}") from e
    
    def copy_file_from_version(
        self, 
        dataset_id: str, 
        source_version_id: str, 
        target_version_id: str, 
        filename: str
    ) -> str:
        """Copy a file from one version to another in S3."""
        source_key = f"{get_s3_version_prefix(dataset_id, source_version_id)}/files/{filename}"
        target_key = f"{get_s3_version_prefix(dataset_id, target_version_id)}/files/{filename}"
        
        copy_source = {'Bucket': self.bucket, 'Key': source_key}
        
        try:
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket,
                Key=target_key
            )
            return f"s3://{self.bucket}/{target_key}"
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Source file not found in S3: s3://{self.bucket}/{source_key}")
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Source file not found in S3: s3://{self.bucket}/{source_key}")
            elif error_code == '403':
                raise PermissionError(f"Access denied copying S3 object: s3://{self.bucket}/{source_key}")
            else:
                raise IOError(f"Failed to copy file in S3: {error_code} - {e}") from e
        except Exception as e:
            raise IOError(f"Unexpected error copying file in S3: {e}") from e
    
    def upload_mapping(self, dataset_id: str, version_id: str, filename: str, mapping: Dict) -> str:
        """Upload a mapping JSON file to S3."""
        mapping_filename = f"{Path(filename).stem}_mapping.json"
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/mappings/{mapping_filename}"
        self.client.put_object(
            Bucket=self.bucket, 
            Key=key, 
            Body=json.dumps(mapping, indent=2),
            ContentType='application/json'
        )
        return f"s3://{self.bucket}/{key}"
    
    def upload_config(self, dataset_id: str, version_id: str, config: Dict) -> str:
        """Upload the dataset configuration to S3."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/dataset_config.json"
        self.client.put_object(
            Bucket=self.bucket, 
            Key=key, 
            Body=json.dumps(config, indent=2),
            ContentType='application/json'
        )
        return f"s3://{self.bucket}/{key}"
    
    def update_latest(self, dataset_id: str, version_id: str) -> str:
        """Update LATEST.json to point to the new version."""
        latest = {
            "latest_version": version_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        key = f"{get_s3_dataset_prefix(dataset_id)}/LATEST.json"
        self.client.put_object(
            Bucket=self.bucket, 
            Key=key, 
            Body=json.dumps(latest, indent=2),
            ContentType='application/json'
        )
        return f"s3://{self.bucket}/{key}"
    
    def get_latest_version(self, dataset_id: str) -> Optional[str]:
        """Get the latest version ID for a dataset."""
        key = f"{get_s3_dataset_prefix(dataset_id)}/LATEST.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data.get("latest_version")
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception:
            return None
    
    def get_config(self, dataset_id: str, version_id: str) -> Optional[Dict]:
        """Get the configuration for a specific version."""
        key = f"{get_s3_version_prefix(dataset_id, version_id)}/dataset_config.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception:
            return None
    
    def list_datasets(self) -> List[str]:
        """List all dataset IDs."""
        prefix = "raw_data/datasets/"
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            datasets = set()
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
                for prefix_obj in page.get('CommonPrefixes', []):
                    # Extract dataset_id from prefix
                    dataset_path = prefix_obj['Prefix']
                    dataset_id = dataset_path.replace(prefix, '').rstrip('/')
                    if dataset_id:
                        datasets.add(dataset_id)
            return sorted(list(datasets))
        except Exception:
            return []
    
    def list_versions(self, dataset_id: str) -> List[str]:
        """List all versions for a dataset."""
        prefix = f"{get_s3_dataset_prefix(dataset_id)}/versions/"
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            versions = set()
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
                for prefix_obj in page.get('CommonPrefixes', []):
                    version_path = prefix_obj['Prefix']
                    version_id = version_path.replace(prefix, '').rstrip('/')
                    if version_id:
                        versions.add(version_id)
            return sorted(list(versions), reverse=True)
        except Exception:
            return []


def get_storage(use_s3: bool = None):
    """
    Get the appropriate storage backend based on environment.
    
    Args:
        use_s3: Force S3 (True) or local (False). If None, auto-detect from environment.
    
    Returns:
        DatasetStorageLocal or DatasetStorageS3 instance
    """
    from app.config import get_config
    config = get_config()
    
    if use_s3 is None:
        use_s3 = config.is_production and config.aws_s3_bucket
    
    if use_s3:
        return DatasetStorageS3(bucket=config.aws_s3_bucket, region=config.aws_region)
    return DatasetStorageLocal()
