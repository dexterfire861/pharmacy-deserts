# pharmacy_deserts/data/s3_loaders.py
"""
S3 data loading functions for AWS deployment.
Provides functions to load data files from S3 buckets with local caching.
"""
import os
import io
import tempfile
from pathlib import Path
from functools import lru_cache

import pandas as pd

# Lazy import boto3 to avoid import errors when not deployed to AWS
_s3_client = None


def get_s3_client():
    """Get or create S3 client (lazy initialization)."""
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
    return _s3_client


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URI."""
    return path.startswith('s3://')


def parse_s3_path(s3_path: str) -> tuple:
    """
    Parse an S3 URI into bucket and key.
    
    Args:
        s3_path: S3 URI like 's3://bucket-name/path/to/file.csv'
        
    Returns:
        tuple: (bucket_name, key)
    """
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path = s3_path[5:]  # Remove 's3://'
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key


def get_s3_bucket():
    """Get the configured S3 bucket name from environment."""
    bucket = os.getenv('AWS_S3_BUCKET')
    if not bucket:
        raise ValueError("AWS_S3_BUCKET environment variable not set")
    return bucket


def s3_to_local_path(s3_key: str) -> str:
    """
    Convert an S3 key to a local path equivalent.
    Used when running locally with the same path structure.
    
    Args:
        s3_key: S3 key like 'raw_data/financial_data.csv'
        
    Returns:
        Local path string
    """
    return s3_key


@lru_cache(maxsize=32)
def download_s3_file_to_memory(bucket: str, key: str) -> bytes:
    """
    Download a file from S3 into memory (cached).
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        File contents as bytes
    """
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def read_csv_from_s3(bucket: str, key: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file from S3 into a pandas DataFrame.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        pandas DataFrame
    """
    data = download_s3_file_to_memory(bucket, key)
    return pd.read_csv(io.BytesIO(data), **kwargs)


def read_excel_from_s3(bucket: str, key: str, **kwargs) -> pd.DataFrame:
    """
    Read an Excel file from S3 into a pandas DataFrame.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        **kwargs: Additional arguments passed to pd.read_excel
        
    Returns:
        pandas DataFrame
    """
    data = download_s3_file_to_memory(bucket, key)
    return pd.read_excel(io.BytesIO(data), **kwargs)


def download_s3_directory(bucket: str, prefix: str, local_dir: str) -> str:
    """
    Download all files from an S3 prefix to a local directory.
    Used for directory-based data like pharmacy Excel files.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (like a directory path)
        local_dir: Local directory to download to
        
    Returns:
        Path to local directory with downloaded files
    """
    import boto3
    
    s3 = get_s3_client()
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # List objects with the given prefix
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            # Skip "directory" markers
            if key.endswith('/'):
                continue
                
            # Get relative path from prefix
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                relative_path = Path(key).name
                
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            s3.download_file(bucket, key, str(local_file))
    
    return str(local_path)


def get_data_path(relative_path: str) -> str:
    """
    Get the appropriate data path based on environment.
    
    In production (ENVIRONMENT=production), returns S3-prefixed path.
    In development, returns local path.
    
    Args:
        relative_path: Relative path like 'raw_data/financial_data.csv'
        
    Returns:
        Full path (local or S3 URI)
    """
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == 'production':
        bucket = get_s3_bucket()
        return f"s3://{bucket}/{relative_path}"
    else:
        return relative_path


def smart_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV from either local filesystem or S3 based on path.
    
    Args:
        path: Local path or S3 URI
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        pandas DataFrame
    """
    if is_s3_path(path):
        bucket, key = parse_s3_path(path)
        return read_csv_from_s3(bucket, key, **kwargs)
    else:
        return pd.read_csv(path, **kwargs)


def smart_read_excel(path: str, **kwargs) -> pd.DataFrame:
    """
    Read Excel from either local filesystem or S3 based on path.
    
    Args:
        path: Local path or S3 URI
        **kwargs: Additional arguments passed to pd.read_excel
        
    Returns:
        pandas DataFrame
    """
    if is_s3_path(path):
        bucket, key = parse_s3_path(path)
        return read_excel_from_s3(bucket, key, **kwargs)
    else:
        return pd.read_excel(path, **kwargs)


def smart_load_directory(path: str) -> str:
    """
    Load a directory from either local filesystem or S3.
    
    For S3 paths, downloads to a temporary directory.
    For local paths, returns the path as-is.
    
    Args:
        path: Local path or S3 URI to a directory
        
    Returns:
        Local directory path (original or temp download location)
    """
    if is_s3_path(path):
        bucket, key = parse_s3_path(path)
        # Create a temp directory for the download
        temp_dir = tempfile.mkdtemp(prefix='pharmacy_data_')
        return download_s3_directory(bucket, key, temp_dir)
    else:
        return path


def clear_s3_cache():
    """Clear the S3 file download cache."""
    download_s3_file_to_memory.cache_clear()

