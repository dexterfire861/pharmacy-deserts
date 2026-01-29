"""
Model registry storage for versioned ML models.
Handles storing, versioning, and retrieving trained models and their metadata.

S3 Structure:
    models/
    ├── LATEST.json                    # Points to latest model version
    └── versions/{model_version_id}/
        ├── model_config.json          # Training config + data lineage
        ├── national_ifae_rank.csv     # Main model output
        ├── glm_coefficients.csv       # Model coefficients
        └── metrics.json               # Training metrics
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from storage.datasets import generate_version_id


@dataclass
class ModelInfo:
    """Information about a trained model version."""
    model_version: str
    trained_at: str
    dataset_id: str
    dataset_version: str
    data_sources: List[str]
    row_count: int
    feature_columns: List[str]
    training_duration_sec: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_s3_model_prefix() -> str:
    """Get the S3 prefix for models."""
    return "models"


def get_s3_model_version_prefix(version_id: str) -> str:
    """Get the S3 prefix for a specific model version."""
    return f"models/versions/{version_id}"


def build_model_config(
    model_version: str,
    dataset_id: str,
    dataset_version: str,
    data_sources: List[str],
    row_count: int,
    feature_columns: List[str],
    scoring_config: Optional[Dict] = None,
    training_params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Build the model configuration JSON structure.
    
    This stores all metadata needed for traceability:
    - What data was used (dataset_id, version, sources)
    - What features were included
    - Training parameters
    """
    return {
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "data_sources": data_sources,
        "row_count": row_count,
        "feature_columns": feature_columns,
        "scoring_config": scoring_config,
        "training_params": training_params or {},
        "status": "active"
    }


class ModelRegistryLocal:
    """Local filesystem model registry for development."""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "versions").mkdir(exist_ok=True)
    
    def _version_path(self, version_id: str) -> Path:
        return self.base_path / "versions" / version_id
    
    def save_model_version(
        self,
        version_id: str,
        model_config: Dict[str, Any],
        result_files: Dict[str, bytes],
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save a new model version.
        
        Args:
            version_id: Unique version identifier
            model_config: Model configuration/metadata
            result_files: Dict of filename -> file bytes (e.g., CSVs)
            metrics: Training metrics
            
        Returns:
            Path to saved model version
        """
        version_path = self._version_path(version_id)
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Add metrics to config
        if metrics:
            model_config["metrics"] = metrics
        
        # Save model config
        config_path = version_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save result files
        for filename, content in result_files.items():
            file_path = version_path / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)
        
        # Save metrics separately for easy access
        if metrics:
            metrics_path = version_path / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return str(version_path)
    
    def update_latest(self, version_id: str):
        """Update LATEST.json to point to the given version."""
        latest_path = self.base_path / "LATEST.json"
        latest_data = {
            "latest_version": version_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        with open(latest_path, 'w') as f:
            json.dump(latest_data, f, indent=2)
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest model version ID."""
        latest_path = self.base_path / "LATEST.json"
        if not latest_path.exists():
            return None
        with open(latest_path) as f:
            data = json.load(f)
        return data.get("latest_version")
    
    def get_model_config(self, version_id: str) -> Optional[Dict]:
        """Get configuration for a specific model version."""
        config_path = self._version_path(version_id) / "model_config.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return json.load(f)
    
    def get_model_metrics(self, version_id: str) -> Optional[Dict]:
        """Get metrics for a specific model version."""
        metrics_path = self._version_path(version_id) / "metrics.json"
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)
    
    def list_model_versions(self) -> List[Dict]:
        """List all model versions with their metadata."""
        versions_path = self.base_path / "versions"
        if not versions_path.exists():
            return []
        
        versions = []
        for version_dir in sorted(versions_path.iterdir(), reverse=True):
            if version_dir.is_dir():
                config = self.get_model_config(version_dir.name)
                if config:
                    versions.append({
                        "version_id": version_dir.name,
                        "trained_at": config.get("trained_at"),
                        "dataset_version": config.get("dataset_version"),
                        "row_count": config.get("row_count"),
                        "metrics": config.get("metrics")
                    })
        return versions
    
    def get_model_file(self, version_id: str, filename: str) -> Optional[bytes]:
        """Get a specific file from a model version."""
        file_path = self._version_path(version_id) / filename
        if not file_path.exists():
            return None
        return file_path.read_bytes()


class ModelRegistryS3:
    """S3-backed model registry for production."""
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.region = region
        self._client = None
    
    @property
    def s3_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client('s3', region_name=self.region)
        return self._client
    
    def _version_prefix(self, version_id: str) -> str:
        return f"models/versions/{version_id}"
    
    def save_model_version(
        self,
        version_id: str,
        model_config: Dict[str, Any],
        result_files: Dict[str, bytes],
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Save a new model version to S3."""
        prefix = self._version_prefix(version_id)
        
        # Add metrics to config
        if metrics:
            model_config["metrics"] = metrics
        
        # Upload model config
        config_key = f"{prefix}/model_config.json"
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=config_key,
            Body=json.dumps(model_config, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        # Upload result files
        for filename, content in result_files.items():
            file_key = f"{prefix}/{filename}"
            body = content if isinstance(content, bytes) else content.encode('utf-8')
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=file_key,
                Body=body
            )
        
        # Upload metrics separately
        if metrics:
            metrics_key = f"{prefix}/metrics.json"
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=metrics_key,
                Body=json.dumps(metrics, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
        
        return f"s3://{self.bucket}/{prefix}"
    
    def update_latest(self, version_id: str):
        """Update LATEST.json in S3."""
        latest_data = {
            "latest_version": version_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key="models/LATEST.json",
            Body=json.dumps(latest_data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest model version ID from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key="models/LATEST.json"
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data.get("latest_version")
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception:
            return None
    
    def get_model_config(self, version_id: str) -> Optional[Dict]:
        """Get configuration for a specific model version from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"{self._version_prefix(version_id)}/model_config.json"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception:
            return None
    
    def get_model_metrics(self, version_id: str) -> Optional[Dict]:
        """Get metrics for a specific model version from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"{self._version_prefix(version_id)}/metrics.json"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception:
            return None
    
    def list_model_versions(self) -> List[Dict]:
        """List all model versions from S3."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            versions = set()
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix="models/versions/"):
                for obj in page.get('Contents', []):
                    # Extract version ID from path
                    parts = obj['Key'].split('/')
                    if len(parts) >= 3:
                        versions.add(parts[2])
            
            # Get config for each version
            result = []
            for version_id in sorted(versions, reverse=True):
                config = self.get_model_config(version_id)
                if config:
                    result.append({
                        "version_id": version_id,
                        "trained_at": config.get("trained_at"),
                        "dataset_version": config.get("dataset_version"),
                        "row_count": config.get("row_count"),
                        "metrics": config.get("metrics")
                    })
            return result
        except Exception:
            return []
    
    def get_model_file(self, version_id: str, filename: str) -> Optional[bytes]:
        """Get a specific file from a model version in S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"{self._version_prefix(version_id)}/{filename}"
            )
            return response['Body'].read()
        except Exception:
            return None


def get_model_registry(use_s3: bool = None):
    """
    Get the appropriate model registry based on environment.
    
    Args:
        use_s3: Force S3 (True) or local (False). If None, auto-detect.
        
    Returns:
        ModelRegistryLocal or ModelRegistryS3 instance
    """
    from app.config import get_config
    
    config = get_config()
    
    if use_s3 is None:
        use_s3 = config.is_production and config.aws_s3_bucket
    
    if use_s3:
        return ModelRegistryS3(config.aws_s3_bucket, config.aws_region)
    else:
        return ModelRegistryLocal("models")

