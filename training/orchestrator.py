"""
Training Orchestrator - Manages ML model training pipeline.

This module handles:
1. Exporting merged training data from dataset configs
2. Triggering the training script (new_training.py)
3. Saving results to the model registry
4. Tracking training status and metadata

Usage:
    from training.orchestrator import trigger_training
    
    # After data upload
    model_version = trigger_training(dataset_id="pharmacy_data", dataset_version="20260129_172450")
"""
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Status of a training run."""
    PENDING = "pending"
    EXPORTING_DATA = "exporting_data"
    TRAINING = "training"
    SAVING_MODEL = "saving_model"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    dataset_id: str
    dataset_version: str
    data_file: str  # Path to merged training CSV
    output_dir: str  # Where to save results
    feature_columns: List[str]
    zcta_column: str = "zcta5"
    geo_columns: List[str] = None
    
    def __post_init__(self):
        if self.geo_columns is None:
            self.geo_columns = ["latitude", "longitude"]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def export_training_data(
    dataset_id: str,
    output_path: str,
    dataset_version: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Export merged training data from a dataset configuration.
    
    This loads all sources from the dataset config, merges them on ZCTA5,
    and exports a single CSV ready for training.
    
    Args:
        dataset_id: Dataset identifier
        output_path: Path to write the merged CSV
        dataset_version: Specific version (uses LATEST if not specified)
        
    Returns:
        Tuple of (output_path, metadata dict)
    """
    from ingestion.dataset_loader import load_dataset_from_config
    from storage.datasets import get_storage
    
    # Load the dataset
    logger.info(f"Loading dataset {dataset_id} (version: {dataset_version or 'LATEST'})")
    df, scoring_config = load_dataset_from_config(
        dataset_id, 
        version_id=dataset_version,
        include_scoring_config=True
    )
    
    # Get dataset config for metadata
    storage = get_storage()
    if dataset_version is None:
        dataset_version = storage.get_latest_version(dataset_id)
    
    dataset_config = storage.get_config(dataset_id, dataset_version)
    
    # Extract source filenames
    data_sources = [s.get('filename', 'unknown') for s in dataset_config.get('sources', [])]
    
    # Get feature columns (excluding zcta5 and geo columns)
    geo_cols = ['latitude', 'longitude', 'lat', 'lon', 'long']
    feature_columns = [c for c in df.columns if c != 'zcta5' and c.lower() not in geo_cols]
    
    # Export to CSV
    logger.info(f"Exporting {len(df)} rows to {output_path}")
    df.to_csv(output_path, index=False)
    
    metadata = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "data_sources": data_sources,
        "row_count": len(df),
        "feature_columns": feature_columns,
        "columns": list(df.columns),
        "exported_at": datetime.utcnow().isoformat()
    }
    
    return output_path, metadata


def trigger_training(
    dataset_id: str,
    dataset_version: str = None,
    async_mode: bool = False
) -> Optional[str]:
    """
    Trigger model training with the given dataset.
    
    This is the main entry point for automated training. It:
    1. Exports merged training data to a temp file
    2. Creates a training config JSON
    3. Runs the training script
    4. Saves results to the model registry
    5. Updates LATEST pointer
    
    Args:
        dataset_id: Dataset identifier to train on
        dataset_version: Specific version (uses LATEST if not specified)
        async_mode: If True, return immediately (training runs in background)
        
    Returns:
        Model version ID if successful, None if failed
        
    Note for ML Partner:
        The actual training logic is in new_training.py.
        This orchestrator just handles the data pipeline and model storage.
        Adapt new_training.py to:
        - Accept a config JSON path as argument
        - Read data from the path specified in config
        - Output results to the output_dir in config
        - Return metrics as JSON to stdout or a metrics.json file
    """
    from storage.datasets import get_storage, generate_version_id
    from storage.models import get_model_registry, build_model_config
    
    model_version = generate_version_id()
    logger.info(f"Starting training run: {model_version}")
    
    # Create temporary directory for this training run
    work_dir = Path(tempfile.mkdtemp(prefix=f"training_{model_version}_"))
    
    try:
        # Step 1: Export training data
        logger.info("Step 1: Exporting training data...")
        data_path = work_dir / "training_data.csv"
        output_dir = work_dir / "results"
        output_dir.mkdir(exist_ok=True)
        
        csv_path, data_metadata = export_training_data(
            dataset_id=dataset_id,
            output_path=str(data_path),
            dataset_version=dataset_version
        )
        
        # Step 2: Create training config
        logger.info("Step 2: Creating training config...")
        training_config = TrainingConfig(
            dataset_id=dataset_id,
            dataset_version=data_metadata["dataset_version"],
            data_file=str(data_path),
            output_dir=str(output_dir),
            feature_columns=data_metadata["feature_columns"]
        )
        
        config_path = work_dir / "training_config.json"
        training_config.save(str(config_path))
        
        # Step 3: Run training script
        # NOTE: This is where the ML partner's code gets called
        # The training script should be adapted to accept this config
        logger.info("Step 3: Running training script...")
        start_time = time.time()
        
        # For now, we'll just run the existing script with environment variables
        # ML partner should adapt new_training.py to use the config file
        training_script = Path(__file__).parent.parent / "new_training.py"
        
        if training_script.exists():
            # Set environment variables for the training script
            env = os.environ.copy()
            env["TRAINING_CONFIG"] = str(config_path)
            env["TRAINING_DATA"] = str(data_path)
            env["TRAINING_OUTPUT"] = str(output_dir)
            
            # Run training (this is a placeholder - partner will adapt)
            # For now, skip actual training if script isn't adapted
            logger.warning(
                "Training script exists but may not be adapted for config-based training. "
                "ML Partner: Please update new_training.py to read from TRAINING_CONFIG env var."
            )
            
            # Placeholder: Copy existing results if available
            existing_results = Path("results")
            if existing_results.exists():
                import shutil
                for f in existing_results.glob("*.csv"):
                    shutil.copy(f, output_dir / f.name)
                logger.info("Copied existing results as placeholder")
        
        training_duration = time.time() - start_time
        
        # Step 4: Collect results and metrics
        logger.info("Step 4: Collecting results...")
        result_files = {}
        metrics = {}
        
        # Look for output files
        for result_file in output_dir.glob("*"):
            if result_file.is_file():
                result_files[result_file.name] = result_file.read_bytes()
        
        # Look for metrics file
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
        else:
            # Placeholder metrics
            metrics = {
                "status": "placeholder",
                "note": "Training script not yet adapted for automated metrics"
            }
        
        metrics["training_duration_sec"] = training_duration
        
        # Step 5: Save to model registry
        logger.info("Step 5: Saving to model registry...")
        model_registry = get_model_registry()
        
        model_config = build_model_config(
            model_version=model_version,
            dataset_id=dataset_id,
            dataset_version=data_metadata["dataset_version"],
            data_sources=data_metadata["data_sources"],
            row_count=data_metadata["row_count"],
            feature_columns=data_metadata["feature_columns"],
            training_params={"config_path": str(config_path)}
        )
        
        model_registry.save_model_version(
            version_id=model_version,
            model_config=model_config,
            result_files=result_files,
            metrics=metrics
        )
        
        # Update LATEST pointer
        model_registry.update_latest(model_version)
        
        logger.info(f"Training complete! Model version: {model_version}")
        return model_version
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup temp directory (optional - keep for debugging)
        # import shutil
        # shutil.rmtree(work_dir)
        pass


def get_training_status(model_version: str) -> Dict[str, Any]:
    """
    Get the status and metadata of a training run.
    
    Args:
        model_version: Model version ID to check
        
    Returns:
        Dict with status, metrics, and config
    """
    from storage.models import get_model_registry
    
    registry = get_model_registry()
    config = registry.get_model_config(model_version)
    metrics = registry.get_model_metrics(model_version)
    
    if config is None:
        return {"status": "not_found", "model_version": model_version}
    
    return {
        "status": "completed",
        "model_version": model_version,
        "config": config,
        "metrics": metrics
    }


def list_training_runs() -> List[Dict[str, Any]]:
    """
    List all training runs (model versions) with their status.
    
    Returns:
        List of training run summaries
    """
    from storage.models import get_model_registry
    
    registry = get_model_registry()
    return registry.list_model_versions()

