# pharmacy_deserts/data_processing/glm_training.py
"""
GLM + Expected-Access Model for Pharmacy Desert Prediction
This module wraps the training logic from data/finalfinalfinal_training.py
Implements: OOF Poisson GLM with per-state calibration, NB variance, neighbor QA
"""

import os
import sys
from pathlib import Path

def run_glm_training(force_retrain=False):
    """
    Run the GLM + Hybrid training pipeline.
    
    Args:
        force_retrain: If False, skip if results already exist
        
    Returns:
        bool: True if training ran successfully, False otherwise
    """
    results_dir = Path("results")
    main_output = results_dir / "national_ifae_rank.csv"
    
    # Check if results exist
    if not force_retrain and main_output.exists():
        print(f"✓ Using existing trained model from {main_output}")
        return True
    
    print("=" * 80)
    print("RUNNING GLM + EXPECTED-ACCESS TRAINING PIPELINE")
    print("=" * 80)
    
    try:
        # Import and run the training script
        # We'll execute it as a module
        training_script = Path("data/finalfinalfinal_training.py")
        
        if not training_script.exists():
            print(f"✗ Training script not found: {training_script}")
            return False
        
        # Execute the training script in the current environment
        print(f"Executing training script: {training_script}")
        with open(training_script, 'r') as f:
            code = f.read()
        
        # Execute in a clean namespace
        exec(code, {'__name__': '__main__'})
        
        # Verify outputs were created
        if main_output.exists():
            print(f"\n✓ Training complete! Results written to {results_dir}/")
            return True
        else:
            print(f"\n✗ Training completed but output file not found: {main_output}")
            return False
            
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GLM+Expected-Access pharmacy desert model")
    parser.add_argument("--force", action="store_true", help="Force retrain even if results exist")
    args = parser.parse_args()
    
    success = run_glm_training(force_retrain=args.force)
    sys.exit(0 if success else 1)

