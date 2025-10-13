"""Model deployment module for Make MLOps Easy."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import joblib


class ModelDeployer:
    """Handles model deployment for ML pipelines."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model deployer.

        Args:
            config: Deployment configuration dictionary.
        """
        self.config = config
        self.deployment_info = {}

    def create_model_metadata(
        self,
        model_path: str,
        preprocessor_path: str,
        training_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create metadata for deployed model.

        Args:
            model_path: Path to model file.
            preprocessor_path: Path to preprocessor file.
            training_results: Training results dictionary.

        Returns:
            Model metadata dictionary.
        """
        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "training_results": training_results,
            "deployment_time": datetime.now().isoformat(),
            "version": "1.0.0",
            "status": "deployed",
        }

    def save_deployment_artifacts(
        self,
        model,
        preprocessor,
        training_results: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Save all deployment artifacts.

        Args:
            model: Trained model trainer instance.
            preprocessor: Data preprocessor instance.
            training_results: Training results dictionary.
            output_dir: Output directory for artifacts.

        Returns:
            Dictionary with paths to saved artifacts.
        """
        if output_dir is None:
            output_dir = self.config.get("output_dir", "./models")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped deployment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deployment_dir = output_dir / f"deployment_{timestamp}"
        deployment_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = deployment_dir / "model.joblib"
        model.save_model(str(model_path))

        # Save preprocessor
        preprocessor_path = deployment_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)

        # Save metadata
        metadata = self.create_model_metadata(
            str(model_path), str(preprocessor_path), training_results
        )
        metadata_path = deployment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.deployment_info = {
            "deployment_dir": str(deployment_dir),
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "metadata_path": str(metadata_path),
        }

        return self.deployment_info

    def create_endpoint_script(self, deployment_dir: str) -> str:
        """Create a simple prediction endpoint script.

        Args:
            deployment_dir: Deployment directory path.

        Returns:
            Path to endpoint script.
        """
        endpoint_script = Path(deployment_dir) / "predict.py"

        script_content = '''#!/usr/bin/env python3
"""Simple prediction endpoint for deployed model."""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path

def load_artifacts():
    """Load model and preprocessor."""
    base_dir = Path(__file__).parent
    model_data = joblib.load(base_dir / "model.joblib")
    preprocessor = joblib.load(base_dir / "preprocessor.joblib")
    return model_data["model"], preprocessor

def predict(data_path):
    """Make predictions on new data."""
    model, preprocessor = load_artifacts()
    
    # Load data
    df = preprocessor.load_data(data_path)
    
    # Preprocess
    X, _ = preprocessor.prepare_data(df, target_column=None, fit=False)
    
    # Predict
    predictions = model.predict(X)
    
    return predictions.tolist()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    predictions = predict(data_path)
    print(json.dumps({"predictions": predictions}))
'''

        with open(endpoint_script, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(endpoint_script, 0o755)

        return str(endpoint_script)

    def deploy(
        self,
        model,
        preprocessor,
        training_results: Dict[str, Any],
        create_endpoint: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Deploy model with all artifacts.

        Args:
            model: Trained model trainer instance.
            preprocessor: Data preprocessor instance.
            training_results: Training results dictionary.
            create_endpoint: Whether to create prediction endpoint script.

        Returns:
            Deployment information dictionary.
        """
        # Save artifacts
        deployment_info = self.save_deployment_artifacts(
            model, preprocessor, training_results
        )

        # Create endpoint if requested
        if create_endpoint is None:
            create_endpoint = self.config.get("create_endpoint", False)

        if create_endpoint:
            endpoint_path = self.create_endpoint_script(
                deployment_info["deployment_dir"]
            )
            deployment_info["endpoint_path"] = endpoint_path

        return deployment_info

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get current deployment information.

        Returns:
            Deployment information dictionary.
        """
        return self.deployment_info

    def load_deployed_model(self, deployment_dir: str) -> tuple:
        """Load a deployed model and preprocessor.

        Args:
            deployment_dir: Path to deployment directory.

        Returns:
            Tuple of (model_data, preprocessor, metadata).
        """
        deployment_dir = Path(deployment_dir)

        model_data = joblib.load(deployment_dir / "model.joblib")
        preprocessor = joblib.load(deployment_dir / "preprocessor.joblib")

        metadata = {}
        metadata_path = deployment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return model_data, preprocessor, metadata
