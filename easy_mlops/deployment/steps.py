"""Composable deployment step implementations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import joblib


@dataclass
class DeploymentContext:
    """Mutable container passed between deployment steps."""

    model: Any
    preprocessor: Any
    training_results: Dict[str, Any]
    base_output_dir: Path
    config: Dict[str, Any]
    metadata_factory: Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
    deployment_dir: Optional[Path] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    endpoint_template: Optional[str] = None
    endpoint_writer: Optional[Callable[[Path, str, str], str]] = None


class DeploymentStep:
    """Contract for deployment steps assembled into a pipeline."""

    name: str

    def __init__(self, **params: Any):
        self.params = params

    @classmethod
    def from_config(cls, params: Optional[Dict[str, Any]] = None) -> "DeploymentStep":
        params = params or {}
        return cls(**params)

    def run(
        self, context: DeploymentContext
    ) -> None:  # pragma: no cover - abstract hook
        raise NotImplementedError


@dataclass
class CreateDeploymentDirectoryStep(DeploymentStep):
    """Create the deployment directory under the output root."""

    prefix: str = "deployment"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    name: str = "create_directory"

    def run(self, context: DeploymentContext) -> None:
        base_dir = context.base_output_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(self.timestamp_format)
        deployment_dir = base_dir / f"{self.prefix}_{timestamp}"

        # Ensure uniqueness if multiple deployments happen in the same second.
        counter = 1
        while deployment_dir.exists():
            deployment_dir = base_dir / f"{self.prefix}_{timestamp}_{counter}"
            counter += 1

        deployment_dir.mkdir(parents=True, exist_ok=True)
        context.deployment_dir = deployment_dir
        context.artifacts["deployment_dir"] = str(deployment_dir)


@dataclass
class SaveModelStep(DeploymentStep):
    """Persist the trained model via the trainer interface."""

    filename: str = "model.joblib"
    name: str = "save_model"

    def run(self, context: DeploymentContext) -> None:
        if context.deployment_dir is None:
            raise RuntimeError(
                "Deployment directory must be created before saving the model."
            )

        model_path = context.deployment_dir / self.filename
        context.model.save_model(str(model_path))
        context.artifacts["model_path"] = str(model_path)


@dataclass
class SavePreprocessorStep(DeploymentStep):
    """Persist the fitted preprocessor alongside the model."""

    filename: str = "preprocessor.joblib"
    name: str = "save_preprocessor"

    def run(self, context: DeploymentContext) -> None:
        if context.deployment_dir is None:
            raise RuntimeError(
                "Deployment directory must be created before saving the preprocessor."
            )

        preprocessor_path = context.deployment_dir / self.filename
        joblib.dump(context.preprocessor, preprocessor_path)
        context.artifacts["preprocessor_path"] = str(preprocessor_path)


@dataclass
class SaveMetadataStep(DeploymentStep):
    """Generate and persist deployment metadata."""

    filename: str = "metadata.json"
    name: str = "save_metadata"

    def run(self, context: DeploymentContext) -> None:
        if context.deployment_dir is None:
            raise RuntimeError(
                "Deployment directory must be created before saving metadata."
            )

        model_path = context.artifacts.get("model_path")
        preprocessor_path = context.artifacts.get("preprocessor_path")

        if not model_path or not preprocessor_path:
            raise RuntimeError(
                "Model and preprocessor paths must be available before creating metadata."
            )

        metadata = context.metadata_factory(
            model_path,
            preprocessor_path,
            context.training_results,
        )
        metadata_path = context.deployment_dir / self.filename
        metadata_path.write_text(json.dumps(metadata, indent=2))

        context.metadata = metadata
        context.artifacts["metadata_path"] = str(metadata_path)


DEFAULT_ENDPOINT_TEMPLATE = '''#!/usr/bin/env python3
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
    df = preprocessor.load_data(data_path)
    X, _ = preprocessor.prepare_data(df, target_column=None, fit=False)
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


@dataclass
class EndpointScriptStep(DeploymentStep):
    """Optionally generate an executable prediction helper script."""

    enabled: bool = False
    filename: str = "predict.py"
    executable_mode: int = 0o755
    template: Optional[str] = None
    name: str = "endpoint_script"

    def run(self, context: DeploymentContext) -> None:
        if not self.enabled:
            return

        if context.deployment_dir is None:
            raise RuntimeError(
                "Deployment directory must be created before writing endpoint script."
            )

        script_template = (
            self.template or context.endpoint_template or DEFAULT_ENDPOINT_TEMPLATE
        )
        writer = context.endpoint_writer

        if writer is not None:
            script_path = writer(context.deployment_dir, self.filename, script_template)
        else:
            script_path = context.deployment_dir / self.filename
            script_path.write_text(script_template)
            os.chmod(script_path, self.executable_mode)
            script_path = str(script_path)

        # writer may already return string
        context.artifacts["endpoint_path"] = (
            script_path if isinstance(script_path, str) else str(script_path)
        )


DEFAULT_STEP_REGISTRY: Dict[str, type[DeploymentStep]] = {
    CreateDeploymentDirectoryStep.name: CreateDeploymentDirectoryStep,
    SaveModelStep.name: SaveModelStep,
    SavePreprocessorStep.name: SavePreprocessorStep,
    SaveMetadataStep.name: SaveMetadataStep,
    EndpointScriptStep.name: EndpointScriptStep,
}

__all__ = [
    "DeploymentContext",
    "DeploymentStep",
    "CreateDeploymentDirectoryStep",
    "SaveModelStep",
    "SavePreprocessorStep",
    "SaveMetadataStep",
    "EndpointScriptStep",
    "DEFAULT_ENDPOINT_TEMPLATE",
    "DEFAULT_STEP_REGISTRY",
]
