"""Model deployment module for Make MLOps Easy."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from easy_mlops.deployment.steps import (
    DEFAULT_ENDPOINT_TEMPLATE,
    DEFAULT_STEP_REGISTRY,
    DeploymentContext,
    DeploymentStep,
    EndpointScriptStep,
)


class ModelDeployer:
    """Handles model deployment for ML pipelines."""

    STEP_REGISTRY: Dict[str, Type[DeploymentStep]] = DEFAULT_STEP_REGISTRY.copy()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model deployer.

        Args:
            config: Deployment configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}
        self.steps: List[DeploymentStep] = self._initialize_steps()
        self.deployment_info: Dict[str, str] = {}

    @classmethod
    def register_step(cls, step: Type[DeploymentStep]) -> None:
        """Register a custom deployment step class."""
        if not issubclass(step, DeploymentStep):
            raise TypeError("Custom step must inherit from DeploymentStep.")
        cls.STEP_REGISTRY[step.name] = step

    def get_step(self, name: str) -> Optional[DeploymentStep]:
        """Retrieve a step instance by its registry name."""
        return next((step for step in self.steps if step.name == name), None)

    def create_model_metadata(
        self,
        model_path: str,
        preprocessor_path: str,
        training_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create metadata for deployed model."""
        deployment_time = (
            self.config.get("deployment_time") or datetime.now().isoformat()
        )

        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "training_results": training_results,
            "deployment_time": deployment_time,
            "version": self.config.get("version", "1.0.0"),
            "status": "deployed",
        }

    def save_deployment_artifacts(
        self,
        model,
        preprocessor,
        training_results: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Save all deployment artifacts."""
        base_output = Path(output_dir or self.config.get("output_dir", "./models"))

        context = DeploymentContext(
            model=model,
            preprocessor=preprocessor,
            training_results=training_results,
            base_output_dir=base_output,
            config=self.config,
            metadata_factory=self.create_model_metadata,
            endpoint_template=self.config.get("endpoint_template"),
            endpoint_writer=self._write_endpoint_script,
        )

        for step in self.steps:
            step.run(context)

        self.deployment_info = dict(context.artifacts)
        return self.deployment_info

    def create_endpoint_script(self, deployment_dir: str) -> str:
        """Create a simple prediction endpoint script."""
        deployment_path = Path(deployment_dir)
        deployment_path.mkdir(parents=True, exist_ok=True)

        template = self.config.get("endpoint_template", DEFAULT_ENDPOINT_TEMPLATE)
        return self._write_endpoint_script(
            deployment_path,
            self.config.get("endpoint_filename", "predict.py"),
            template,
        )

    def deploy(
        self,
        model,
        preprocessor,
        training_results: Dict[str, Any],
        create_endpoint: Optional[bool] = None,
    ) -> Dict[str, str]:
        """Deploy model with all artifacts."""
        deployment_info = self.save_deployment_artifacts(
            model, preprocessor, training_results
        )

        if create_endpoint is None:
            create_endpoint = self.config.get("create_endpoint", False)

        endpoint_already_created = "endpoint_path" in deployment_info

        if create_endpoint and not endpoint_already_created:
            endpoint_path = self.create_endpoint_script(
                deployment_info["deployment_dir"]
            )
            deployment_info["endpoint_path"] = endpoint_path

        self.deployment_info = deployment_info
        return deployment_info

    def get_deployment_info(self) -> Dict[str, str]:
        """Get current deployment information."""
        return self.deployment_info

    def load_deployed_model(self, deployment_dir: str) -> tuple:
        """Load a deployed model and preprocessor."""
        deployment_path = Path(deployment_dir)

        import joblib
        import json

        model_data = joblib.load(deployment_path / "model.joblib")
        preprocessor = joblib.load(deployment_path / "preprocessor.joblib")

        metadata = {}
        metadata_path = deployment_path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        return model_data, preprocessor, metadata

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _initialize_steps(self) -> List[DeploymentStep]:
        steps_config = self.config.get("steps")
        if steps_config:
            return [self._build_step_from_spec(spec) for spec in steps_config]

        steps: List[DeploymentStep] = [
            self._create_step(
                "create_directory",
                {"prefix": self.config.get("deployment_prefix", "deployment")},
            ),
            self._create_step("save_model", {}),
            self._create_step("save_preprocessor", {}),
            self._create_step(
                "save_metadata",
                {"filename": self.config.get("metadata_filename", "metadata.json")},
            ),
        ]

        if self.config.get("create_endpoint", False):
            steps.append(
                self._create_step(
                    EndpointScriptStep.name,
                    {
                        "enabled": True,
                        "filename": self.config.get("endpoint_filename", "predict.py"),
                    },
                )
            )

        return steps

    def _create_step(
        self, step_name: str, params: Optional[Dict[str, Any]] = None
    ) -> DeploymentStep:
        params = params or {}
        registry = self.STEP_REGISTRY

        if step_name not in registry:
            raise ValueError(f"Unknown deployment step: {step_name}")

        step_cls = registry[step_name]
        return step_cls.from_config(params)

    def _build_step_from_spec(self, spec: Any) -> DeploymentStep:
        if isinstance(spec, str):
            return self._create_step(spec)

        if isinstance(spec, dict):
            step_type = spec.get("type")
            if not step_type:
                raise ValueError(
                    "Step configuration dictionaries must include a 'type' key."
                )
            params = spec.get("params") or {}
            return self._create_step(step_type, params)

        raise TypeError("Step specification must be a string or dict.")

    @staticmethod
    def _write_endpoint_script(
        deployment_dir: Path, filename: str, template: str
    ) -> str:
        script_path = deployment_dir / filename
        script_path.write_text(template)
        script_path.chmod(0o755)
        return str(script_path)
