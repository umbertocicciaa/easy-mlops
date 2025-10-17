"""Model observability module for Make MLOps Easy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from easy_mlops.observability.steps import (
    DEFAULT_STEP_REGISTRY,
    MetricThresholdStep,
    MetricsLoggerStep,
    ObservabilityStep,
    PredictionsLoggerStep,
)


class ModelMonitor:
    """Handles model monitoring and observability."""

    STEP_REGISTRY: Dict[str, Type[ObservabilityStep]] = DEFAULT_STEP_REGISTRY.copy()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model monitor.

        Args:
            config: Observability configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}
        self.steps: List[ObservabilityStep] = self._initialize_steps()
        self.metrics_history: List[Dict[str, Any]] = []
        self.predictions_log: List[Dict[str, Any]] = []
        self._refresh_step_shortcuts()

    @classmethod
    def register_step(cls, step: Type[ObservabilityStep]) -> None:
        """Register a custom observability step class."""
        if not issubclass(step, ObservabilityStep):
            raise TypeError("Custom step must inherit from ObservabilityStep.")
        cls.STEP_REGISTRY[step.name] = step

    def get_step(self, name: str) -> Optional[ObservabilityStep]:
        """Retrieve a step instance by its registry name."""
        return next((step for step in self.steps if step.name == name), None)

    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        model_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a prediction for monitoring."""
        for step in self.steps:
            step.on_log_prediction(
                input_data=input_data,
                prediction=prediction,
                model_version=model_version,
                metadata=metadata,
            )

        self._refresh_step_shortcuts()

    def log_metrics(
        self, metrics: Dict[str, float], model_version: str = "1.0.0"
    ) -> None:
        """Log model metrics."""
        for step in self.steps:
            step.on_log_metrics(metrics=metrics, model_version=model_version)

        self._refresh_step_shortcuts()

    def check_metric_threshold(self, metric_name: str, value: float) -> bool:
        """Check if a metric exceeds alert threshold."""
        for step in self.steps:
            result = step.check_metric_threshold(metric_name, value)
            if result is not None:
                return result

        # Fallback to default behaviour for backwards compatibility.
        default_threshold = self.config.get("alert_threshold", 0.8)
        if metric_name in {"mse", "rmse", "mae"}:
            return value > default_threshold

        return value < default_threshold

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        step = self.get_step(MetricsLoggerStep.name)
        if step is None:
            return {"message": "Metrics logging disabled"}

        summary = step.get_metrics_summary()
        return summary if summary is not None else {"message": "No metrics logged yet"}

    def get_predictions_summary(self) -> Dict[str, Any]:
        """Get summary of logged predictions."""
        step = self.get_step(PredictionsLoggerStep.name)
        if step is None:
            return {"message": "Prediction logging disabled"}

        summary = step.get_predictions_summary()
        return (
            summary if summary is not None else {"message": "No predictions logged yet"}
        )

    def save_logs(self, output_dir: str) -> Dict[str, str]:
        """Save logs to files."""
        output_path = Path(output_dir)
        saved_files: Dict[str, str] = {}

        for step in self.steps:
            saved_files.update(step.save(output_path))

        return saved_files

    def load_logs(self, log_dir: str) -> None:
        """Load logs from files."""
        log_path = Path(log_dir)
        for step in self.steps:
            step.load(log_path)

        self._refresh_step_shortcuts()

    def generate_report(self) -> str:
        """Generate a monitoring report."""
        report = ["=" * 60, "MODEL MONITORING REPORT", "=" * 60, ""]

        report.append("METRICS SUMMARY:")
        report.append("-" * 60)
        metrics_summary = self.get_metrics_summary()
        if "message" in metrics_summary:
            report.append(f"  {metrics_summary['message']}")
        else:
            report.append(f"  Total logs: {metrics_summary['total_logs']}")
            report.append(f"  First log: {metrics_summary['first_log']}")
            report.append(f"  Last log: {metrics_summary['last_log']}")

            latest = metrics_summary.get("latest_metrics")
            if latest:
                report.append("  Latest metrics:")
                for key, value in latest["metrics"].items():
                    report.append(f"    {key}: {value:.4f}")

        report.append("")
        report.append("PREDICTIONS SUMMARY:")
        report.append("-" * 60)
        pred_summary = self.get_predictions_summary()
        if "message" in pred_summary:
            report.append(f"  {pred_summary['message']}")
        else:
            report.append(f"  Total predictions: {pred_summary['total_predictions']}")
            report.append(f"  First prediction: {pred_summary['first_prediction']}")
            report.append(f"  Last prediction: {pred_summary['last_prediction']}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    # Internal helpers -----------------------------------------------------

    def _initialize_steps(self) -> List[ObservabilityStep]:
        steps_config = self.config.get("steps")

        if steps_config:
            return [self._build_step_from_spec(spec) for spec in steps_config]

        steps: List[ObservabilityStep] = []

        if self.config.get("track_metrics", True):
            steps.append(self._create_step(MetricsLoggerStep.name, {"enabled": True}))

        if self.config.get("log_predictions", True):
            steps.append(
                self._create_step(PredictionsLoggerStep.name, {"enabled": True})
            )

        default_threshold = self.config.get("alert_threshold", 0.8)
        metric_thresholds = self.config.get("metric_thresholds")
        metric_directions = self.config.get("metric_directions")

        steps.append(
            self._create_step(
                MetricThresholdStep.name,
                {
                    "default_threshold": default_threshold,
                    "metric_thresholds": metric_thresholds,
                    "metric_directions": metric_directions,
                },
            )
        )

        return steps

    def _create_step(
        self, step_name: str, params: Optional[Dict[str, Any]] = None
    ) -> ObservabilityStep:
        params = params or {}
        registry = self.STEP_REGISTRY

        if step_name not in registry:
            raise ValueError(f"Unknown observability step: {step_name}")

        step_cls = registry[step_name]
        return step_cls.from_config(params)

    def _build_step_from_spec(self, spec: Any) -> ObservabilityStep:
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

    def _refresh_step_shortcuts(self) -> None:
        metrics_step = self.get_step(MetricsLoggerStep.name)
        self.metrics_history = metrics_step.history if metrics_step else []

        predictions_step = self.get_step(PredictionsLoggerStep.name)
        self.predictions_log = predictions_step.log if predictions_step else []
