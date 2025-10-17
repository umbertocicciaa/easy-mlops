"""Composable observability step implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import numpy as np
from datetime import datetime


class ObservabilityStep(ABC):
    """Contract for observability steps that react to monitoring events."""

    name: str

    def __init__(self, **params):
        self.params = params

    @classmethod
    def from_config(
        cls, params: Optional[Dict[str, object]] = None
    ) -> "ObservabilityStep":
        params = params or {}
        return cls(**params)

    # Hook methods ---------------------------------------------------------

    def on_log_metrics(self, metrics: Dict[str, float], model_version: str) -> None:
        """React to metrics logging events."""

    def on_log_prediction(
        self,
        input_data,
        prediction,
        model_version: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """React to prediction logging events."""

    def check_metric_threshold(self, metric_name: str, value: float) -> Optional[bool]:
        """Return True/False if this step can evaluate threshold, or None to defer."""
        return None

    def get_metrics_summary(self) -> Optional[Dict[str, object]]:
        """Return metrics summary information produced by this step."""
        return None

    def get_predictions_summary(self) -> Optional[Dict[str, object]]:
        """Return prediction summary information produced by this step."""
        return None

    def save(self, output_dir: Path) -> Dict[str, str]:
        """Persist step state to disk and return a mapping of log types to paths."""
        return {}

    def load(self, log_dir: Path) -> None:
        """Restore step state from disk."""


@dataclass
class MetricsLoggerStep(ObservabilityStep):
    """Collect model metric logs and provide summary statistics."""

    enabled: bool = True
    name: str = "metrics_logger"
    history: List[Dict[str, object]] = field(default_factory=list, init=False)

    def on_log_metrics(self, metrics: Dict[str, float], model_version: str) -> None:
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "metrics": metrics,
        }
        self.history.append(entry)

    def get_metrics_summary(self) -> Optional[Dict[str, object]]:
        if not self.history:
            return {"message": "No metrics logged yet"}

        latest_metrics = self.history[-1]

        summary: Dict[str, object] = {
            "latest_metrics": latest_metrics,
            "total_logs": len(self.history),
            "first_log": self.history[0]["timestamp"],
            "last_log": latest_metrics["timestamp"],
        }

        if len(self.history) > 1:
            trends: Dict[str, Dict[str, float]] = {}
            metric_names = latest_metrics["metrics"].keys()

            for metric_name in metric_names:
                values = [
                    entry["metrics"].get(metric_name)
                    for entry in self.history
                    if metric_name in entry["metrics"]
                ]
                if values:
                    arr = np.array(values, dtype=float)
                    trends[metric_name] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }

            if trends:
                summary["trends"] = trends

        return summary

    def save(self, output_dir: Path) -> Dict[str, str]:
        if not self.history:
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics_history.json"
        with metrics_path.open("w") as fp:
            json.dump(self.history, fp, indent=2)
        return {"metrics": str(metrics_path)}

    def load(self, log_dir: Path) -> None:
        metrics_path = log_dir / "metrics_history.json"
        if metrics_path.exists():
            with metrics_path.open("r") as fp:
                self.history = json.load(fp)


@dataclass
class PredictionsLoggerStep(ObservabilityStep):
    """Collect model prediction logs."""

    enabled: bool = True
    name: str = "predictions_logger"
    log: List[Dict[str, object]] = field(default_factory=list, init=False)

    def on_log_prediction(
        self,
        input_data,
        prediction,
        model_version: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.enabled:
            return

        cleaned_prediction = (
            prediction if isinstance(prediction, (int, float, str)) else str(prediction)
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "prediction": cleaned_prediction,
            "metadata": metadata or {},
        }
        self.log.append(entry)

    def get_predictions_summary(self) -> Optional[Dict[str, object]]:
        if not self.log:
            return {"message": "No predictions logged yet"}

        return {
            "total_predictions": len(self.log),
            "first_prediction": self.log[0]["timestamp"],
            "last_prediction": self.log[-1]["timestamp"],
            "recent_predictions": self.log[-5:],
        }

    def save(self, output_dir: Path) -> Dict[str, str]:
        if not self.log:
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions_log.json"
        with predictions_path.open("w") as fp:
            json.dump(self.log, fp, indent=2)
        return {"predictions": str(predictions_path)}

    def load(self, log_dir: Path) -> None:
        predictions_path = log_dir / "predictions_log.json"
        if predictions_path.exists():
            with predictions_path.open("r") as fp:
                self.log = json.load(fp)


@dataclass
class MetricThresholdStep(ObservabilityStep):
    """Evaluate metric thresholds to trigger alerts."""

    default_threshold: float = 0.8
    higher_is_better: Iterable[str] = (
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "r2_score",
    )
    lower_is_better: Iterable[str] = ("mse", "rmse", "mae", "log_loss")
    metric_thresholds: Optional[Dict[str, float]] = None
    metric_directions: Optional[Dict[str, str]] = None
    name: str = "metric_threshold"

    def __post_init__(self):
        self.metric_thresholds = self.metric_thresholds or {}
        self.metric_directions = self.metric_directions or {}

        higher = set(self.higher_is_better)
        lower = set(self.lower_is_better)

        for metric, direction in self.metric_directions.items():
            if direction == "higher":
                higher.add(metric)
                lower.discard(metric)
            elif direction == "lower":
                lower.add(metric)
                higher.discard(metric)

        self._higher = higher
        self._lower = lower

    def check_metric_threshold(self, metric_name: str, value: float) -> Optional[bool]:
        threshold = self.metric_thresholds.get(metric_name, self.default_threshold)

        if metric_name in self._lower:
            return value > threshold

        # Default behaviour treats metrics as higher-is-better
        return value < threshold


DEFAULT_STEP_REGISTRY: Dict[str, type[ObservabilityStep]] = {
    MetricsLoggerStep.name: MetricsLoggerStep,
    PredictionsLoggerStep.name: PredictionsLoggerStep,
    MetricThresholdStep.name: MetricThresholdStep,
}

__all__ = [
    "ObservabilityStep",
    "MetricsLoggerStep",
    "PredictionsLoggerStep",
    "MetricThresholdStep",
    "DEFAULT_STEP_REGISTRY",
]
