"""Model observability module for Make MLOps Easy."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class ModelMonitor:
    """Handles model monitoring and observability."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model monitor.

        Args:
            config: Observability configuration dictionary.
        """
        self.config = config
        self.metrics_history = []
        self.predictions_log = []

    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        model_version: str = "1.0.0",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a prediction for monitoring.

        Args:
            input_data: Input data for prediction.
            prediction: Model prediction.
            model_version: Version of the model used.
            metadata: Additional metadata to log.
        """
        if not self.config.get("log_predictions", True):
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "prediction": (
                prediction
                if isinstance(prediction, (int, float, str))
                else str(prediction)
            ),
            "metadata": metadata or {},
        }

        self.predictions_log.append(log_entry)

    def log_metrics(
        self, metrics: Dict[str, float], model_version: str = "1.0.0"
    ) -> None:
        """Log model metrics.

        Args:
            metrics: Dictionary of metric names and values.
            model_version: Version of the model.
        """
        if not self.config.get("track_metrics", True):
            return

        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "metrics": metrics,
        }

        self.metrics_history.append(metric_entry)

    def check_metric_threshold(self, metric_name: str, value: float) -> bool:
        """Check if a metric exceeds alert threshold.

        Args:
            metric_name: Name of the metric.
            value: Metric value.

        Returns:
            True if alert should be triggered, False otherwise.
        """
        threshold = self.config.get("alert_threshold", 0.8)

        # For accuracy-like metrics (higher is better)
        if metric_name in ["accuracy", "f1_score", "r2_score"]:
            return value < threshold

        # For error-like metrics (lower is better)
        if metric_name in ["mse", "rmse", "mae"]:
            return value > threshold

        return False

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics.

        Returns:
            Dictionary with metrics summary.
        """
        if not self.metrics_history:
            return {"message": "No metrics logged yet"}

        latest_metrics = self.metrics_history[-1]

        summary = {
            "latest_metrics": latest_metrics,
            "total_logs": len(self.metrics_history),
            "first_log": self.metrics_history[0]["timestamp"],
            "last_log": latest_metrics["timestamp"],
        }

        # Calculate metric trends if multiple entries
        if len(self.metrics_history) > 1:
            trends = {}
            metric_names = latest_metrics["metrics"].keys()

            for metric_name in metric_names:
                values = [
                    entry["metrics"].get(metric_name)
                    for entry in self.metrics_history
                    if metric_name in entry["metrics"]
                ]
                if values:
                    trends[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                    }

            summary["trends"] = trends

        return summary

    def get_predictions_summary(self) -> Dict[str, Any]:
        """Get summary of logged predictions.

        Returns:
            Dictionary with predictions summary.
        """
        if not self.predictions_log:
            return {"message": "No predictions logged yet"}

        return {
            "total_predictions": len(self.predictions_log),
            "first_prediction": self.predictions_log[0]["timestamp"],
            "last_prediction": self.predictions_log[-1]["timestamp"],
            "recent_predictions": self.predictions_log[-5:],
        }

    def save_logs(self, output_dir: str) -> Dict[str, str]:
        """Save logs to files.

        Args:
            output_dir: Directory to save logs.

        Returns:
            Dictionary with paths to saved log files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save metrics history
        if self.metrics_history:
            metrics_path = output_dir / "metrics_history.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            saved_files["metrics"] = str(metrics_path)

        # Save predictions log
        if self.predictions_log:
            predictions_path = output_dir / "predictions_log.json"
            with open(predictions_path, "w") as f:
                json.dump(self.predictions_log, f, indent=2)
            saved_files["predictions"] = str(predictions_path)

        return saved_files

    def load_logs(self, log_dir: str) -> None:
        """Load logs from files.

        Args:
            log_dir: Directory containing log files.
        """
        log_dir = Path(log_dir)

        # Load metrics history
        metrics_path = log_dir / "metrics_history.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics_history = json.load(f)

        # Load predictions log
        predictions_path = log_dir / "predictions_log.json"
        if predictions_path.exists():
            with open(predictions_path, "r") as f:
                self.predictions_log = json.load(f)

    def generate_report(self) -> str:
        """Generate a monitoring report.

        Returns:
            Formatted report string.
        """
        report = ["=" * 60]
        report.append("MODEL MONITORING REPORT")
        report.append("=" * 60)
        report.append("")

        # Metrics summary
        report.append("METRICS SUMMARY:")
        report.append("-" * 60)
        metrics_summary = self.get_metrics_summary()
        if "message" in metrics_summary:
            report.append(f"  {metrics_summary['message']}")
        else:
            report.append(f"  Total logs: {metrics_summary['total_logs']}")
            report.append(f"  First log: {metrics_summary['first_log']}")
            report.append(f"  Last log: {metrics_summary['last_log']}")

            if "latest_metrics" in metrics_summary:
                report.append(f"  Latest metrics:")
                for key, value in metrics_summary["latest_metrics"]["metrics"].items():
                    report.append(f"    {key}: {value:.4f}")

        report.append("")

        # Predictions summary
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
