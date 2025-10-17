from pathlib import Path

import numpy as np

from easy_mlops.observability import ModelMonitor, ObservabilityStep


def test_log_metrics_and_predictions():
    monitor = ModelMonitor(
        {"track_metrics": True, "log_predictions": True, "alert_threshold": 0.8}
    )

    monitor.log_metrics({"accuracy": 0.92})
    monitor.log_prediction(input_data={"x": 1}, prediction=0.7)

    assert len(monitor.metrics_history) == 1
    assert len(monitor.predictions_log) == 1


def test_alert_threshold_logic():
    monitor = ModelMonitor(
        {"track_metrics": True, "log_predictions": True, "alert_threshold": 0.75}
    )
    assert monitor.check_metric_threshold("accuracy", 0.7) is True
    assert monitor.check_metric_threshold("rmse", 0.5) is False
    assert monitor.check_metric_threshold("rmse", 0.9) is True


def test_save_and_load_logs(tmp_path):
    monitor = ModelMonitor(
        {"track_metrics": True, "log_predictions": True, "alert_threshold": 0.5}
    )
    monitor.log_metrics({"accuracy": 0.85})
    monitor.log_prediction(input_data=None, prediction=1)

    saved = monitor.save_logs(str(tmp_path))
    assert Path(saved["metrics"]).exists()
    assert Path(saved["predictions"]).exists()

    new_monitor = ModelMonitor(
        {"track_metrics": True, "log_predictions": True, "alert_threshold": 0.5}
    )
    new_monitor.load_logs(str(tmp_path))
    assert len(new_monitor.metrics_history) == 1
    assert len(new_monitor.predictions_log) == 1


def test_generate_report_contains_sections(tmp_path):
    monitor = ModelMonitor(
        {"track_metrics": True, "log_predictions": True, "alert_threshold": 0.5}
    )
    monitor.log_metrics({"accuracy": 0.8, "f1_score": 0.75})
    monitor.log_prediction(input_data=None, prediction=1)
    monitor.save_logs(str(tmp_path))

    report = monitor.generate_report()
    assert "MODEL MONITORING REPORT" in report
    assert "METRICS SUMMARY" in report
    assert "PREDICTIONS SUMMARY" in report


def test_steps_configuration_allows_custom_pipeline():
    monitor = ModelMonitor(
        {
            "steps": [
                "metrics_logger",
                {"type": "metric_threshold", "params": {"default_threshold": 0.7}},
            ]
        }
    )

    monitor.log_metrics({"accuracy": 0.75})
    monitor.log_prediction(input_data=None, prediction=1)

    assert len(monitor.metrics_history) == 1
    assert monitor.predictions_log == []
    assert monitor.get_predictions_summary()["message"] == "Prediction logging disabled"


def test_custom_observability_step_registration():
    class CountingStep(ObservabilityStep):
        name = "counting_step"

        def __init__(self):
            super().__init__()
            self.metrics_calls = 0
            self.prediction_calls = 0

        def on_log_metrics(self, metrics, model_version):
            self.metrics_calls += 1

        def on_log_prediction(
            self, input_data, prediction, model_version, metadata=None
        ):
            self.prediction_calls += 1

    original_registry = ModelMonitor.STEP_REGISTRY.copy()
    try:
        ModelMonitor.register_step(CountingStep)
        monitor = ModelMonitor(
            {
                "steps": [
                    "metrics_logger",
                    "predictions_logger",
                    "counting_step",
                ]
            }
        )

        monitor.log_metrics({"accuracy": 0.9})
        monitor.log_prediction(input_data=None, prediction=1)

        counting_step = monitor.get_step("counting_step")
        assert counting_step.metrics_calls == 1
        assert counting_step.prediction_calls == 1
    finally:
        ModelMonitor.STEP_REGISTRY = original_registry
