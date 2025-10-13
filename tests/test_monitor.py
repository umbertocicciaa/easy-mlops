from pathlib import Path

import numpy as np

from easy_mlops.observability import ModelMonitor


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
