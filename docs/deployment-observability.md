# Deployment & Observability

This section dives deeper into artifact management, deployment layouts, and the observability tooling bundled with Make MLOps Easy.

## Deployment Layout

Running `make-mlops-easy train` (without `--no-deploy`) creates a timestamped directory under `models/`:

```
models/
└── deployment_20240101_120000/
    ├── metadata.json
    ├── model.joblib
    ├── preprocessor.joblib
    ├── predict.py          # optional endpoint script
    └── logs/
        ├── metrics_history.json
        └── predictions_log.json
```

- `model.joblib` – Serialized estimator plus metadata such as metrics and problem type.
- `preprocessor.joblib` – Serialized `DataPreprocessor` instance capturing fitted scalers/encoders.
- `metadata.json` – Deployment metadata (paths, training summary, version, timestamps).
- `predict.py` – Lightweight CLI for inference (created when `create_endpoint` is enabled).
- `logs/` – Persisted metrics and prediction entries managed by `ModelMonitor`.

## Loading Artifacts

The CLI `predict`, `status`, and `observe` commands rehydrate the model and preprocessor using `ModelDeployer.load_deployed_model`. You can follow the same pattern in custom scripts:

```python
from easy_mlops.deployment import ModelDeployer

deployer = ModelDeployer({"output_dir": "./models"})
model_data, preprocessor, metadata = deployer.load_deployed_model("models/deployment_20240101_120000")

df = preprocessor.load_data("data/new_samples.csv")
X, _ = preprocessor.prepare_data(df, target_column=None, fit=False)
predictions = model_data["model"].predict(X)
```

## Observability Features

`ModelMonitor` captures two streams:

1. **Metrics history** – Recorded after each training run. Summaries report:
   - Latest metrics and timestamps.
   - Number of logged entries.
   - Trend statistics (mean, std, min, max) when multiple logs exist.
2. **Prediction log** – Captures prediction outputs, timestamps, and optional metadata.

### Threshold Checks

Use `check_metric_threshold(metric_name, value)` to determine if alerting is required based on configuration:

```python
monitor = ModelMonitor({"alert_threshold": 0.8, "track_metrics": True, "log_predictions": True})
should_alert = monitor.check_metric_threshold("accuracy", value=0.72)
if should_alert:
    # Trigger your notification mechanism
```

### Reports

`make-mlops-easy observe` renders a text report summarizing both logs. You can generate the same report programmatically:

```python
report = monitor.generate_report()
print(report)
```

## Log Rotation & Storage

Logs are JSON files suited for downstream ingestion. Consider copying them to durable storage or feeding them into monitoring dashboards. If you integrate with external observability stacks, replace the `save_logs` method or extend `ModelMonitor` to emit events to your preferred sink.

## Automation Tips

- Use the `status` command in scheduled jobs to verify metric health and deployment metadata.
- Combine `predict` with automated data freshness checks; the log files provide audit trails.
- Retain the deployment directory structure for reproducibility: each run bundles the model, preprocessing, and configuration in one folder.
