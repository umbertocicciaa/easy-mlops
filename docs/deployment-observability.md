# Deployment & Observability

This guide explains how Make MLOps Easy materialises a deployment, how to shape the deployment behaviour through configuration, and how the built-in observability stack records what happens after a model goes live.

## Deployment Pipeline (High-Level)

`make-mlops-easy train` and `MLOpsPipeline.run(..., deploy=True)` construct a `ModelDeployer` (see `easy_mlops/deployment/deployer.py:18`). The deployer pulls options from the `deployment` section of your configuration, builds a `DeploymentContext` with the trained estimator, the fitted preprocessor, run metadata, and the resolved output directory (`easy_mlops/deployment/steps.py:15`), then executes a sequence of deployment steps. Steps are sourced from `ModelDeployer.STEP_REGISTRY` and run in order, each enriching the context with new artifacts.

### Step Catalogue

- **Context bootstrap**. `DeploymentContext` tracks the shared state: training results, absolute paths, generated metadata, and a central artifact dictionary. Every step can read or write to this context, making downstream steps deterministic.
- **`create_directory` (`CreateDeploymentDirectoryStep`)**. Creates a timestamped directory underneath `output_dir`. The default prefix is `deployment`, but any prefix can be supplied. If multiple deployments land within the same second, the step appends an incrementing suffix to keep paths unique (`easy_mlops/deployment/steps.py:62`).
- **`save_model` (`SaveModelStep`)**. Delegates to the trainer backend by calling `context.model.save_model(<path>)`, ensuring the persisted artifact includes backend-specific metadata (`easy_mlops/deployment/steps.py:83`).
- **`save_preprocessor` (`SavePreprocessorStep`)**. Serialises the fitted `DataPreprocessor` with `joblib.dump`, guaranteeing that later inference reproduces the same transformations (`easy_mlops/deployment/steps.py:99`).
- **`save_metadata` (`SaveMetadataStep`)**. Calls `ModelDeployer.create_model_metadata` to write `metadata.json` describing the run: absolute artifact paths, training metrics, configured version tag, and the deployment timestamp (`easy_mlops/deployment/deployer.py:43`).
- **`endpoint_script` (`EndpointScriptStep`, optional)**. When enabled, writes an executable helper that loads the persisted model and preprocessor to score new data. You can provide a custom script template or filename (`easy_mlops/deployment/steps.py:177`).

The pipeline short-circuits if a prerequisite is missing (for example, trying to save a model before the directory exists raises a `RuntimeError`), which keeps deployments predictable and debuggable.

### Interaction with the CLI and Pipeline

- `make-mlops-easy train` performs preprocessing, training, deployment, and log persistence in a single command. Use `--config path/to/config.yaml` to apply custom deployment settings.
- Pass `--no-deploy` to skip the deployment stage entirely; the training results are still returned, but no deployment directory is created.
- Call `ModelDeployer.deploy(model, preprocessor, training_results, create_endpoint=...)` directly when orchestrating bespoke workflows. Overriding `create_endpoint` at call time allows you to toggle endpoint generation without editing configuration files.

## Deployment Configuration Recipes

The deployer reads options from the `deployment` node of the YAML configuration. Keys that are omitted fall back to defaults declared in `easy_mlops/config/config.py:20` and inside `ModelDeployer._initialize_steps`. The following scenarios cover the supported deployment modes.

### Default timestamped deployments

With no custom configuration, deployments land in `./models/deployment_<timestamp>/` and contain `model.joblib`, `preprocessor.joblib`, `metadata.json`, and `logs/` when observability persists metrics. This mode is ideal for quick experiments and CI pipelines.

```yaml
deployment: {}
```

- `output_dir` defaults to `./models`.
- `deployment_prefix` defaults to `deployment`.
- `create_endpoint` is disabled unless explicitly requested.

### Custom output location and naming

Change where artifacts live and how directories are named by setting `output_dir`, `deployment_prefix`, and optionally overriding filenames.

```yaml
deployment:
  output_dir: ./artifacts/churn
  deployment_prefix: churn_model
  metadata_filename: manifest.json
```

- The directory becomes `./artifacts/churn/churn_model_<timestamp>/`.
- `manifest.json` replaces the default metadata filename (`easy_mlops/deployment/deployer.py:162`).
- All other steps continue to run with their defaults unless specified.

### Bundling an endpoint helper script

Enable the built-in `predict.py` template or supply your own script. The script is created during deployment and marked executable.

```yaml
deployment:
  create_endpoint: true
  endpoint_filename: predict_sales.py
  endpoint_template: |
    #!/usr/bin/env python3
    import json
    from pathlib import Path
    import joblib

    def predict(path):
        base = Path(__file__).parent
        model = joblib.load(base / "model.joblib")["model"]
        data = json.loads(Path(path).read_text())
        return model.predict(data)

    if __name__ == "__main__":
        import sys
        print(predict(sys.argv[1]))
```

- `create_endpoint: true` injects `EndpointScriptStep` during `_initialize_steps` (`easy_mlops/deployment/deployer.py:168`).
- `endpoint_filename` and `endpoint_template` override the defaults. If you omit the template, the bundled `DEFAULT_ENDPOINT_TEMPLATE` is used (`easy_mlops/deployment/steps.py:150`).
- You can still toggle script creation per run by calling `ModelDeployer.deploy(..., create_endpoint=False)` when needed.

### Explicit step lists

Provide an ordered list of step specifications to gain full control over the deployment pipeline. Strings refer to registered step names, while dictionaries let you override parameters.

```yaml
deployment:
  steps:
    - create_directory
    - save_model
    - save_preprocessor
    - type: save_metadata
      params:
        filename: detailed_metadata.json
    - type: endpoint_script
      params:
        enabled: true
        filename: predict.sh
```

- When `steps` is supplied, the default step sequence is ignored, so include every action you require.
- Any step not in `ModelDeployer.STEP_REGISTRY` raises a helpful `ValueError` (`easy_mlops/deployment/deployer.py:182`).
- Use this pattern to skip steps (for example, omit `save_preprocessor` if your backend embeds preprocessing) or to insert custom ones.

### Registering custom deployment steps

Extend the framework by subclassing `DeploymentStep`, registering it, then referencing it from the `steps` list.

```python
from easy_mlops.deployment import ModelDeployer, DeploymentStep, DeploymentContext

class UploadToS3Step(DeploymentStep):
    name = "upload_to_s3"

    def run(self, context: DeploymentContext) -> None:
        artifact_dir = context.deployment_dir
        uri = push_directory_to_s3(artifact_dir)
        context.artifacts["s3_uri"] = uri

ModelDeployer.register_step(UploadToS3Step)
```

```yaml
deployment:
  steps:
    - create_directory
    - save_model
    - save_preprocessor
    - save_metadata
    - upload_to_s3
```

- `ModelDeployer.register_step` guards against invalid subclasses and makes the new step available globally (`easy_mlops/deployment/deployer.py:34`).
- Custom steps can write to the shared artifact map so later automation (for example, CI notifications) can read new locations or identifiers.

### Controlling metadata content

The metadata writer picks up optional keys from the configuration:

```yaml
deployment:
  version: 2.1.0
  deployment_time: "2024-05-01T09:00:00Z"
```

- `version` becomes part of `metadata.json` and is surfaced by `ModelDeployer.load_deployed_model` (`easy_mlops/deployment/deployer.py:45`).
- Supplying `deployment_time` is useful when replaying historical runs. If omitted, the current time is recorded with ISO format.
- You can also override `output_dir` at call time via `ModelDeployer.save_deployment_artifacts(..., output_dir="./staging")` when the destination depends on runtime context.

## Deployment Layout

Running `make-mlops-easy train` (without `--no-deploy`) creates a timestamped directory under `models/` by default:

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

- `model.joblib` - Serialized estimator plus metadata such as metrics and problem type.
- `preprocessor.joblib` - Serialized `DataPreprocessor` instance capturing fitted scalers and encoders.
- `metadata.json` - Deployment metadata (paths, training summary, version, timestamps).
- `predict.py` - Lightweight CLI for inference (created when `create_endpoint` is enabled).
- `logs/` - Persisted metrics and prediction entries managed by `ModelMonitor` or any custom observability steps.

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

## Observability at a Glance

- `ModelMonitor` orchestrates a pipeline of `ObservabilityStep` instances (see `easy_mlops/observability/steps.py`). Each step reacts to `log_metrics` and `log_prediction` events emitted during training, inference, and CLI usage.
- When you call `make-mlops-easy train`, the pipeline logs evaluation metrics immediately after training completes. Running `predict`, `status`, or `observe` hydrates a fresh monitor using the same configuration and replays saved logs.
- Every observability step persists its own state underneath the deployment directory (`logs/`), keeping metrics, predictions, and alert evaluations colocated with the model artifacts.
- The CLI surfaces monitoring insights through `status` (metrics and predictions summaries) and `observe` (formatted report). You can generate the same output in code via `ModelMonitor.generate_report()`.
- The observability subsystem is extensible: register additional steps with `ModelMonitor.register_step` to stream events to third-party platforms or apply custom heuristics.

## Observability Capabilities and Configuration

The observability section of `config.yaml` determines which steps are active and how they behave. Defaults are:

```yaml
observability:
  track_metrics: true
  log_predictions: true
  alert_threshold: 0.8
```

### Metrics Logger (`metrics_logger`)

- Collects every metrics payload the pipeline emits, enriching it with timestamps and model versions.
- Produces trend statistics (mean, standard deviation, minimum, maximum) when multiple logs exist.
- Persists to `logs/metrics_history.json`. Disable by setting `track_metrics: false`.

### Predictions Logger (`predictions_logger`)

- Records individual predictions with timestamps and optional metadata. Non scalar predictions are converted to strings for safe JSON storage.
- Summaries report totals, first and last timestamps, and the last five entries for quick inspection.
- Persists to `logs/predictions_log.json`. Disable with `log_predictions: false`.

### Metric Threshold Evaluator (`metric_threshold`)

- Supplies alert decisions via `check_metric_threshold(metric_name, value)`. By default it expects higher values for metrics such as accuracy or F1 and lower values for error metrics such as RMSE.
- Configuration keys:
  - `alert_threshold`: Default threshold when no per metric value is provided.
  - `metric_thresholds`: Mapping of metric name to custom threshold.
  - `metric_directions`: Optional mapping to override direction (`higher` or `lower`) for custom metrics.
- When the threshold step is disabled (for example, by redefining the step list), `ModelMonitor` falls back to the basic `alert_threshold` logic for compatibility.

### Step Orchestration with `steps`

For full control, provide an explicit ordered list under `observability.steps`:

```yaml
observability:
  steps:
    - metrics_logger                    # simple string uses default params
    - type: metric_threshold            # dict allows parameter overrides
      params:
        default_threshold: 0.75
        metric_thresholds:
          accuracy: 0.8
        metric_directions:
          rmse: higher                  # invert direction if needed
```

- Strings refer to registered step names (`metrics_logger`, `predictions_logger`, `metric_threshold`).
- Dictionaries must include a `type` key and can supply nested `params`.
- Omitting `predictions_logger` removes prediction capture entirely; downstream summaries will indicate that prediction logging is disabled.

### Custom Steps and Integrations

Add instrumentation by subclassing `ObservabilityStep` and registering it:

```python
from easy_mlops.observability import ModelMonitor, ObservabilityStep

class SlackAlertStep(ObservabilityStep):
    name = "slack_alert"

    def on_log_metrics(self, metrics, model_version):
        if metrics.get("accuracy", 1) < 0.7:
            send_to_slack(metrics, model_version)

ModelMonitor.register_step(SlackAlertStep)
```

Once registered, list the step inside `observability.steps`. Each custom step can persist its own files through `save()` or restore state with `load()`.

### Complete Configuration Example

```yaml
observability:
  track_metrics: true
  log_predictions: true
  alert_threshold: 0.82              # used when no per metric override exists
  metric_thresholds:
    f1_score: 0.78
    rmse: 0.45
  metric_directions:
    rmse: lower                      # explicit direction for custom metrics
  steps:
    - metrics_logger
    - predictions_logger
    - type: metric_threshold
      params:
        default_threshold: 0.82
        metric_thresholds:
          precision: 0.75
```

Any keys you omit fall back to the defaults baked into `easy_mlops/config/config.py`.

## Observability in Practice

### Threshold Checks

Use `check_metric_threshold(metric_name, value)` to determine if alerting is required based on configuration:

```python
monitor = ModelMonitor(
    {
        "track_metrics": True,
        "log_predictions": True,
        "alert_threshold": 0.8,
        "metric_thresholds": {"accuracy": 0.85},
    }
)
should_alert = monitor.check_metric_threshold("accuracy", value=0.82)
if should_alert:
    trigger_notification()
```

The evaluator respects per metric thresholds and direction overrides, falling back to the global `alert_threshold` when no specific rule is defined.

### Reports

`make-mlops-easy observe` renders a text report summarizing both logs. You can generate the same report programmatically:

```python
report = monitor.generate_report()
print(report)
```

The report includes total entries, first and last timestamps, and the latest metrics snapshot to support runbooks or dashboards.

## Log Management and Automation

- Logs are JSON files suited for downstream ingestion. Copy them to durable storage or enhance `save_logs` to forward directly into your monitoring stack.
- Schedule the `status` command to verify metric health and deployment metadata on a cadence.
- Combine `predict` with automated data freshness checks; prediction logs provide an auditable trail.
- Preserve each deployment directory for reproducibility - every run bundles the model, preprocessing artifacts, configuration snapshot, and observability logs together.
