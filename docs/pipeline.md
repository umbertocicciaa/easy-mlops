# Pipeline Architecture

`easy_mlops.pipeline.MLOpsPipeline` is the backbone of Make MLOps Easy. It strings together configuration loading, data preprocessing, model training, deployment, and observability into a deterministic workflow. The pipeline can be executed directly from Python or indirectly through the distributed runtime (CLI → master → worker → pipeline).

```
config.yaml ─┐
             ▼
        ┌───────────────┐      ┌──────────────┐
 data ─▶│ DataPreprocessor│ ──▶ │ ModelTrainer │
        └───────────────┘      └──────┬───────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │ ModelDeployer  │
                              └──────┬─────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │ ModelMonitor   │
                              └────────────────┘
```

Each component is configurable, testable in isolation, and extensible through registries.

## Execution flow

1. **Configuration** – `Config` merges defaults with an optional YAML file. The resulting dictionary is passed to every subsystem so defaults stay centralised.
2. **Preprocessing** – `DataPreprocessor` loads the dataset, applies configured `PreprocessingStep` instances, records feature metadata, and returns `(X, y)`. During inference it reuses the fitted state and reorders columns to match training time.
3. **Training** – `ModelTrainer` delegates to a registered `BaseTrainingBackend`. The default scikit-learn backend detects the problem type, builds an estimator, runs train/test splits, computes metrics, and (optionally) cross-validates. The backend yields a `TrainingRunResult` containing the fitted model and metadata.
4. **Deployment** – `ModelDeployer` assembles `DeploymentStep` instances (create directory, save model, persist preprocessor, write metadata, optional endpoint script). Steps share a mutable `DeploymentContext`, making the pipeline easy to extend.
5. **Observability** – `ModelMonitor` triggers `ObservabilityStep` hooks to log metrics and predictions, manage thresholds, and generate summaries. Logs are written to JSON files under the deployment directory.

The pipeline returns a dictionary with `preprocessing`, `training`, `deployment` (if enabled), and `logs` metadata so callers can chain additional automation.

## Distributed runtime integration

Workers inside `easy_mlops.distributed.worker` call `MLOpsPipeline` through `TaskRunner`:

- `train` tasks instantiate the pipeline, call `run`, and stream the printed progress back to the master.
- `predict`, `status`, and `observe` tasks hydrate the same pipeline to load artifacts from a deployment directory.

This design keeps the CLI, distributed runtime, and in-process usage aligned—one implementation powers every entry point.

## Configuration surface

Every stage is configured via the YAML file consumed by `Config`. The top-level keys map directly to pipeline components:

- `preprocessing` – toggles legacy options (`handle_missing`, `encode_categorical`, `scale_features`) or declares an explicit `steps` list. Steps refer to names registered in `DataPreprocessor.STEP_REGISTRY`.
- `training` – defines the backend (`sklearn`, `neural_network`, `callable`, `deep_learning`, `nlp`) and its parameters (`model_type`, `test_size`, `cv_folds`, `random_state`, etc.). Custom backends read arbitrary keys from this section.
- `deployment` – controls artifact destinations, filenames, and optional endpoint generation. Advanced users can supply a custom `steps` list to inject additional deployment logic (for example, uploading to S3).
- `observability` – toggles metric/prediction logging and configures thresholds. Like the other components it supports an explicit `steps` list to fine-tune monitoring.

`make-mlops-easy init` builds a skeleton file that mirrors the defaults in `easy_mlops/config/config.py`.

## Component deep dive

- **DataPreprocessor** (`easy_mlops/preprocessing/preprocessor.py`) wraps step classes defined in `easy_mlops/preprocessing/steps.py`. Steps follow a minimal contract (`fit`, `transform`, optional `save/load` hooks). Custom steps can be registered globally via `DataPreprocessor.register_step`.
- **ModelTrainer** (`easy_mlops/training/trainer.py`) wraps the backend registry in `easy_mlops/training/backends.py`. The built-in scikit-learn backend covers random forests, logistic/linear regression, XGBoost, and MLPs. Callable-style backends let you plug in frameworks such as PyTorch or TensorFlow.
- **ModelDeployer** (`easy_mlops/deployment/deployer.py`) executes a list of `DeploymentStep` instances. The default stack creates a timestamped directory (`deployment_YYYYMMDD_HHMMSS`), serialises artifacts with `joblib`, writes `metadata.json`, and optionally generates an executable prediction helper.
- **ModelMonitor** (`easy_mlops/observability/monitor.py`) manages `ObservabilityStep` implementations such as `MetricsLoggerStep`, `PredictionsLoggerStep`, and `MetricThresholdStep`. Steps persist their own state and can expose summaries for CLI rendering.

Each subsystem exposes registration helpers so you can inject new behaviour without forking the project.

## Testing the pipeline

`tests/test_pipeline.py` exercises the end-to-end flow with sample data. Additional unit tests cover the preprocessor, trainer backends, deployment steps, observability pipeline, and distributed state store. Use these tests as a blueprint when contributing new components.

## Extension ideas

- Add a LightGBM training backend that consumes hyperparameters from `training:`.
- Register a preprocessing step that enriches features from an external service.
- Extend deployment with a step that pushes artifacts to an object store and records the URI in `DeploymentContext.artifacts`.
- Register an observability step that forwards metrics to Prometheus or Slack.

See [development guidelines](development.md) for tips on structure, testing, and documentation updates when adding new pieces.
