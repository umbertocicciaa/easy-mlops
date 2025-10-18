# Make MLOps Easy

Make MLOps Easy is an opinionated framework that automates the repetitive pieces of an end‑to‑end machine learning operations workflow. It wraps data preparation, model training, deployment, and observability behind a consistent API and a distributed CLI so teams can focus on delivering models instead of plumbing.

## Why Make MLOps Easy

- **Pipeline in a box** – preprocess tabular data, train a model, ship artifacts, and generate observability reports with a single command or Python call.
- **Distributed runtime** – offload long running jobs to workers coordinated by a FastAPI master service while driving everything from the `make-mlops-easy` CLI.
- **Composable steps** – extend preprocessing, deployment, and observability through registries, or swap the training backend for neural networks, bespoke callables, or vendor integrations.
- **Reproducible deployments** – every run produces a versioned deployment directory containing the model, preprocessor, metadata, and monitoring logs.
- **Friendly developer experience** – MkDocs documentation, Makefile shortcuts, and an examples catalogue make local iteration straightforward.

## High-Level Architecture

The project consists of two cooperating layers:

1. **Core pipeline (`easy_mlops.pipeline.MLOpsPipeline`)** – orchestrates configuration, preprocessing, training, deployment, and observability. It can be embedded directly in Python code or invoked by the worker runtime.
2. **Distributed runtime (`easy_mlops.distributed`)** – a FastAPI master keeps workflow state, hands tasks to workers, and streams results back to the CLI. Workers execute tasks via `TaskRunner`, which simply calls into the pipeline.

```
┌──────────────┐     submit workflow     ┌──────────────┐
│   CLI user   │ ───────────────────────▶│   Master API │
└─────┬────────┘                         └────┬─────────┘
      │   poll status / logs                  │ assign task
      │                                       ▼
      │                               ┌──────────────┐
      └──────────────────────────────▶│   Worker(s)  │
                                      └────┬─────────┘
                                           │ runs
                                           ▼
                                   ┌──────────────┐
                                   │ MLOpsPipeline│
                                   └──────────────┘
```

See the [docs](https://umbertocicciaa.github.io/easy-mlops/) for deeper dives into each subsystem.

## Quick Start

### 1. Install

```bash
git clone https://github.com/umbertocicciaa/easy-mlops.git
cd easy-mlops
pip install -e .
```

Or install the published package:

```bash
pip install make-mlops-easy
```

### 2. Launch the distributed runtime

The CLI submits work to a master service. For local runs you can either start the components manually or use the helper script that ships with the examples.

**Manual startup**

```bash
# Terminal 1 – master (FastAPI + uvicorn)
make-mlops-easy master start

# Terminal 2 – worker agent
make-mlops-easy worker start
```

**Helper script**

```bash
./examples/distributed_runtime.sh up   # starts master + worker in the background
```

The master listens on `http://127.0.0.1:8000` by default; override with `--host/--port` or `EASY_MLOPS_MASTER_URL`.

### 3. Run the pipeline

Train, inspect, predict, and observe using the CLI:

```bash
make-mlops-easy train examples/sample_data.csv --target approved
make-mlops-easy status models/deployment_20240101_120000
make-mlops-easy predict examples/sample_data.csv models/deployment_20240101_120000 --output predictions.json
make-mlops-easy observe models/deployment_20240101_120000
```

All commands accept `--config` to point at a YAML configuration (see `examples/pipeline/configs/**` for templates). Use `./examples/distributed_runtime.sh down` to stop the local runtime.

### 4. Programmatic usage

Embed the pipeline directly in Python when you do not need the distributed runtime:

```python
from easy_mlops.pipeline import MLOpsPipeline

pipeline = MLOpsPipeline(config_path="configs/quickstart.yaml")
results = pipeline.run("data/train.csv", target_column="price")

predictions = pipeline.predict("data/new_rows.csv", results["deployment"]["deployment_dir"])
status = pipeline.get_status(results["deployment"]["deployment_dir"])
report = pipeline.observe(results["deployment"]["deployment_dir"])
```

## CLI at a Glance

| Command | Purpose | Key options |
| ------- | ------- | ----------- |
| `train` | Submit a training workflow; optionally skip deployment with `--no-deploy`. | `--target`, `--config`, `--master-url`, `--poll-interval` |
| `predict` | Score a dataset with a previously deployed model. | `--output`, `--config`, `--master-url` |
| `status` | Retrieve deployment metadata and observability summaries. | `--config`, `--master-url` |
| `observe` | Produce a detailed monitoring report. | `--output`, `--config`, `--master-url` |
| `master start` | Run the FastAPI orchestration service. | `--host`, `--port`, `--state-path` |
| `worker start` | Launch a worker agent that executes queued tasks. | `--worker-id`, `--capability`, `--poll-interval`, `--master-url` |
| `init` | Generate a default `mlops-config.yaml`. | `--output` |

The Makefile exposes the same workflows via `make train`, `make predict`, `make status`, and `make observe` for convenience once the virtual environment is set up (`make install-dev`).

## Configuration System

All behaviour is driven by YAML configuration. Each top-level section corresponds to a subsystem:

```yaml
preprocessing:
  steps:
    - type: missing_values
      params: {strategy: median}
    - categorical_encoder
    - feature_scaler

training:
  backend: sklearn         # or neural_network, callable, …
  model_type: auto
  test_size: 0.2
  cv_folds: 5

deployment:
  output_dir: ./models
  create_endpoint: true
  endpoint_filename: predict.py

observability:
  track_metrics: true
  log_predictions: true
  metric_thresholds: {accuracy: 0.85}
```

- **Preprocessing** – compose `PreprocessingStep` instances to clean data. Multiple formats (CSV, JSON, Parquet) are supported out of the box.
- **Training** – select a backend (scikit-learn random forests, neural networks, or your own callables) and configure evaluation settings.
- **Deployment** – control where artifacts land, which files are emitted, and whether auxiliary scripts are generated.
- **Observability** – toggle metric/prediction logging and configure alert thresholds or custom monitoring steps.

Use `make-mlops-easy init` to scaffold a baseline file, then extend it or import presets from `examples/pipeline/configs/`.

## Extending the Framework

- **Custom preprocessing** – subclass `PreprocessingStep`, register it with `DataPreprocessor.register_step`, and reference it by name in configuration.
- **Alternative training backends** – subclass `BaseTrainingBackend`, implement `train`, and register it with `ModelTrainer.register_backend`.
- **Deployment hooks** – create new `DeploymentStep` implementations (for example, uploading artifacts to cloud storage) and add them to the `deployment.steps` list.
- **Monitoring integrations** – extend `ObservabilityStep` to forward metrics/predictions to external systems or implement bespoke alert logic.

Because all registries are global, custom components become available to both the pipeline and the CLI once imported.

## Repository Layout

```
easy-mlops/
├── easy_mlops/
│   ├── cli.py                    # Click-based CLI targeting the distributed runtime
│   ├── pipeline.py               # High-level orchestration
│   ├── config/                   # YAML loader and defaults
│   ├── preprocessing/            # Step framework + built-in transformers
│   ├── training/                 # Trainer abstraction and backends
│   ├── deployment/               # Deployment steps and artifact writers
│   ├── observability/            # Monitoring steps and log management
│   └── distributed/              # FastAPI master, worker agent, task runner, state store
├── docs/                         # MkDocs sources (https://umbertocicciaa.github.io/easy-mlops/)
├── examples/                     # Datasets, scripts, and curated configuration files
├── tests/                        # Pytest suite covering core subsystems
├── mkdocs.yml                    # Documentation site configuration
└── Makefile                      # Common development shortcuts
```

## Development

```bash
make install-dev   # create .venv/ and install package with dev extras
make format lint   # black + flake8
make test          # run pytest suite
make coverage      # generate trace-based coverage summary
make docs-serve    # live MkDocs preview at http://127.0.0.1:8000/
```

Continuous integration (GitHub Actions) runs tests on Linux, macOS, and Windows, ensures formatting, builds documentation, and publishes packages/images on tagged releases. See `docs/cicd.md` for details.

## Additional Resources

- Documentation: https://umbertocicciaa.github.io/easy-mlops/
- Examples walkthroughs: `examples/README.md`
- Issues & roadmap: https://github.com/umbertocicciaa/easy-mlops/issues

## License

Make MLOps Easy is released under the MIT License. See [LICENSE](LICENSE).

## Author

Created by Umberto Domenico Ciccia.
