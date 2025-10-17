# Pipeline Examples

This folder provides end-to-end examples that exercise every CLI command and the
programmatic API exposed by `easy_mlops.pipeline.MLOpsPipeline`. Each example is
paired with a ready-to-use YAML configuration file so you can copy, tweak, and
run scenarios that span the full MLOps lifecycle: data preprocessing, model
training, deployment, prediction, and observability.

## Layout

- `configs/` – curated configuration files that demonstrate different pipeline
  behaviours (quickstart, regression with neural networks, strict monitoring).
- `scripts/` – shell and Python helpers that invoke the CLI or pipeline API for
  each stage of the workflow.

All scripts assume you run them from the repository root with `make-mlops-easy`
installed in the current environment. They default to the datasets under
`examples/`.

## Scenario Overview

| Script | What it demonstrates | Uses configuration |
| --- | --- | --- |
| `00_init_project.sh` | Generate a baseline `mlops-config.yaml` file | Built-in defaults |
| `01_train_quickstart.sh` | Train & deploy a classification model with endpoint generation | `configs/quickstart.yaml` |
| `02_train_regression_nn.sh` | Train a regression model with the neural-network backend | `configs/regression_neural_network.yaml` |
| `03_predict_latest.sh` | Batch predictions against the most recent deployment | `configs/quickstart.yaml` (optional) |
| `04_status_latest.sh` | Inspect deployment metadata and logged metrics | `configs/quickstart.yaml` (optional) |
| `05_observe_latest.sh` | Produce a monitoring report with stricter alert thresholds | `configs/observability_strict.yaml` |
| `06_run_programmatic_pipeline.py` | Invoke the pipeline from Python for custom automation | `configs/regression_neural_network.yaml` |

### Suggested Execution Order

1. `scripts/00_init_project.sh` *(optional)* – generate a starter config you can
   extend.
2. `scripts/01_train_quickstart.sh` – trains & deploys a classification model
   using `examples/sample_data.csv`.
3. `scripts/03_predict_latest.sh` / `scripts/04_status_latest.sh` –
   consume the freshly deployed model.
4. `scripts/05_observe_latest.sh` – view the observability report.
5. `scripts/02_train_regression_nn.sh` – experiment with a regression-specific
   configuration.
6. `scripts/06_run_programmatic_pipeline.py` – explore programmatic orchestration.

The scripts automatically locate the latest deployment under `models/` so you
can mix and match runs without manual bookkeeping.

## Configuration Highlights

- **`quickstart.yaml`** – baseline classification settings with endpoint
  creation enabled and sensible preprocessing defaults.
- **`regression_neural_network.yaml`** – custom preprocessing steps combined
  with the `neural_network` backend to illustrate alternative trainers.
- **`observability_strict.yaml`** – stricter metric thresholds and explicit
  direction hints that surface alerting capabilities.

Each configuration file is heavily commented to explain the knobs you can turn.
Use them as a reference when authoring your own project-specific settings.
