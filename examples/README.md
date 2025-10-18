# Make MLOps Easy Examples

This directory collects datasets, runnable scripts, and configuration presets
that exercise every stage of the Make MLOps Easy pipeline. Use them as
blueprints when adapting the framework to your own projects.

## Datasets

- `sample_data.csv` – binary loan approval dataset used by quickstart workflows
  (`approved` is the target column).
- `data/house_prices.csv` – small regression dataset for exploring alternative
  trainers (`price` is the target column).

All files are lightweight and safe to version control.

## Distributed Runtime Helpers

The CLI now communicates with a master service that orchestrates worker agents.
All example scripts source `examples/distributed_runtime.sh`, which will:

- Start a local master at `http://127.0.0.1:8000` if none is running.
- Launch a worker connected to the master.
- Store process logs under `examples/.runtime/`.
- Tear both processes down when the script exits.

Set `EXAMPLES_USE_EMBEDDED_MASTER=0` or `EXAMPLES_USE_EMBEDDED_WORKER=0` if you
prefer to manage the runtime yourself (for example, when pointing at a remote
cluster). Override `MASTER_URL` to target a different endpoint.

## CLI Workflows

- `complete_workflow.sh` – orchestrates the full CLI journey: train, status,
  predict, observe. It now relies on the curated configuration under
  `pipeline/configs/quickstart.yaml`.
- `pipeline/scripts/*.sh` – granular helpers covering each command
  (`init`, `train`, `predict`, `status`, `observe`) plus regression and neural
  network variants. They automatically resolve repository-relative paths and
  always target the latest deployment.

Make the scripts executable once (`chmod +x examples/pipeline/scripts/*.sh`) and
run them from the repository root.

## Configuration Library

The `pipeline/configs/` folder ships with ready-to-use YAML files:

- `quickstart.yaml` – baseline classification setup with endpoint generation.
- `regression_neural_network.yaml` – demonstrates the neural-network backend on
  a regression problem.
- `observability_strict.yaml` – rewires observability steps and metric
  thresholds for stricter alerting.
- `mlops-config.yaml` – generated via `scripts/00_init_project.sh` if you want a
  clean copy of the default settings.

Reference these presets directly via `make-mlops-easy train ... -c <path>` (add
`--master-url` if you manage the runtime manually) or use them as a starting
point for your own configuration files.

## Programmatic Pipeline Usage

`pipeline/scripts/06_run_programmatic_pipeline.py` shows how to call
`easy_mlops.pipeline.MLOpsPipeline` from Python. This is handy for notebooks or
automation where invoking the CLI isn’t convenient.

Run it with:

```bash
python examples/pipeline/scripts/06_run_programmatic_pipeline.py
```

## Creating Your Own Dataset

Make MLOps Easy supports CSV, JSON, and Parquet inputs. Ensure:

1. Your dataset has a clearly named target column.
2. Feature columns contain clean numerical/categorical values (missing values
   are handled according to your configuration).
3. The dataset lives in a supported format and is reachable by the CLI command.

Need ideas? Copy one of the sample CSVs and adjust the columns to match your
domain—everything else in the examples will continue to work.
