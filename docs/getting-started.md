# Quick Start

This guide helps you install Make MLOps Easy, launch the distributed runtime, and run the full workflow with the CLI or the Python API.

## Prerequisites

- Python 3.10 or newer
- `git`
- (Optional) Docker if you plan to run inside a container
- macOS users need Xcode Command Line Tools when compiling `xgboost`

## Install the project

Clone the repository and install it in editable mode with development conveniences:

```bash
git clone https://github.com/umbertocicciaa/easy-mlops.git
cd easy-mlops
pip install -e .[dev]
```

If you only need the library/CLI, install the published package:

```bash
pip install make-mlops-easy
```

Docker users can build and run the image locally:

```bash
docker build -t make-mlops-easy .
docker run --rm make-mlops-easy --help
```

## Launch the distributed runtime

The CLI delegates work to a FastAPI master that hands tasks to worker agents. Start them manually or rely on the example helper script.

### Manual startup

```bash
# Terminal 1 – master
make-mlops-easy master start --host 127.0.0.1 --port 8000

# Terminal 2 – worker
make-mlops-easy worker start --master-url http://127.0.0.1:8000
```

### Helper script

```bash
./examples/distributed_runtime.sh up
```

The script spins up a master and a worker, storing logs under `examples/.runtime/`. Stop the processes with `./examples/distributed_runtime.sh down`.

Set `EASY_MLOPS_MASTER_URL` or pass `--master-url` to point the CLI at a remote master.

## Train, predict, and observe

Use the CLI to drive the full workflow. The examples below use the bundled binary classification dataset (`examples/sample_data.csv`).

```bash
# Train and deploy
make-mlops-easy train examples/sample_data.csv --target approved

# Inspect the latest deployment directory reported by the previous command
make-mlops-easy status models/deployment_20240101_120000

# Generate predictions (and save them to disk)
make-mlops-easy predict examples/sample_data.csv models/deployment_20240101_120000 --output predictions.json

# Produce an observability report
make-mlops-easy observe models/deployment_20240101_120000
```

All commands accept `--config path/to/config.yaml` to override defaults. Sample configurations live under `examples/pipeline/configs/`.

## Programmatic pipeline

Prefer staying inside Python? Instantiate the pipeline directly:

```python
from easy_mlops.pipeline import MLOpsPipeline

pipeline = MLOpsPipeline(config_path="examples/pipeline/configs/quickstart.yaml")
results = pipeline.run(
    data_path="examples/data/house_prices.csv",
    target_column="price",
    deploy=True,
)

predictions = pipeline.predict(
    data_path="examples/data/house_prices.csv",
    model_dir=results["deployment"]["deployment_dir"],
)
status = pipeline.get_status(results["deployment"]["deployment_dir"])
report = pipeline.observe(results["deployment"]["deployment_dir"])
```

The Python API uses the same configuration files and produces identical artifacts/logs as the CLI-driven path.

## Docker workflow

Mount local directories into the container when you want to persist datasets and models on the host:

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/models:/app/models" \
  make-mlops-easy train /data/train.csv --target price
```

Use `--master-url` to point the containerised CLI at a master running outside the container (for example, on the host).

## Next steps

- Visit the [CLI reference](cli.md) for the full command catalogue.
- Dive into the [Pipeline Architecture](pipeline.md) to learn how preprocessing, training, deployment, and observability interact.
- Explore runnable scripts and configurations in `examples/README.md`.
