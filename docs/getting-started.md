# Quick Start

This guide walks through installing Make MLOps Easy, preparing a dataset, and running the end-to-end pipeline locally.

## Prerequisites

- Python 3.10 or later
- `git`
- (Optional) Docker for containerized runs

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/umbertocicciaa/easy-mlops.git
cd easy-mlops
pip install -e .
```

`make install-dev` creates a virtual environment, installs the project, and pulls in development tools such as pytest, black, flake8, and MkDocs.

### Alternative: pip install

You will be able to install the released package from PyPI:

```bash
pip install make-mlops-easy
```

## Training a Model

Prepare a CSV file containing your features and target column (the target is assumed to be the last column when not specified). Then run:

```bash
make-mlops-easy train DATA=data/train.csv TARGET=price
```

This executes the CLI, which orchestrates preprocessing, training, deployment, and observability setup. Artifacts are written to `models/deployment_<timestamp>/`.

## Making Predictions

To score a new dataset using a previously deployed model directory:

```bash
make-mlops-easy predict DATA=data/new_samples.csv MODEL_DIR=models/deployment_20240101_120000 OUTPUT=predictions.json
```

Predictions are printed to the console and saved to `predictions.json` if the `OUTPUT` argument is supplied.

## Docker Workflow

Build and run the containerized CLI if you prefer not to manage Python environments:

```bash
make docker-build
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/models:/app/models" \
  make-mlops-easy train /data/train.csv --target price
```

Mount a host directory when you need to persist models or pass datasets into the container.

## Next Steps

- Explore the [`CLI Reference`](cli.md) for command options.
- Review the [`Pipeline`](pipeline.md) section to understand the architecture.
- Visit [`Development`](development.md) to contribute or extend the framework.
