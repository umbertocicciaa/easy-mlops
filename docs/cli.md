# CLI Reference

Easy MLOps exposes a Click-based command-line interface under the executable `easy-mlops`. All commands share the same configuration system and logging facilities.

## Global Usage

```bash
easy-mlops [COMMAND] [OPTIONS]
```

Run `easy-mlops --help` to see top-level commands and `easy-mlops <command> --help` for per-command options.

## Commands

### `train`

```bash
easy-mlops train DATA_PATH [--target COLUMN] [--config PATH] [--no-deploy]
```

| Option | Description |
| :----- | :---------- |
| `DATA_PATH` | Input dataset (CSV, JSON, or Parquet). |
| `--target`, `-t` | Target column name. If omitted, the last column is used. |
| `--config`, `-c` | Path to a YAML configuration file to override defaults. |
| `--no-deploy` | Skip the deployment stage and only train the model. |

The command prints pipeline progress and yields the deployment directory (unless skipped).

### `predict`

```bash
easy-mlops predict DATA_PATH MODEL_DIR [--config PATH] [--output PATH]
```

| Option | Description |
| :----- | :---------- |
| `DATA_PATH` | Dataset to score. |
| `MODEL_DIR` | Deployment directory produced by `train`. |
| `--config`, `-c` | Optional configuration overrides. |
| `--output`, `-o` | Save predictions to a JSON file. |

Predictions are displayed in the terminal. When `--output` is provided, predictions are serialized under `{"predictions": [...]}`.

### `status`

```bash
easy-mlops status MODEL_DIR [--config PATH]
```

Displays deployment metadata, evaluation metrics, and monitoring summaries by loading artifacts from `MODEL_DIR`.

### `observe`

```bash
easy-mlops observe MODEL_DIR [--config PATH]
```

Generates a textual monitoring report summarizing metric trends and prediction logs.

## Configuration Overrides

Provide a YAML configuration to customize preprocessing, training, deployment, and observability. A few highlights:

```yaml
preprocessing:
  handle_missing: mean
  scale_features: true
  encode_categorical: true

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 3
  model_type: random_forest_classifier

deployment:
  output_dir: ./models
  create_endpoint: true

observability:
  alert_threshold: 0.75
```

Pass the file with `--config path/to/config.yaml`. Absent keys fall back to defaults.

## Makefile Shortcuts

The repository includes friendly wrappers around the CLI:

- `make train DATA=... TARGET=...`
- `make predict DATA=... MODEL_DIR=...`
- `make status MODEL_DIR=...`
- `make observe MODEL_DIR=...`

These targets rely on the virtual environment created by `make install-dev`.
