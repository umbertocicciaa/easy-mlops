# Make MLOps Easy

Make MLOps easier with an AI-powered framework that abstracts all phases of an MLOps pipeline.

## Overview

Make MLOps Easy is a comprehensive framework that simplifies the entire machine learning operations pipeline. With a single command, you can:

- **Data Preprocessing**: Automatic handling of missing values, feature scaling, and categorical encoding
- **Model Training**: Intelligent model selection and hyperparameter tuning
- **Model Deployment**: Automated model packaging and endpoint creation
- **Model Observability**: Built-in monitoring, logging, and performance tracking

## Installation

### From Source

```bash
git clone https://github.com/umbertocicciaa/easy-mlops.git
cd easy-mlops
pip install -e .
```

### Using pip

```bash
pip install make-mlops-easy
```

### Using Docker

Build the image locally and expose the CLI:

```bash
docker build -t make-mlops-easy .
docker run --rm make-mlops-easy --help
```

Mount data and persist model artifacts by binding host directories:

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/models:/app/models" \
  make-mlops-easy train /data/dataset.csv --target price
```

The container entrypoint is `make-mlops-easy`, so any CLI subcommand can be passed as usual.

## Quick Start

### 1. Train a Model

Train a machine learning model with a single command:

```bash
make-mlops-easy train data.csv --target price
```

This will:
- Load and preprocess your data
- Automatically detect the problem type (classification/regression)
- Train an appropriate model
- Deploy the model with all artifacts
- Set up monitoring and logging

### 2. Make Predictions

Use your deployed model to make predictions:

```bash
make-mlops-easy predict new_data.csv models/deployment_20240101_120000
```

### 3. Check Model Status

Get information about your deployed model:

```bash
make-mlops-easy status models/deployment_20240101_120000
```

### 4. View Observability Reports

Generate detailed monitoring reports:

```bash
make-mlops-easy observe models/deployment_20240101_120000
```

### Examples & Templates

Ready-to-run scripts and curated configuration files live under
`examples/pipeline/`. The accompanying README walks through staged CLI commands,
regression/neural-network scenarios, and a programmatic pipeline example. Start
with `examples/pipeline/scripts/01_train_quickstart.sh` to see the complete
workflow in action.

## CLI Commands

### `make-mlops-easy train`

Train a machine learning model on your data.

**Usage:**
```bash
make-mlops-easy train DATA_PATH [OPTIONS]
```

**Options:**
- `-t, --target TEXT`: Name of the target column
- `-c, --config PATH`: Path to configuration file (YAML)
- `--no-deploy`: Skip deployment step (only train the model)

**Examples:**
```bash
# Basic training
make-mlops-easy train data.csv --target label

# With custom configuration
make-mlops-easy train data.json -t price -c config.yaml

# Train without deploying
make-mlops-easy train data.csv --target label --no-deploy
```

### `make-mlops-easy predict`

Make predictions using a deployed model.

**Usage:**
```bash
make-mlops-easy predict DATA_PATH MODEL_DIR [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Path to configuration file (YAML)
- `-o, --output PATH`: Output file for predictions (JSON format)

**Examples:**
```bash
# Basic prediction
make-mlops-easy predict new_data.csv models/deployment_20240101_120000

# Save predictions to file
make-mlops-easy predict data.json models/latest -o predictions.json
```

### `make-mlops-easy status`

Get status and metrics for a deployed model.

**Usage:**
```bash
make-mlops-easy status MODEL_DIR [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Path to configuration file (YAML)

**Example:**
```bash
make-mlops-easy status models/deployment_20240101_120000
```

### `make-mlops-easy observe`

Generate observability report for a deployed model.

**Usage:**
```bash
make-mlops-easy observe MODEL_DIR [OPTIONS]
```

**Options:**
- `-c, --config PATH`: Path to configuration file (YAML)
- `-o, --output PATH`: Output file for the report

**Examples:**
```bash
# View report in terminal
make-mlops-easy observe models/deployment_20240101_120000

# Save report to file
make-mlops-easy observe models/latest -o report.txt
```

### `make-mlops-easy init`

Initialize a new MLOps project with default configuration.

**Usage:**
```bash
make-mlops-easy init [OPTIONS]
```

**Options:**
- `-o, --output PATH`: Output path for the configuration file (default: mlops-config.yaml)

**Example:**
```bash
make-mlops-easy init -o my-config.yaml
```

## Configuration

Make MLOps Easy uses YAML configuration files to customize behavior. Generate a default configuration:

```bash
make-mlops-easy init
```

### Configuration Structure

```yaml
preprocessing:
  handle_missing: drop  # Options: drop, mean, median, mode
  scale_features: true
  encode_categorical: true

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  model_type: auto  # Options: auto, random_forest_classifier, random_forest_regressor, logistic_regression, linear_regression

deployment:
  output_dir: ./models
  save_format: joblib
  create_endpoint: false

observability:
  track_metrics: true
  log_predictions: true
  alert_threshold: 0.8
```

## Features

### Automatic Data Preprocessing

- **Missing Value Handling**: Multiple strategies (drop, mean, median, mode)
- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: Automatic label encoding for categorical variables
- **Multiple Format Support**: CSV, JSON, and Parquet files

### Intelligent Model Training

- **Problem Type Detection**: Automatically detects classification vs regression
- **Model Selection**: Chooses appropriate algorithms based on your data
- **Cross-Validation**: Built-in k-fold cross-validation
- **Performance Metrics**: Comprehensive metrics for model evaluation

### Streamlined Deployment

- **Model Artifacts**: Saves model, preprocessor, and metadata
- **Versioning**: Timestamped deployments for easy tracking
- **Endpoint Creation**: Optional prediction endpoint script generation
- **Reproducibility**: All components saved for consistent predictions

### Built-in Observability

- **Metrics Tracking**: Automatic logging of model performance metrics
- **Prediction Logging**: Track all predictions for analysis
- **Performance Monitoring**: Detect metric degradation over time
- **Reporting**: Generate detailed monitoring reports

## Supported Data Formats

- CSV (`.csv`)
- JSON (`.json`)
- Parquet (`.parquet`)

## Supported Model Types

### Classification
- Random Forest Classifier
- Logistic Regression

### Regression
- Random Forest Regressor
- Linear Regression

## Project Structure

```
easy-mlops/
├── easy_mlops/
│   ├── __init__.py
│   ├── cli.py                 # CLI interface
│   ├── pipeline.py            # Main pipeline orchestrator
│   ├── config/                # Configuration management
│   ├── preprocessing/         # Data preprocessing
│   ├── training/              # Model training
│   ├── deployment/            # Model deployment
│   └── observability/         # Model monitoring
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.10
- click >= 8.0.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- joblib >= 1.0.0
- pyyaml >= 6.0
- rich >= 10.0.0

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black easy_mlops/
```

### Documentation

Project documentation is powered by MkDocs and the Material theme. Source files live in `docs/`, and the site configuration is `mkdocs.yml`.

```bash
# Serve docs locally (http://localhost:8000)
make docs-serve

# Build static site into site/
make docs-build
```

The GitHub Actions workflow `.github/workflows/docs.yml` automatically builds and deploys the site to GitHub Pages on pushes to `main`. Enable GitHub Pages in the repository settings (source: “GitHub Actions”). For manual publishing or previews outside CI, run:

```bash
make docs-deploy
```

### Type Checking

```bash
mypy easy_mlops/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Umberto Domenico Ciccia

## Links

- GitHub: https://github.com/umbertocicciaa/easy-mlops
- Issues: https://github.com/umbertocicciaa/easy-mlops/issues
