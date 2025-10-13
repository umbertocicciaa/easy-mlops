import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification, make_regression

from easy_mlops.config import Config
from easy_mlops.preprocessing import DataPreprocessor
from easy_mlops.training import ModelTrainer


@pytest.fixture
def classification_data():
    """Synthetic classification dataset for tests."""
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


@pytest.fixture
def regression_data():
    """Synthetic regression dataset for tests."""
    X, y = make_regression(
        n_samples=60,
        n_features=3,
        noise=0.1,
        random_state=21,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


@pytest.fixture
def trainer_config():
    """Training configuration tuned for quick tests."""
    return {
        "test_size": 0.25,
        "random_state": 123,
        "cv_folds": 2,
        "model_type": "auto",
    }


@pytest.fixture
def trained_trainer(classification_data, trainer_config):
    """ModelTrainer instance fitted on synthetic classification data."""
    X, y = classification_data
    trainer = ModelTrainer(trainer_config)
    results = trainer.train(X, y)
    return trainer, results


@pytest.fixture
def fitted_preprocessor(classification_data):
    """DataPreprocessor fitted on synthetic data."""
    X, y = classification_data
    df = X.copy()
    df["target"] = y
    preprocessor = DataPreprocessor(
        {
            "handle_missing": "drop",
            "scale_features": True,
            "encode_categorical": False,
        }
    )
    preprocessor.prepare_data(df, target_column="target", fit=True)
    return preprocessor


@pytest.fixture
def pipeline_training_csv(tmp_path):
    """CSV file containing data with categorical and numeric features."""
    rows = []
    categories = ["A", "B", "C"]
    rng = np.random.default_rng(7)

    for idx in range(90):
        rows.append(
            {
                "feature_numeric": float(idx) / 10.0,
                "feature_noise": rng.normal(),
                "feature_categorical": categories[idx % len(categories)],
                "target": idx % 2,
            }
        )

    df = pd.DataFrame(rows)
    path = tmp_path / "training.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def pipeline_prediction_csv(tmp_path):
    """CSV file to simulate inference data (target retained to test dropping)."""
    rows = []
    categories = ["A", "B", "C"]
    rng = np.random.default_rng(11)

    for idx in range(30):
        rows.append(
            {
                "feature_numeric": float(idx) / 20.0,
                "feature_noise": rng.normal(),
                "feature_categorical": categories[idx % len(categories)],
                "target": idx % 2,
            }
        )

    df = pd.DataFrame(rows)
    path = tmp_path / "prediction.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def pipeline_config_file(tmp_path):
    """YAML configuration tuned for integration pipeline tests."""
    config = Config.DEFAULT_CONFIG.copy()

    config["training"] = {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 2,
        "model_type": "auto",
    }
    config["deployment"] = {
        "output_dir": str(tmp_path / "models"),
        "save_format": "joblib",
        "create_endpoint": False,
    }
    config["observability"] = {
        "track_metrics": True,
        "log_predictions": True,
        "alert_threshold": 0.6,
    }
    config["preprocessing"] = {
        "handle_missing": "drop",
        "scale_features": True,
        "encode_categorical": True,
    }

    path = tmp_path / "config.yaml"
    with open(path, "w") as fp:
        yaml.safe_dump(config, fp)

    return path
