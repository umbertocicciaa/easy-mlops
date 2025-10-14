import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from easy_mlops.training import ModelTrainer


def test_detect_problem_type_classification(classification_data, trainer_config):
    _, y = classification_data
    trainer = ModelTrainer(trainer_config)
    assert trainer.detect_problem_type(y) == "classification"


def test_detect_problem_type_regression(regression_data, trainer_config):
    _, y = regression_data
    trainer = ModelTrainer(trainer_config)
    assert trainer.detect_problem_type(y) == "regression"


def test_train_classification_produces_metrics(classification_data, trainer_config):
    X, y = classification_data
    trainer = ModelTrainer(trainer_config)
    results = trainer.train(X, y)

    assert trainer.model is not None
    assert results["model_type"] == "classification"
    assert "accuracy" in results["metrics"]
    assert "cv_mean" in results["metrics"]


def test_predict_requires_training(classification_data, trainer_config):
    X, _ = classification_data
    trainer = ModelTrainer(trainer_config)

    with pytest.raises(ValueError):
        trainer.predict(X)


def test_save_model_requires_training(tmp_path, trainer_config):
    trainer = ModelTrainer(trainer_config)
    with pytest.raises(ValueError):
        trainer.save_model(str(tmp_path / "model.joblib"))


def test_save_and_load_model_roundtrip(tmp_path, classification_data, trainer_config):
    X, y = classification_data
    trainer = ModelTrainer(trainer_config)
    trainer.train(X, y)

    model_path = tmp_path / "model.joblib"
    trainer.save_model(str(model_path))
    assert model_path.exists()

    new_trainer = ModelTrainer(trainer_config)
    new_trainer.load_model(str(model_path))

    assert new_trainer.model is not None
    assert new_trainer.model.__class__ == trainer.model.__class__
    assert new_trainer.metrics.keys() == trainer.metrics.keys()


def test_select_model_variants(trainer_config):
    trainer = ModelTrainer(trainer_config)

    assert isinstance(
        trainer.select_model("classification", "logistic_regression"),
        LogisticRegression,
    )
    assert isinstance(
        trainer.select_model("regression", "linear_regression"),
        LinearRegression,
    )
    assert isinstance(
        trainer.select_model("classification", "random_forest_classifier"),
        RandomForestClassifier,
    )
    assert isinstance(
        trainer.select_model("regression", "random_forest_regressor"),
        RandomForestRegressor,
    )
    assert isinstance(
        trainer.select_model("regression", "xgboost"),
        XGBRegressor,
    )
    assert isinstance(
        trainer.select_model("classification", "xgboost"),
        XGBClassifier,
    )

    with pytest.raises(ValueError):
        trainer.select_model("classification", "unknown_model")


def test_train_regression_computes_metrics(regression_data, trainer_config):
    X, y = regression_data
    trainer = ModelTrainer(trainer_config)
    results = trainer.train(X, y, model_type="linear_regression")

    assert results["model_type"] == "regression"
    assert "mse" in results["metrics"]
    assert "rmse" in results["metrics"]
    assert "r2_score" in results["metrics"]
