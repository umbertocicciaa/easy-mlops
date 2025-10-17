import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from easy_mlops.training import ModelTrainer


def build_fake_dl_model(config, problem_type):
    if problem_type != "classification":
        raise ValueError("Fake DL model only supports classification in tests.")
    return LogisticRegression(max_iter=200)


def fake_dl_train_fn(model, X_train, y_train, _config):
    model.fit(X_train, y_train)


def fake_dl_predict_fn(model, X):
    return model.predict(X)


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
    assert results["backend"] == "sklearn"


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


def test_available_backends_includes_new_options():
    backends = set(ModelTrainer.available_backends())
    assert {"sklearn", "neural_network", "deep_learning", "nlp"}.issubset(backends)


def test_neural_network_backend_training(classification_data, trainer_config):
    X, y = classification_data
    config = dict(trainer_config)
    config.update(
        {
            "backend": "neural_network",
            "hidden_layer_sizes": (8, 4),
            "max_iter": 200,
            "solver": "lbfgs",
        }
    )
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)

    assert isinstance(trainer.model, MLPClassifier)
    assert results["backend"] == "neural_network"
    assert "accuracy" in results["metrics"]


def test_deep_learning_backend_with_callables(classification_data, trainer_config):
    X, y = classification_data
    config = dict(trainer_config)
    config.update(
        {
            "backend": "deep_learning",
            "problem_type": "classification",
            "build_model": build_fake_dl_model,
            "train_fn": fake_dl_train_fn,
            "predict_fn": fake_dl_predict_fn,
            "cv_fn": lambda model, X_data, y_data, _: {
                "cv_mean": float(np.mean(model.predict(X_data) == y_data))
            },
        }
    )
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)

    assert results["backend"] == "deep_learning"
    assert "accuracy" in results["metrics"]
    assert "cv_mean" in results["metrics"]
    assert trainer.model.predict(X).shape[0] == X.shape[0]


@pytest.mark.parametrize(
    "model_type,is_classifier,fixture_name",
    [
        ("logistic_regression", True, "classification_data"),
        ("random_forest_classifier", True, "classification_data"),
        ("xgboost", True, "classification_data"),
        ("linear_regression", False, "regression_data"),
        ("random_forest_regressor", False, "regression_data"),
        ("xgboost", False, "regression_data"),
    ],
)
def test_train_handles_all_supported_model_types(
    request, trainer_config, model_type, is_classifier, fixture_name
):
    data = request.getfixturevalue(fixture_name)
    X, y = data
    config = dict(trainer_config)
    config["cv_folds"] = 2
    trainer = ModelTrainer(config)

    results = trainer.train(X, y, model_type=model_type)

    expected_type = "classification" if is_classifier else "regression"
    assert results["model_type"] == expected_type
    assert results["metrics"]
    assert trainer.model is not None


def test_neural_network_backend_regression(regression_data, trainer_config):
    X, y = regression_data
    config = dict(trainer_config)
    config.update({"backend": "neural_network", "max_iter": 100})
    trainer = ModelTrainer(config)

    results = trainer.train(X, y)

    assert results["backend"] == "neural_network"
    assert isinstance(trainer.model, MLPRegressor)
    assert "mse" in results["metrics"]


def test_split_data_requires_training(classification_data, trainer_config):
    X, y = classification_data
    trainer = ModelTrainer(trainer_config)

    with pytest.raises(ValueError):
        trainer.split_data(X, y)

    trainer.train(X, y)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


def test_callable_backend_missing_callable_raises(classification_data):
    X, y = classification_data
    config = {
        "backend": "deep_learning",
        "problem_type": "classification",
        "build_model": lambda _config, _ptype: LogisticRegression(max_iter=200),
        "train_fn": lambda model, X_train, y_train, _config: model.fit(
            X_train, y_train
        ),
        # Intentionally omit predict_fn
    }
    trainer = ModelTrainer(config)

    with pytest.raises(ValueError, match="predict_fn"):
        trainer.train(X, y)


def test_callable_backend_custom_hooks(tmp_path, classification_data):
    X, y = classification_data
    saved_metadata = {}
    save_path = tmp_path / "custom_model.joblib"

    def build_model(_config, _ptype):
        return LogisticRegression(max_iter=200)

    def train_fn(model, X_train, y_train, _config):
        model.fit(X_train, y_train)

    def predict_fn(model, data):
        return model.predict(data)

    def evaluate_fn(model, X_eval, y_eval, _config):
        return {"custom_metric": float(np.mean(model.predict(X_eval) == y_eval))}

    def save_fn(model, output_path, metadata, _config):
        saved_metadata.update(metadata)
        joblib.dump({"model": model, "metadata": metadata}, output_path)

    def load_fn(model_path, _config):
        data = joblib.load(model_path)
        return data["model"], data["metadata"]

    config = {
        "backend": "deep_learning",
        "problem_type": "classification",
        "build_model": build_model,
        "train_fn": train_fn,
        "predict_fn": predict_fn,
        "evaluate_fn": evaluate_fn,
        "save_fn": save_fn,
        "load_fn": load_fn,
    }

    trainer = ModelTrainer(config)
    results = trainer.train(X, y)

    assert "custom_metric" in results["metrics"]

    trainer.save_model(str(save_path))
    assert save_path.exists()
    assert saved_metadata["backend"] == "deep_learning"

    restored_trainer = ModelTrainer(config)
    restored_trainer.load_model(str(save_path))

    preds = restored_trainer.predict(X)
    assert preds.shape == y.shape
