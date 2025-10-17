"""Training backend abstractions for Make MLOps Easy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class TrainingRunResult:
    """Container for training outputs emitted by a backend."""

    model: Any
    metrics: Dict[str, float]
    model_name: str
    model_type: str
    is_classifier: bool
    n_features: int
    n_samples: int


class BaseTrainingBackend(ABC):
    """Abstract base class for training backends."""

    name: str = "base"
    supported_problem_types: Iterable[str] = ("classification", "regression")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.model: Optional[Any] = None
        self.metrics: Dict[str, float] = {}
        self.problem_type: Optional[str] = None
        self.is_classifier: Optional[bool] = None

    # --------------------------------------------------------------------- #
    # Core template methods                                                 #
    # --------------------------------------------------------------------- #
    @abstractmethod
    def train(
        self, X: pd.DataFrame, y: pd.Series, model_type: Optional[str] = None
    ) -> TrainingRunResult:
        """Train a model and return the result object."""

    # --------------------------------------------------------------------- #
    # Default hooks that backends may override                              #
    # --------------------------------------------------------------------- #
    def detect_problem_type(self, y: pd.Series) -> str:
        """Infer the problem type from the target."""
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique() / max(len(y), 1)
            if unique_ratio < 0.05 or y.nunique() <= 20:
                return "classification"
            return "regression"
        return "classification"

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classifier: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and validation sets."""
        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", 42)
        stratify = y if is_classifier else None
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        is_classifier: bool,
    ) -> Dict[str, float]:
        """Compute default evaluation metrics."""
        y_pred = model.predict(X_test)
        metrics: Dict[str, float] = {}

        if is_classifier:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
        else:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["r2_score"] = r2_score(y_test, y_pred)

        return metrics

    def cross_validate(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Run cross validation when requested."""
        cv_folds = self.config.get("cv_folds", 5)
        if cv_folds and cv_folds > 1:
            scores = cross_val_score(model, X, y, cv=cv_folds)
            return {"cv_mean": scores.mean(), "cv_std": scores.std()}
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate predictions to the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save_model(self, output_path: str) -> None:
        """Persist the trained model using joblib."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.problem_type,
                "is_classifier": self.is_classifier,
                "metrics": self.metrics,
                "backend": self.name,
            },
            output_path,
        )

    def load_model(self, model_path: str) -> None:
        """Load a previously persisted model."""
        data = joblib.load(model_path)
        self.model = data["model"]
        self.problem_type = data.get("model_type")
        self.is_classifier = data.get("is_classifier")
        self.metrics = data.get("metrics", {})

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #
    def _finalize(
        self,
        result: TrainingRunResult,
    ) -> TrainingRunResult:
        """Persist shared state and return the result."""
        self.model = result.model
        self.metrics = result.metrics
        self.problem_type = result.model_type
        self.is_classifier = result.is_classifier
        return result


class SklearnBackend(BaseTrainingBackend):
    """Backend dedicated to traditional scikit-learn style estimators."""

    name = "sklearn"

    def select_model(self, problem_type: str, model_type: Optional[str]) -> Any:
        """Create an estimator based on the configuration."""
        if model_type is None or model_type == "auto":
            if problem_type == "classification":
                return RandomForestClassifier(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                )
            return RandomForestRegressor(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
            )

        if model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.config.get("random_state", 42),
                max_iter=1000,
            )
        if model_type == "linear_regression":
            return LinearRegression()
        if model_type == "random_forest_classifier":
            return RandomForestClassifier(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
            )
        if model_type == "random_forest_regressor":
            return RandomForestRegressor(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
            )
        if model_type == "xgboost":
            if problem_type == "classification":
                return XGBClassifier(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                    learning_rate=0.1,
                )
            return XGBRegressor(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
                learning_rate=0.1,
            )

        raise ValueError(f"Unknown model type for scikit-learn backend: {model_type}")

    def train(
        self, X: pd.DataFrame, y: pd.Series, model_type: Optional[str] = None
    ) -> TrainingRunResult:
        problem_type = self.detect_problem_type(y)
        is_classifier = problem_type == "classification"
        model = self.select_model(problem_type, model_type)

        X_train, X_test, y_train, y_test = self.split_data(X, y, is_classifier)
        model.fit(X_train, y_train)

        metrics = self.evaluate(model, X_test, y_test, is_classifier)
        metrics.update(self.cross_validate(model, X, y))

        return self._finalize(
            TrainingRunResult(
                model=model,
                metrics=metrics,
                model_name=model.__class__.__name__,
                model_type=problem_type,
                is_classifier=is_classifier,
                n_features=X.shape[1],
                n_samples=X.shape[0],
            )
        )


class NeuralNetworkBackend(SklearnBackend):
    """Backend leveraging scikit-learn neural network estimators."""

    name = "neural_network"

    def select_model(self, problem_type: str, model_type: Optional[str]) -> Any:
        """Create a neural network model for the given problem."""
        random_state = self.config.get("random_state", 42)
        hidden_layer_sizes = self.config.get("hidden_layer_sizes", (50, 50))
        max_iter = self.config.get("max_iter", 200)
        optional_keys = [
            "activation",
            "solver",
            "alpha",
            "learning_rate",
            "learning_rate_init",
            "early_stopping",
        ]

        model_kwargs = {
            "random_state": random_state,
            "hidden_layer_sizes": hidden_layer_sizes,
            "max_iter": max_iter,
        }
        for key in optional_keys:
            if key in self.config:
                model_kwargs[key] = self.config[key]

        if problem_type == "classification":
            return MLPClassifier(**model_kwargs)
        return MLPRegressor(**model_kwargs)


class CallableBackend(BaseTrainingBackend):
    """Backend driven entirely by user-provided callables."""

    name = "callable"
    required_callables = ("build_model", "train_fn", "predict_fn")

    def _get_callable(self, name: str) -> Callable[..., Any]:
        fn = self.config.get(name)
        if not callable(fn):
            raise ValueError(
                f"Callable backend requires a callable '{name}' entry in the config."
            )
        return fn

    def train(
        self, X: pd.DataFrame, y: pd.Series, model_type: Optional[str] = None
    ) -> TrainingRunResult:
        problem_type = (
            model_type or self.config.get("problem_type") or self.detect_problem_type(y)
        )
        is_classifier = problem_type == "classification"

        build_model = self._get_callable("build_model")
        train_fn = self._get_callable("train_fn")
        predict_fn = self._get_callable("predict_fn")

        X_train, X_test, y_train, y_test = self.split_data(X, y, is_classifier)

        model = build_model(self.config, problem_type)
        train_fn(model, X_train, y_train, self.config)

        evaluate_fn = self.config.get("evaluate_fn")
        if callable(evaluate_fn):
            metrics = evaluate_fn(model, X_test, y_test, self.config)
        else:
            metrics = self._evaluate_with_custom_predict(
                predict_fn, model, X_test, y_test, is_classifier
            )

        cv_fn = self.config.get("cv_fn")
        if callable(cv_fn):
            metrics.update(cv_fn(model, X, y, self.config))

        wrapped_model = _CallableModelWrapper(model, predict_fn)

        return self._finalize(
            TrainingRunResult(
                model=wrapped_model,
                metrics=metrics,
                model_name=model.__class__.__name__,
                model_type=problem_type,
                is_classifier=is_classifier,
                n_features=X.shape[1],
                n_samples=X.shape[0],
            )
        )

    def _evaluate_with_custom_predict(
        self,
        predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        is_classifier: bool,
    ) -> Dict[str, float]:
        y_pred = predict_fn(model, X_test)
        if is_classifier:
            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }
        mse = mean_squared_error(y_test, y_pred)
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2_score": r2_score(y_test, y_pred),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def save_model(self, output_path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        save_fn = self.config.get("save_fn")
        if callable(save_fn):
            save_fn(
                self.model.base_model,
                output_path,
                {
                    "model_type": self.problem_type,
                    "is_classifier": self.is_classifier,
                    "metrics": self.metrics,
                    "backend": self.name,
                },
                self.config,
            )
            return

        super().save_model(output_path)

    def load_model(self, model_path: str) -> None:
        load_fn = self.config.get("load_fn")
        if callable(load_fn):
            restored = load_fn(model_path, self.config)
            if isinstance(restored, tuple) and len(restored) == 2:
                base_model, metadata = restored
            elif isinstance(restored, dict):
                base_model = restored["model"]
                metadata = restored.get("metadata", {})
            else:
                raise ValueError(
                    "Custom load_fn must return (model, metadata) or {'model': model}."
                )

            predict_fn = self._get_callable("predict_fn")
            self.model = _CallableModelWrapper(base_model, predict_fn)
            self.problem_type = metadata.get("model_type")
            self.is_classifier = metadata.get("is_classifier")
            self.metrics = metadata.get("metrics", {})
            return

        super().load_model(model_path)


class DeepLearningBackend(CallableBackend):
    """Convenience backend alias tailored to deep learning workloads."""

    name = "deep_learning"


class NLPBackend(CallableBackend):
    """Convenience backend alias tailored to NLP workloads."""

    name = "nlp"


class TrainerRegistry:
    """Simple registry to track available backends."""

    _registry: Dict[str, Type[BaseTrainingBackend]] = {}

    @classmethod
    def register_backend(cls, backend: Type[BaseTrainingBackend]) -> None:
        cls._registry[backend.name] = backend

    @classmethod
    def get_backend(cls, name: str) -> Type[BaseTrainingBackend]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown training backend '{name}'. "
                f"Available options: {', '.join(sorted(cls._registry.keys()))}"
            )
        return cls._registry[name]

    @classmethod
    def available_backends(cls) -> Iterable[str]:
        return tuple(sorted(cls._registry.keys()))


@dataclass
class _CallableModelWrapper:
    """Thin wrapper that provides a predict interface for callable backends."""

    base_model: Any
    predict_fn: Callable[[Any, pd.DataFrame], np.ndarray]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_fn(self.base_model, X)


# Register built-in backends
TrainerRegistry.register_backend(SklearnBackend)
TrainerRegistry.register_backend(NeuralNetworkBackend)
TrainerRegistry.register_backend(CallableBackend)
TrainerRegistry.register_backend(DeepLearningBackend)
TrainerRegistry.register_backend(NLPBackend)
