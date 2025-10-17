"""Model training module for Make MLOps Easy."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from easy_mlops.training.backends import (
    BaseTrainingBackend,
    TrainerRegistry,
    TrainingRunResult,
)


class ModelTrainer:
    """Training orchestrator that delegates work to registered backends."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model trainer.

        Args:
            config: Training configuration dictionary.
        """
        self.config = config or {}
        self._init_backend(self.config.get("backend", "sklearn"))

    # ------------------------------------------------------------------ #
    # Backend management                                                 #
    # ------------------------------------------------------------------ #
    def _init_backend(self, backend_name: str) -> None:
        backend_cls = TrainerRegistry.get_backend(backend_name)
        self.backend: BaseTrainingBackend = backend_cls(self.config)
        self.model: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.is_classifier: Optional[bool] = None

    @staticmethod
    def available_backends() -> Iterable[str]:
        """Expose registered backend names."""
        return TrainerRegistry.available_backends()

    @staticmethod
    def register_backend(backend: type[BaseTrainingBackend]) -> None:
        """Register a new backend implementation."""
        TrainerRegistry.register_backend(backend)

    def _sync_state(self) -> None:
        self.model = self.backend.model
        self.model_type = self.backend.problem_type
        self.metrics = self.backend.metrics
        self.is_classifier = self.backend.is_classifier

    # ------------------------------------------------------------------ #
    # Legacy-compatible convenience methods                              #
    # ------------------------------------------------------------------ #
    def detect_problem_type(self, y: pd.Series) -> str:
        """Detect whether the task is classification or regression."""
        return self.backend.detect_problem_type(y)

    def select_model(self, problem_type: str, model_type: Optional[str] = None) -> Any:
        """Select or build a model using the active backend."""
        if not hasattr(self.backend, "select_model"):
            raise AttributeError(
                f"Backend '{self.backend.name}' does not expose 'select_model'."
            )

        desired_model_type = model_type or self.config.get("model_type")
        return getattr(self.backend, "select_model")(problem_type, desired_model_type)

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test partitions."""
        if self.is_classifier is None:
            raise ValueError(
                "Call train() or detect_problem_type() before splitting data."
            )
        return self.backend.split_data(X, y, self.is_classifier)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def train(
        self, X: pd.DataFrame, y: pd.Series, model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train model on provided data."""
        result: TrainingRunResult = self.backend.train(X, y, model_type=model_type)
        self._sync_state()

        return {
            "model_type": result.model_type,
            "model_name": result.model_name,
            "metrics": result.metrics,
            "n_features": result.n_features,
            "n_samples": result.n_samples,
            "backend": self.backend.name,
        }

    def predict(self, X: pd.DataFrame):
        """Generate predictions using the trained backend."""
        predictions = self.backend.predict(X)
        self.model = self.backend.model  # keep state aligned
        return predictions

    def save_model(self, output_path: str) -> None:
        """Persist trained model via the backend."""
        self.backend.save_model(output_path)

    def load_model(self, model_path: str) -> None:
        """Load a persisted model via the backend."""
        self.backend.load_model(model_path)
        self._sync_state()
