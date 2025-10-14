"""Model training module for Make MLOps Easy."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBRegressor, XGBClassifier
from typing import Dict, Any, Optional, Tuple
import joblib


class ModelTrainer:
    """Handles model training for ML pipelines."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer.

        Args:
            config: Training configuration dictionary.
        """
        self.config = config
        self.model = None
        self.model_type = None
        self.metrics = {}
        self.is_classifier = None

    def detect_problem_type(self, y: pd.Series) -> str:
        """Detect whether problem is classification or regression.

        Args:
            y: Target variable.

        Returns:
            'classification' or 'regression'.
        """
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            # Check unique values ratio
            unique_ratio = y.nunique() / len(y)
            if unique_ratio < 0.05 or y.nunique() <= 20:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"

    def select_model(self, problem_type: str, model_type: Optional[str] = None):
        """Select appropriate model based on problem type.

        Args:
            problem_type: 'classification' or 'regression'.
            model_type: Specific model type or 'auto' for automatic selection.

        Returns:
            Initialized model instance.
        """
        if model_type is None:
            model_type = self.config.get("model_type", "auto")

        if model_type == "auto":
            if problem_type == "classification":
                return RandomForestClassifier(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                )
            else:
                return RandomForestRegressor(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.config.get("random_state", 42),
                max_iter=1000,
            )
        elif model_type == "linear_regression":
            return LinearRegression()
        elif model_type == "random_forest_classifier":
            return RandomForestClassifier(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
            )
        elif model_type == "random_forest_regressor":
            return RandomForestRegressor(
                random_state=self.config.get("random_state", 42),
                n_estimators=100,
            )
        elif model_type == "xgboost":
            if problem_type == "classification":
                return XGBClassifier(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                    learning_rate=0.1,
                )
            else:
                return XGBRegressor(
                    random_state=self.config.get("random_state", 42),
                    n_estimators=100,
                    learning_rate=0.1,
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", 42)

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if self.is_classifier else None,
        )

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of metrics.
        """
        y_pred = self.model.predict(X_test)
        metrics = {}

        if self.is_classifier:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")
        else:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2_score"] = r2_score(y_test, y_pred)

        return metrics

    def train(
        self, X: pd.DataFrame, y: pd.Series, model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train model on data.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            model_type: Specific model type or None for auto-detection.

        Returns:
            Dictionary with training results and metrics.
        """
        # Detect problem type
        problem_type = self.detect_problem_type(y)
        self.is_classifier = problem_type == "classification"
        self.model_type = problem_type

        # Select model
        self.model = self.select_model(problem_type, model_type)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        self.metrics = self.evaluate_model(X_test, y_test)

        # Cross-validation
        cv_folds = self.config.get("cv_folds", 5)
        if cv_folds > 1:
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
            self.metrics["cv_mean"] = cv_scores.mean()
            self.metrics["cv_std"] = cv_scores.std()

        return {
            "model_type": problem_type,
            "model_name": self.model.__class__.__name__,
            "metrics": self.metrics,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model.

        Args:
            X: Feature DataFrame.

        Returns:
            Predictions array.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def save_model(self, output_path: str) -> None:
        """Save trained model to disk.

        Args:
            output_path: Path to save model file.

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "is_classifier": self.is_classifier,
                "metrics": self.metrics,
            },
            output_path,
        )

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk.

        Args:
            model_path: Path to model file.
        """
        data = joblib.load(model_path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.is_classifier = data["is_classifier"]
        self.metrics = data.get("metrics", {})
