"""Data preprocessing module for Make MLOps Easy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from easy_mlops.preprocessing.steps import (
    CategoricalEncoder,
    DEFAULT_STEP_REGISTRY,
    FeatureScaler,
    MissingValueHandler,
    PreprocessingStep,
)


class DataPreprocessor:
    """Handles data preprocessing for ML pipelines."""

    STEP_REGISTRY: Dict[str, Type[PreprocessingStep]] = DEFAULT_STEP_REGISTRY.copy()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data preprocessor.

        Args:
            config: Preprocessing configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}
        self.steps: List[PreprocessingStep] = self._initialize_steps()
        self.feature_columns: Optional[List[str]] = None
        self.feature_dtypes: Dict[str, Any] = {}
        self.target_column: Optional[str] = None
        self.encoders = {}
        self.scaler = None
        self._refresh_step_shortcuts()

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file.

        Args:
            data_path: Path to data file (CSV, JSON, or Parquet).

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If file format is not supported.
        """
        data_path = Path(data_path)

        if data_path.suffix == ".csv":
            return pd.read_csv(data_path)
        elif data_path.suffix == ".json":
            return pd.read_json(data_path)
        elif data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    @classmethod
    def register_step(cls, step: Type[PreprocessingStep]) -> None:
        """Register a custom preprocessing step class."""
        if not issubclass(step, PreprocessingStep):
            raise TypeError("Custom step must inherit from PreprocessingStep.")
        cls.STEP_REGISTRY[step.name] = step

    def get_step(self, name: str) -> Optional[PreprocessingStep]:
        """Retrieve a step instance by its registry name."""
        return next((step for step in self.steps if step.name == name), None)

    def handle_missing_values(
        self, df: pd.DataFrame, fit: Optional[bool] = None
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame via the configured step."""
        step = self.get_step(MissingValueHandler.name)
        if step is None:
            return df

        should_fit = fit if fit is not None else not getattr(step, "_is_fitted", False)
        return step.fit_transform(df) if should_fit else step.transform(df)

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables via the configured step."""
        step = self.get_step(CategoricalEncoder.name)
        if step is None:
            return df

        result = step.fit_transform(df) if fit else step.transform(df)
        self.encoders = getattr(step, "encoders", {})
        return result

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features via the configured step."""
        step = self.get_step(FeatureScaler.name)
        if step is None:
            return df

        result = step.fit_transform(df) if fit else step.transform(df)
        self.scaler = getattr(step, "scaler", None)
        return result

    def prepare_data(
        self, df: pd.DataFrame, target_column: Optional[str] = None, fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training or prediction.

        Args:
            df: Input DataFrame.
            target_column: Name of target column. If None, no target is extracted.
            fit: Whether to fit transformers or use existing ones.

        Returns:
            Tuple of (features DataFrame, target Series or None).
        """
        # Store target column name
        if target_column:
            self.target_column = target_column

        # Extract target if present
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column]
            df = df.drop(columns=[target_column])

        # Apply preprocessing steps
        df_processed = df.copy()
        for step in self.steps:
            df_processed = (
                step.fit_transform(df_processed)
                if fit
                else step.transform(df_processed)
            )

        # Refresh shortcuts to expose fitted state (scaler/encoders)
        self._refresh_step_shortcuts()

        # Store feature columns
        if fit:
            self.feature_columns = df_processed.columns.tolist()
            self.feature_dtypes = {
                column: df_processed[column].dtype for column in df_processed.columns
            }

        # Align target with processed features if rows were dropped
        if y is not None:
            y = y.loc[df_processed.index]

        if not fit and self.feature_columns:
            existing_columns = set(df_processed.columns)
            df_processed = df_processed.reindex(columns=self.feature_columns)

            for column in self.feature_columns:
                dtype = self.feature_dtypes.get(column)
                kind = getattr(dtype, "kind", None) if dtype is not None else None

                if column not in existing_columns:
                    if kind in {"U", "S", "O"}:
                        df_processed[column] = ""
                    elif kind == "b":
                        df_processed[column] = False
                    else:
                        df_processed[column] = 0
                    continue

                if kind in {"U", "S", "O"}:
                    df_processed[column] = df_processed[column].fillna("")
                elif kind == "b":
                    df_processed[column] = df_processed[column].fillna(False)
                else:
                    df_processed[column] = df_processed[column].fillna(0)

        return df_processed, y

    def preprocess(
        self, data_path: str, target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Complete preprocessing pipeline.

        Args:
            data_path: Path to data file.
            target_column: Name of target column.

        Returns:
            Tuple of (processed features, target).
        """
        df = self.load_data(data_path)
        return self.prepare_data(df, target_column, fit=True)

    def _initialize_steps(self) -> List[PreprocessingStep]:
        """Create step instances from configuration."""
        steps_config = self.config.get("steps")

        if steps_config:
            return [self._build_step_from_spec(spec) for spec in steps_config]

        # Backwards-compatible defaults
        steps: List[PreprocessingStep] = []
        missing_strategy = self.config.get("handle_missing", "drop")
        steps.append(
            self._create_step(MissingValueHandler.name, {"strategy": missing_strategy})
        )

        if self.config.get("encode_categorical", True):
            steps.append(self._create_step(CategoricalEncoder.name, {}))

        if self.config.get("scale_features", True):
            steps.append(self._create_step(FeatureScaler.name, {}))

        return steps

    def _create_step(
        self, step_name: str, params: Optional[Dict[str, Any]] = None
    ) -> PreprocessingStep:
        """Instantiate a step from the registry and provided parameters."""
        params = params or {}
        registry = self.STEP_REGISTRY

        if step_name not in registry:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

        step_cls = registry[step_name]
        return step_cls.from_config(params)

    def _build_step_from_spec(self, spec: Any) -> PreprocessingStep:
        """Build a step instance from a configuration spec."""
        if isinstance(spec, str):
            return self._create_step(spec)

        if isinstance(spec, dict):
            step_type = spec.get("type")
            if not step_type:
                raise ValueError(
                    "Step configuration dictionaries must include a 'type' key."
                )
            params = spec.get("params") or {}
            return self._create_step(step_type, params)

        raise TypeError("Step specification must be a string or dict.")

    def _refresh_step_shortcuts(self) -> None:
        """Expose commonly used state (scaler, encoders) for compatibility."""
        categorical_step = self.get_step(CategoricalEncoder.name)
        self.encoders = (
            getattr(categorical_step, "encoders", {}) if categorical_step else {}
        )

        scaler_step = self.get_step(FeatureScaler.name)
        self.scaler = getattr(scaler_step, "scaler", None) if scaler_step else None
