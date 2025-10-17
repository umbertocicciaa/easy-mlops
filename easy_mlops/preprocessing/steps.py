"""Composable preprocessing step implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreprocessingStep(ABC):
    """Contract for preprocessing steps that can be composed into a pipeline."""

    #: Unique identifier used inside configuration dictionaries.
    name: str

    def __init__(self, **params):
        self.params = params

    @classmethod
    def from_config(
        cls, params: Optional[Dict[str, object]] = None
    ) -> "PreprocessingStep":
        """Instantiate the step from configuration parameters."""
        params = params or {}
        return cls(**params)

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "PreprocessingStep":
        """Learn state from the provided dataframe."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned state to transform the dataframe."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default implementation wiring fit and transform together."""
        self.fit(df)
        return self.transform(df)


@dataclass
class MissingValueHandler(PreprocessingStep):
    """Handle missing values according to a configured strategy."""

    strategy: str = "drop"
    fill_value: Optional[object] = None
    name: str = "missing_values"
    _numeric_fill_values: Optional[pd.Series] = field(default=None, init=False)
    _mode_fill_values: Optional[Dict[str, object]] = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.strategy not in {"drop", "mean", "median", "mode", "constant"}:
            # Fallback to drop to preserve backwards compatibility with unknown strategies.
            self.strategy = "drop"

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        if self.strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self._numeric_fill_values = df[numeric_cols].mean()
        elif self.strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self._numeric_fill_values = df[numeric_cols].median()
        elif self.strategy == "mode":
            self._mode_fill_values = {
                col: (
                    series.mode(dropna=True)[0]
                    if not series.mode(dropna=True).empty
                    else np.nan
                )
                for col, series in df.items()
            }
        elif self.strategy == "constant":
            if isinstance(self.fill_value, dict):
                self._mode_fill_values = self.fill_value
            else:
                self._mode_fill_values = {col: self.fill_value for col in df.columns}
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if not self._is_fitted and self.strategy != "drop":
            raise RuntimeError(
                "MissingValueHandler must be fitted before calling transform."
            )

        if self.strategy == "drop":
            return df.dropna()

        df_out = df.copy()

        if (
            self.strategy in {"mean", "median"}
            and self._numeric_fill_values is not None
        ):
            numeric_cols = self._numeric_fill_values.index
            df_out.loc[:, numeric_cols] = df_out.loc[:, numeric_cols].fillna(
                self._numeric_fill_values
            )
            return df_out

        if self.strategy in {"mode", "constant"} and self._mode_fill_values is not None:
            fill_map = {
                col: val
                for col, val in self._mode_fill_values.items()
                if col in df_out.columns
            }
            if fill_map:
                df_out = df_out.fillna(fill_map)
            return df_out

        return df_out


@dataclass
class CategoricalEncoder(PreprocessingStep):
    """Encode categorical columns via sklearn's LabelEncoder."""

    columns: Optional[Iterable[str]] = None
    handle_unknown: str = "use_first"
    name: str = "categorical_encoder"
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict, init=False)
    _fitted_columns: List[str] = field(default_factory=list, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        if self.columns is not None:
            target_columns = [col for col in self.columns if col in df.columns]
        else:
            target_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self.encoders = {}
        self._fitted_columns = target_columns

        for col in target_columns:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            self.encoders[col] = encoder

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "CategoricalEncoder must be fitted before calling transform."
            )

        if not self._fitted_columns:
            return df

        df_out = df.copy()

        for col in self._fitted_columns:
            if col not in df_out.columns:
                continue

            encoder = self.encoders.get(col)
            if encoder is None:
                continue

            series = df_out[col].astype(str)
            known_classes = set(encoder.classes_)

            if not series.isin(known_classes).all():
                if self.handle_unknown == "use_first":
                    fallback = encoder.classes_[0]
                    series = series.where(series.isin(known_classes), fallback)
                elif self.handle_unknown == "ignore":
                    valid_mask = series.isin(known_classes)
                    series = series[valid_mask]
                    encoded = encoder.transform(series)
                    df_out.loc[valid_mask, col] = encoded
                    df_out.loc[~valid_mask, col] = np.nan
                    continue
                else:
                    raise ValueError(
                        f"Unsupported handle_unknown option: {self.handle_unknown}"
                    )

            df_out.loc[:, col] = encoder.transform(series)

        return df_out


@dataclass
class FeatureScaler(PreprocessingStep):
    """Scale numerical columns using sklearn's StandardScaler."""

    scaler: Optional[StandardScaler] = None
    name: str = "feature_scaler"
    _numeric_columns: List[str] = field(default_factory=list, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.scaler is None:
            self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_columns = numeric_cols

        if numeric_cols:
            self.scaler.fit(df[numeric_cols])

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform.")

        if not self._numeric_columns:
            return df

        df_out = df.copy()
        df_out.loc[:, self._numeric_columns] = self.scaler.transform(
            df_out[self._numeric_columns]
        )
        return df_out


DEFAULT_STEP_REGISTRY: Dict[str, type[PreprocessingStep]] = {
    MissingValueHandler.name: MissingValueHandler,
    CategoricalEncoder.name: CategoricalEncoder,
    FeatureScaler.name: FeatureScaler,
}

__all__ = [
    "PreprocessingStep",
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureScaler",
    "DEFAULT_STEP_REGISTRY",
]
