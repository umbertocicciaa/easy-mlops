"""Data preprocessing module for Make MLOps Easy."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class DataPreprocessor:
    """Handles data preprocessing for ML pipelines."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data preprocessor.

        Args:
            config: Preprocessing configuration dictionary.
        """
        self.config = config
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = None
        self.target_column = None

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

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        strategy = self.config.get("handle_missing", "drop")

        if strategy == "drop":
            return df.dropna()
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df
        elif strategy == "mode":
            for col in df.columns:
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else df[col]
                )
            return df
        else:
            return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame.
            fit: Whether to fit encoders or use existing ones.

        Returns:
            DataFrame with encoded categorical variables.
        """
        if not self.config.get("encode_categorical", True):
            return df

        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    # Handle unseen categories
                    known_classes = set(self.encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: (
                            x if x in known_classes else self.encoders[col].classes_[0]
                        )
                    )
                    df[col] = self.encoders[col].transform(df[col].astype(str))

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input DataFrame.
            fit: Whether to fit scaler or use existing one.

        Returns:
            DataFrame with scaled features.
        """
        if not self.config.get("scale_features", True):
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

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

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical variables
        df = self.encode_categorical(df, fit=fit)

        # Scale features
        df = self.scale_features(df, fit=fit)

        # Store feature columns
        if fit:
            self.feature_columns = df.columns.tolist()

        return df, y

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
