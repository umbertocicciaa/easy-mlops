import pandas as pd
import pytest

from easy_mlops.preprocessing import DataPreprocessor, PreprocessingStep


def test_preprocess_handles_missing_values(tmp_path):
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", None, "b", "a"],
            "target": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    preprocessor = DataPreprocessor(
        {
            "handle_missing": "mean",
            "scale_features": True,
            "encode_categorical": True,
        }
    )

    X, y = preprocessor.preprocess(str(path), target_column="target")

    assert y.tolist() == [0, 1, 0, 1]
    assert not X.isna().values.any()
    assert preprocessor.feature_columns == ["num", "cat"]


def test_prepare_data_reuses_encoders_for_unseen_values():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "category": ["red", "blue", "red", "green"],
            "target": [0, 1, 0, 1],
        }
    )

    preprocessor = DataPreprocessor(
        {
            "handle_missing": "drop",
            "scale_features": False,
            "encode_categorical": True,
        }
    )

    X, _ = preprocessor.prepare_data(df, target_column="target", fit=True)
    assert "category" in preprocessor.encoders
    known_values = set(
        preprocessor.encoders["category"].transform(["red", "blue", "green"])
    )

    new_df = pd.DataFrame(
        {
            "num": [5, 6],
            "category": ["yellow", "red"],  # yellow is unseen
        }
    )

    X_new, _ = preprocessor.prepare_data(new_df, target_column=None, fit=False)
    assert set(X_new["category"]) <= known_values


def test_load_data_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("raw data")

    preprocessor = DataPreprocessor(
        {
            "handle_missing": "drop",
            "scale_features": False,
            "encode_categorical": False,
        }
    )

    with pytest.raises(ValueError):
        preprocessor.load_data(str(path))


def test_prepare_data_aligns_columns_on_inference():
    df = pd.DataFrame(
        {
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )

    preprocessor = DataPreprocessor(
        {
            "handle_missing": "drop",
            "scale_features": True,
            "encode_categorical": False,
        }
    )

    X_train, _ = preprocessor.prepare_data(df, target_column="target", fit=True)

    assert preprocessor.feature_columns == list(X_train.columns)

    new_df = pd.DataFrame(
        {
            "feature_1": [10.0, 11.0],
            "feature_extra": [99.0, 100.0],
            "feature_0": [7.0, 8.0],
        }
    )

    X_new, _ = preprocessor.prepare_data(new_df, fit=False)

    assert list(X_new.columns) == preprocessor.feature_columns
    assert "feature_extra" not in X_new.columns


def test_steps_configuration_builds_pipeline():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 22],
            "city": ["Rome", "Milan", "Rome", "Naples"],
            "target": [1, 0, 1, 0],
        }
    )

    preprocessor = DataPreprocessor(
        {
            "steps": [
                {"type": "missing_values", "params": {"strategy": "mean"}},
                "categorical_encoder",
                "feature_scaler",
            ]
        }
    )

    X, y = preprocessor.prepare_data(df, target_column="target", fit=True)

    assert y is not None
    assert "city" in preprocessor.encoders
    assert pytest.approx(X["age"].mean(), abs=1e-6) == 0.0


def test_custom_step_registration_allows_extension():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "remove_me": [9, 9, 9],
        }
    )

    class DropColumnsStep(PreprocessingStep):
        name = "drop_columns"

        def __init__(self, columns):
            super().__init__(columns=columns)
            self.columns = list(columns)
            self._is_fitted = False

        @classmethod
        def from_config(cls, params=None):
            params = params or {}
            return cls(columns=params.get("columns", []))

        def fit(self, df):
            self._is_fitted = True
            return self

        def transform(self, df):
            if not self._is_fitted:
                raise RuntimeError("DropColumnsStep must be fitted before transform.")
            cols = [col for col in self.columns if col in df.columns]
            return df.drop(columns=cols)

    original_registry = DataPreprocessor.STEP_REGISTRY.copy()
    try:
        DataPreprocessor.register_step(DropColumnsStep)
        preprocessor = DataPreprocessor(
            {
                "steps": [
                    {"type": "drop_columns", "params": {"columns": ["remove_me"]}},
                ]
            }
        )

        X, _ = preprocessor.prepare_data(df, fit=True)
        assert "remove_me" not in X.columns
        assert "feature" in X.columns
    finally:
        DataPreprocessor.STEP_REGISTRY = original_registry
