import pandas as pd
import pytest

from easy_mlops.preprocessing import DataPreprocessor


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
