# Data Preprocessing

Make MLOps Easy ships with a composable preprocessing system that turns raw tabular datasets into model-ready feature matrices. The `DataPreprocessor` class (`easy_mlops/preprocessing/preprocessor.py`) is instantiated by the pipeline and can also be reused directly in your code to prepare training or inference data.

## High-Level Workflow

1. **Configuration ingestion** – The pipeline reads the `preprocessing` section from your YAML (or a Python dict) and instantiates `DataPreprocessor` with it. When no explicit step list is supplied, legacy toggles (`handle_missing`, `encode_categorical`, `scale_features`) are honoured for backwards compatibility.
2. **Data loading** – `DataPreprocessor.load_data` accepts CSV, JSON, and Parquet files. In bespoke scripts you can bypass file IO and call `prepare_data` with an in-memory `DataFrame`.
3. **Target management** – `prepare_data(df, target_column, fit=True)` splits off the target series (if provided), keeps feature/target indices aligned, and records the target column name for later reuse.
4. **Step execution** – Configured preprocessing steps are executed sequentially. Each step implements the `PreprocessingStep` contract, so you can mix built-in and custom components. During the initial fit each step learns state (e.g. encoders, scalers) before transforming the dataset.
5. **State retention for inference** – On the first run `DataPreprocessor` snapshots feature columns, dtypes, encoders, and scalers. Subsequent calls with `fit=False` (used during deployment and prediction) reuse the fitted state, automatically realigning columns and filling any missing values so that feature matrices remain compatible with the trained model.
6. **Artifact sharing** – The fitted preprocessor is saved alongside the model artifacts. Deployment and CLI commands load it back, call `prepare_data(..., fit=False)`, and guarantee that prediction-time preprocessing mirrors the training-time pipeline.

## Step Catalogue and Configuration

### Configuration entry point

All options live under the top-level `preprocessing` key in your YAML configuration. By default the framework applies the following settings:

```yaml
preprocessing:
  handle_missing: drop          # fallbacks to sequential step definitions
  encode_categorical: true
  scale_features: true
```

To take full control of the pipeline, declare an explicit `steps` list. Steps run in the order listed. Items can be a plain string (uses default parameters) or a mapping with `type` and `params` keys for explicit configuration:

```yaml
preprocessing:
  steps:
    - type: missing_values
      params:
        strategy: median
    - type: categorical_encoder
      params:
        handle_unknown: ignore
    - feature_scaler
```

Omit a step to skip it altogether. If a `steps` list is present the legacy flags (`handle_missing`, `encode_categorical`, `scale_features`) are ignored.

### Built-in steps

#### Missing Value Handler (`missing_values`)

- **Purpose**: learns how to handle blanks and applies the same rule during inference.
- **Strategies** (`strategy`):
  - `drop` *(default)* – removes rows containing any `NaN`.
  - `mean`, `median` – compute column-wise statistics on numeric columns and fill with the learned values.
  - `mode` – fills each column with its most frequent non-null value.
  - `constant` – uses `fill_value` (scalar or column mapping) provided in the config.
- **YAML examples**:

  ```yaml
  preprocessing:
    steps:
      - type: missing_values
        params:
          strategy: constant
          fill_value:
            age: 0
            city: "unknown"
  ```

#### Categorical Encoder (`categorical_encoder`)

- **Purpose**: label-encodes categorical features with `sklearn.preprocessing.LabelEncoder`.
- **Parameters**:
  - `columns` – optional list restricting encoding to specific column names. When omitted, all object/category columns are encoded.
  - `handle_unknown` – controls unseen categories at inference time:
    - `use_first` *(default)* – swaps unknown values with the first known class before transforming.
    - `ignore` – keeps unknown values as `NaN` after transformation so that downstream logic can handle them explicitly.
- **YAML examples**:

  ```yaml
  preprocessing:
    steps:
      - type: categorical_encoder
        params:
          columns: ["gender", "segment"]
          handle_unknown: ignore
  ```

During training the encoder stores one `LabelEncoder` per fitted column. They are exposed through `DataPreprocessor.encoders` for advanced inspection or custom persistence.

#### Feature Scaler (`feature_scaler`)

- **Purpose**: standardises numeric columns using `sklearn.preprocessing.StandardScaler`.
- **Behaviour**:
  - Automatically discovers numeric columns during `fit`.
  - Reuses the fitted scaler during inference and leaves non-numeric columns untouched.
- **Configuration**:
  - In YAML you typically toggle the step on/off by adding or omitting it from the `steps` list (or set `scale_features: false` when relying on legacy flags).
  - For advanced use cases you can pass a preconfigured scaler when constructing the preprocessor from Python:

    ```python
    from sklearn.preprocessing import MinMaxScaler
    from easy_mlops.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(
        {"steps": [{"type": "feature_scaler", "params": {"scaler": MinMaxScaler()}}]}
    )
    ```

### Custom steps and extensions

The registry pattern lets you add new preprocessing capabilities without modifying the core. Create a subclass that implements `fit` and `transform`, set a unique `name`, and register it:

```python
from easy_mlops.preprocessing import DataPreprocessor, PreprocessingStep

class TextCleaner(PreprocessingStep):
    name = "text_cleaner"

    def fit(self, df):
        return self  # no state to learn

    def transform(self, df):
        df = df.copy()
        df["description"] = df["description"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
        return df

DataPreprocessor.register_step(TextCleaner)
```

Once registered you can reference the new step in your YAML:

```yaml
preprocessing:
  steps:
    - text_cleaner
    - categorical_encoder
```

### Putting it all together

Below is a realistic pipeline configuration that combines multiple built-in options:

```yaml
preprocessing:
  steps:
    - type: missing_values
      params:
        strategy: mode
    - type: categorical_encoder
      params:
        handle_unknown: use_first
    - feature_scaler

training:
  backend: sklearn
  model_type: random_forest_classifier
```

This configuration first imputes missing values with per-column modes, encodes categorical columns, and standardises numeric features before handing the dataset to the training backend. During deployment the saved preprocessor reproduces the exact transformations so your model continues to receive the feature layout it was trained on.
