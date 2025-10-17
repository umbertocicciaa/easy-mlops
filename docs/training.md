# Model Training

The training stage in Make MLOps Easy is orchestrated by the `ModelTrainer` class (`easy_mlops/training/trainer.py`). It turns the `training` configuration section into concrete training runs by delegating to pluggable backends, handling evaluation, and synchronising state for downstream deployment and observability steps.

## Trainer Architecture

`ModelTrainer` wraps a backend that implements the `BaseTrainingBackend` contract. The lifecycle is consistent across backends:

1. **Configuration ingestion** – The pipeline passes the `training` section from your YAML (or dict). If no `backend` is declared, the trainer defaults to `sklearn`. Common keys include `test_size`, `random_state`, `cv_folds`, and an optional `model_type`.
2. **Backend selection** – `ModelTrainer.available_backends()` exposes registered backends. You can extend the registry with `ModelTrainer.register_backend(CustomBackend)`.
3. **Problem-type detection** – When you call `train(X, y)`, the backend inspects the label series to decide between `classification` and `regression` (heuristics live in `BaseTrainingBackend.detect_problem_type` but can be overridden).
4. **Data splitting** – `BaseTrainingBackend.split_data` produces train/test folds. Classification problems are stratified automatically. Override or adjust `test_size`/`random_state` in config as needed.
5. **Model selection & fitting** – Each backend maps the `problem_type` and `model_type` into a concrete estimator. The estimator is trained on the split created above.
6. **Evaluation & cross-validation** – `BaseTrainingBackend.evaluate` computes default metrics (accuracy/F1 for classification, MSE/RMSE/R² for regression). If `cv_folds > 1`, `cross_validate` adds `cv_mean` and `cv_std` from a scikit-learn `cross_val_score` run. Backends may plug in custom scoring.
7. **Result packaging** – Backends return a `TrainingRunResult`, which stores the fitted model, metrics, model metadata, and dataset dimensions. `ModelTrainer.train` surfaces this as a dictionary and keeps the backend state in sync for later prediction, deployment (`save_model`), and reloading (`load_model`).

### Backend responsibilities

`BaseTrainingBackend` defines the template methods a backend can override:

- `train` **(required)** – fit a model and return `TrainingRunResult`.
- `detect_problem_type`, `split_data`, `evaluate`, `cross_validate` – optional hooks with sensible defaults.
- `predict`, `save_model`, `load_model` – handle inference and persistence. The default implementations store metadata with `joblib`, but specialised backends may customise them.

Because the trainer only interacts with the backend interface, you can introduce new frameworks by subclassing `BaseTrainingBackend` and registering it. The pipeline will automatically pick it up when the `training.backend` field matches the registered `name`.

## Model Backends & Configuration

### Shared training settings

Regardless of backend, the following YAML snippet illustrates the baseline options:

```yaml
training:
  backend: sklearn        # defaults to sklearn if omitted
  model_type: auto        # backend-specific catalogue
  test_size: 0.2          # float in (0, 1), controls hold-out split
  random_state: 42        # used by split + compatible estimators
  cv_folds: 5             # set to 0 or 1 to disable cross validation
```

Additional keys are backend-specific and described below.

### Scikit-learn backend (`backend: sklearn`)

`SklearnBackend` covers traditional estimators and is the default choice. Supported `model_type` values are:

- `auto` *(default)* – picks `RandomForestClassifier` for classification, `RandomForestRegressor` for regression.
- `random_forest_classifier` / `random_forest_regressor` – explicit random forest variants; honours `random_state`.
- `logistic_regression` – multi-class logistic regression (`max_iter` fixed at 1000).
- `linear_regression` – ordinary least squares regression.
- `xgboost` – delegates to `XGBClassifier` or `XGBRegressor`. Ensure the `xgboost` extra is installed in your environment.

Sample configuration:

```yaml
training:
  backend: sklearn
  model_type: logistic_regression
  test_size: 0.3
  cv_folds: 10
  random_state: 123
```

### Neural network backend (`backend: neural_network`)

`NeuralNetworkBackend` builds on the scikit-learn MLP estimators (`MLPClassifier`/`MLPRegressor`). Alongside the shared keys, it recognises:

- `hidden_layer_sizes` (tuple of ints, default `(50, 50)`)
- `max_iter` (default `200`)
- Optional scikit-learn MLP params: `activation`, `solver`, `alpha`, `learning_rate`, `learning_rate_init`, `early_stopping`

Example:

```yaml
training:
  backend: neural_network
  hidden_layer_sizes: [128, 64, 32]
  activation: relu
  solver: adam
  max_iter: 500
  test_size: 0.15
  cv_folds: 0          # disable CV for faster experimentation
```

### Callable family backends (`backend: callable`, `deep_learning`, `nlp`)

These backends allow you to plug in arbitrary frameworks (PyTorch, TensorFlow, Hugging Face, etc.). They share the `CallableBackend` implementation and therefore require you to provide callables in the configuration dictionary:

- `build_model(config, problem_type)` – instantiates a model.
- `train_fn(model, X_train, y_train, config)` – trains the model in-place.
- `predict_fn(model, X)` – generates predictions; used for evaluation and inference.

Optional hooks:

- `evaluate_fn(model, X_test, y_test, config)` – custom metric computation.
- `cv_fn(model, X, y, config)` – custom cross-validation routine that should return a `dict` of metrics.
- `save_fn(model, output_path, metadata, config)` / `load_fn(model_path, config)` – override persistence.

YAML cannot serialise Python callables directly, so these backends are typically configured programmatically:

```python
from easy_mlops.training import ModelTrainer
from my_project.training import build_model, train_model, predict

trainer = ModelTrainer(
    {
        "backend": "deep_learning",
        "build_model": build_model,
        "train_fn": train_model,
        "predict_fn": predict,
        "evaluate_fn": custom_eval,   # optional
        "test_size": 0.1,
        "cv_folds": 0,
    }
)
```

Use the `deep_learning` or `nlp` backend names when you want clearer intent in logs; they behave identically to `callable`.

### Registering additional backends

To expose another model catalogue, subclass `BaseTrainingBackend`, implement `train`, and any extra hooks you need. Set a unique `name` attribute and register it:

```python
from easy_mlops.training import ModelTrainer, BaseTrainingBackend

class LightGBMBackend(BaseTrainingBackend):
    name = "lightgbm"
    # implement train(...) and overrides here

ModelTrainer.register_backend(LightGBMBackend)
```

Once registered, `training.backend: lightgbm` becomes available in the YAML configuration without further pipeline changes.
