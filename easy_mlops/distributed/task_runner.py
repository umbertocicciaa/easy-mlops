"""Task execution helpers used by worker agents."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

from easy_mlops.pipeline import MLOpsPipeline

try:  # numpy is optional until training runs
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    np = None  # type: ignore


class TaskExecutionError(RuntimeError):
    """Raised when task execution fails."""

    def __init__(self, message: str, logs: str) -> None:
        super().__init__(message)
        self.logs = logs


class TaskRunner:
    """Execute a workflow operation based on task payload."""

    def __init__(self, operation: str, payload: Dict[str, Any]):
        self.operation = operation
        self.payload = payload

    def run(self) -> tuple[Optional[Dict[str, Any]], str]:
        """Execute the task and return (result, logs)."""
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                result = self._execute()
        except Exception as exc:  # pragma: no cover - runtime behaviour
            logs = buffer.getvalue()
            raise TaskExecutionError(str(exc), logs) from exc

        logs = buffer.getvalue()
        return result, logs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _execute(self) -> Optional[Dict[str, Any]]:
        operation = self.operation.lower()
        if operation == "train":
            return self._run_train()
        if operation == "predict":
            return self._run_predict()
        if operation == "status":
            return self._run_status()
        if operation == "observe":
            return self._run_observe()

        raise ValueError(f"Unsupported operation: {self.operation}")

    def _run_train(self) -> Dict[str, Any]:
        pipeline = MLOpsPipeline(config_path=self.payload.get("config_path"))
        results = pipeline.run(
            data_path=self.payload["data_path"],
            target_column=self.payload.get("target"),
            deploy=self.payload.get("deploy", True),
        )
        return {"operation": "train", "data": _make_json_safe(results)}

    def _run_predict(self) -> Dict[str, Any]:
        pipeline = MLOpsPipeline(config_path=self.payload.get("config_path"))
        predictions = pipeline.predict(
            data_path=self.payload["data_path"],
            model_dir=self.payload["model_dir"],
        )

        serializable = (
            predictions.tolist()
            if hasattr(predictions, "tolist")
            else list(predictions)
        )

        output_path = self.payload.get("output_path")
        if output_path:
            Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"predictions": serializable}, f, indent=2)

        return {
            "operation": "predict",
            "predictions": serializable,
            "output_path": output_path,
        }

    def _run_status(self) -> Dict[str, Any]:
        pipeline = MLOpsPipeline(config_path=self.payload.get("config_path"))
        status = pipeline.get_status(model_dir=self.payload["model_dir"])
        return {"operation": "status", "status": _make_json_safe(status)}

    def _run_observe(self) -> Dict[str, Any]:
        pipeline = MLOpsPipeline(config_path=self.payload.get("config_path"))
        report = pipeline.observe(model_dir=self.payload["model_dir"])

        output_path = self.payload.get("output_path")
        if output_path:
            Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).expanduser().write_text(report)

        return {"operation": "observe", "report": report, "output_path": output_path}


def _make_json_safe(data: Any) -> Any:
    """Recursively convert objects into JSON-serializable structures."""
    if isinstance(data, dict):
        return {str(key): _make_json_safe(value) for key, value in data.items()}

    if isinstance(data, (list, tuple, set)):
        return [_make_json_safe(value) for value in data]

    if isinstance(data, Path):
        return str(data)

    if np is not None:
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, np.generic):
            return data.item()

    try:
        import pandas as pd  # local import to avoid hard dependency earlier

        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
        if isinstance(data, pd.Series):
            return data.tolist()
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        pass

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    return str(data)
