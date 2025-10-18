"""FastAPI application exposing distributed workflow endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .state import StateStore
from .errors import format_error, safe_call


class TrainRequest(BaseModel):
    data_path: str = Field(..., description="Path to the training dataset.")
    target: Optional[str] = Field(
        default=None, description="Target column name. Defaults to last column."
    )
    config_path: Optional[str] = Field(
        default=None, description="Optional configuration file path."
    )
    deploy: bool = Field(
        default=True, description="Whether to run deployment after training."
    )


class PredictRequest(BaseModel):
    data_path: str = Field(..., description="Path to the inference dataset.")
    model_dir: str = Field(..., description="Deployment directory to load.")
    config_path: Optional[str] = Field(
        default=None, description="Optional configuration file path."
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Optional path to persist predictions (handled by worker).",
    )


class StatusRequest(BaseModel):
    model_dir: str = Field(..., description="Deployment directory to inspect.")
    config_path: Optional[str] = Field(
        default=None, description="Optional configuration file path."
    )


class ObserveRequest(BaseModel):
    model_dir: str = Field(..., description="Deployment directory to observe.")
    config_path: Optional[str] = Field(
        default=None, description="Optional configuration file path."
    )


class TaskPollRequest(BaseModel):
    worker_id: str = Field(..., description="Unique identifier for the worker.")
    capabilities: Optional[List[str]] = Field(
        default=None, description="Optional list of worker capabilities."
    )


class TaskCompleteRequest(BaseModel):
    worker_id: str
    result: Optional[Dict[str, Any]] = None
    logs: str = ""


class TaskFailRequest(BaseModel):
    worker_id: str
    error: str
    logs: str = ""


def create_app(state_path: Optional[str] = None) -> FastAPI:
    """Create a FastAPI app bound to a state store."""
    store = StateStore(state_path=state_path)
    app = FastAPI(title="Easy MLOps Master", version="1.0.0")

    # ------------------------------------------------------------------ #
    # Routes                                                             #
    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    def _create_workflow(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        workflow, error = safe_call(store.create_workflow, operation, payload)
        if error:
            exc, tb = error
            raise HTTPException(status_code=500, detail=format_error(exc, tb))
        return {"workflow_id": workflow["id"], "status": workflow["status"]}

    @app.post("/api/workflows/train")
    def submit_train(req: TrainRequest) -> Dict[str, Any]:
        return _create_workflow("train", req.model_dump())

    @app.post("/api/workflows/predict")
    def submit_predict(req: PredictRequest) -> Dict[str, Any]:
        return _create_workflow("predict", req.model_dump())

    @app.post("/api/workflows/status")
    def submit_status(req: StatusRequest) -> Dict[str, Any]:
        return _create_workflow("status", req.model_dump())

    @app.post("/api/workflows/observe")
    def submit_observe(req: ObserveRequest) -> Dict[str, Any]:
        return _create_workflow("observe", req.model_dump())

    @app.get("/api/workflows/{workflow_id}")
    def fetch_workflow(workflow_id: str) -> Dict[str, Any]:
        workflow = store.get_workflow(workflow_id)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found.")
        return workflow

    @app.post("/api/tasks/next")
    def request_next_task(req: TaskPollRequest) -> Dict[str, Any]:
        task, error = safe_call(store.assign_next_task, str(req.worker_id))
        if error:
            exc, tb = error
            raise HTTPException(status_code=500, detail=format_error(exc, tb))
        return {"task": task}

    @app.post("/api/tasks/{task_id}/complete")
    def complete_task(task_id: str, req: TaskCompleteRequest) -> Dict[str, Any]:
        workflow, error = safe_call(
            store.complete_task,
            task_id=task_id,
            worker_id=req.worker_id,
            result=req.result,
            logs=req.logs,
        )
        if error:
            exc, tb = error
            raise HTTPException(status_code=500, detail=format_error(exc, tb))
        if workflow is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return workflow

    @app.post("/api/tasks/{task_id}/fail")
    def fail_task(task_id: str, req: TaskFailRequest) -> Dict[str, Any]:
        workflow, error = safe_call(
            store.fail_task,
            task_id=task_id,
            worker_id=req.worker_id,
            error=req.error,
            logs=req.logs,
        )
        if error:
            exc, tb = error
            raise HTTPException(status_code=500, detail=format_error(exc, tb))
        if workflow is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return workflow

    return app


# Default app used by uvicorn when importing module path.
app = create_app()
