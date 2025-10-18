"""In-memory state store with JSON persistence for distributed workflows."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def _utcnow() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat()


class StateStore:
    """Thread-safe workflow state store persisted to a JSON file."""

    def __init__(self, state_path: Optional[str] = None) -> None:
        base_dir = Path.home() / ".easy_mlops"
        base_dir.mkdir(parents=True, exist_ok=True)

        self._state_path = (
            Path(state_path) if state_path else base_dir / "master_state.json"
        )
        self._state_lock = threading.Lock()
        self._state: Dict[str, Any] = {"workflows": {}, "task_index": {}}
        self._load_state()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def create_workflow(
        self, operation: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new workflow with a single pending task."""
        workflow_id = str(uuid4())
        task_id = str(uuid4())
        timestamp = _utcnow()

        workflow = {
            "id": workflow_id,
            "operation": operation,
            "status": "pending",
            "payload": payload,
            "created_at": timestamp,
            "updated_at": timestamp,
            "result": None,
            "error": None,
            "tasks": {
                task_id: {
                    "id": task_id,
                    "operation": operation,
                    "status": "pending",
                    "payload": payload,
                    "logs": "",
                    "result": None,
                    "error": None,
                    "worker_id": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                }
            },
            "task_order": [task_id],
        }

        with self._state_lock:
            self._state["workflows"][workflow_id] = workflow
            self._state["task_index"][task_id] = workflow_id
            self._persist_state_locked()

        return self._clone(workflow)

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Fetch workflow metadata."""
        with self._state_lock:
            workflow = self._state["workflows"].get(workflow_id)
            return None if workflow is None else self._clone(workflow)

    def assign_next_task(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Assign the next pending task to a worker."""
        with self._state_lock:
            for workflow in self._state["workflows"].values():
                if workflow["status"] not in {"pending", "running"}:
                    continue

                for task_id in workflow["task_order"]:
                    task = workflow["tasks"][task_id]
                    if task["status"] != "pending":
                        continue

                    task["status"] = "running"
                    task["worker_id"] = worker_id
                    task["updated_at"] = _utcnow()

                    workflow["status"] = "running"
                    workflow["updated_at"] = _utcnow()

                    self._persist_state_locked()

                    return self._clone(
                        {
                            "task_id": task_id,
                            "workflow_id": workflow["id"],
                            "operation": task["operation"],
                            "payload": task["payload"],
                        }
                    )

        return None

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: Optional[Dict[str, Any]],
        logs: str,
    ) -> Optional[Dict[str, Any]]:
        """Mark a task as completed and update workflow state."""
        with self._state_lock:
            workflow = self._get_workflow_by_task_locked(task_id)
            if workflow is None:
                return None

            task = workflow["tasks"][task_id]
            if task["worker_id"] != worker_id:
                raise ValueError("Worker does not own this task.")

            task["status"] = "completed"
            task["result"] = result
            task["logs"] = logs
            task["updated_at"] = _utcnow()

            workflow["status"] = "completed"
            workflow["result"] = result
            workflow["updated_at"] = _utcnow()

            self._persist_state_locked()
            return self._clone(workflow)

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: str,
        logs: str,
    ) -> Optional[Dict[str, Any]]:
        """Mark a task as failed and record error details."""
        with self._state_lock:
            workflow = self._get_workflow_by_task_locked(task_id)
            if workflow is None:
                return None

            task = workflow["tasks"][task_id]
            if task["worker_id"] != worker_id:
                raise ValueError("Worker does not own this task.")

            task["status"] = "failed"
            task["error"] = error
            task["logs"] = logs
            task["updated_at"] = _utcnow()

            workflow["status"] = "failed"
            workflow["error"] = error
            workflow["updated_at"] = _utcnow()

            self._persist_state_locked()
            return self._clone(workflow)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _get_workflow_by_task_locked(self, task_id: str) -> Optional[Dict[str, Any]]:
        workflow_id = self._state["task_index"].get(task_id)
        if workflow_id is None:
            return None
        return self._state["workflows"].get(workflow_id)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return

        try:
            data = json.loads(self._state_path.read_text())
            if isinstance(data, dict):
                self._state.update(data)
                self._state.setdefault("workflows", {})
                self._state.setdefault("task_index", {})
        except json.JSONDecodeError:
            # If the state file is corrupted, start fresh but keep a backup.
            backup_path = self._state_path.with_suffix(".bak")
            self._state_path.replace(backup_path)

    def _persist_state_locked(self) -> None:
        serialized = json.dumps(self._state, indent=2)
        self._state_path.write_text(serialized)

    @staticmethod
    def _clone(item: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(item))
