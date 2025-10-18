"""Worker agent that executes tasks assigned by the master service."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

from .task_runner import TaskExecutionError, TaskRunner


class WorkerAgent:
    """Simple long-polling worker that executes tasks from the master."""

    def __init__(
        self,
        master_url: str,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        capabilities: Optional[List[str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.master_url = master_url.rstrip("/")
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.poll_interval = max(poll_interval, 0.5)
        self.capabilities = capabilities or []
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the worker loop until interrupted."""
        print(f"[Worker {self.worker_id}] Connected to master at {self.master_url}")

        try:
            while True:
                try:
                    task = self._fetch_task()
                except requests.RequestException as exc:
                    print(
                        f"[Worker {self.worker_id}] Unable to reach master: {exc}. "
                        "Retrying...",
                    )
                    time.sleep(self.poll_interval)
                    continue

                if task is None:
                    time.sleep(self.poll_interval)
                    continue

                self._execute_task(task)
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            print(f"[Worker {self.worker_id}] Shutting down gracefully.")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _fetch_task(self) -> Optional[Dict[str, Any]]:
        url = f"{self.master_url}/api/tasks/next"
        response = self.session.post(
            url,
            json={
                "worker_id": self.worker_id,
                "capabilities": self.capabilities,
            },
        )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            print(
                f"[Worker {self.worker_id}] Master rejected task poll "
                f"({response.status_code}): {detail}"
            )
            return None
        response.raise_for_status()

        payload = response.json()
        task = payload.get("task")

        if task:
            print(
                f"[Worker {self.worker_id}] Received task "
                f"{task['task_id']} ({task['operation']})"
            )

        return task

    def _execute_task(self, task: Dict[str, Any]) -> None:
        task_id = task["task_id"]
        operation = task["operation"]

        try:
            runner = TaskRunner(operation=operation, payload=task.get("payload", {}))
            result, logs = runner.run()
            self._send_completion(task_id, result, logs)
            print(f"[Worker {self.worker_id}] Completed task {task_id}")
        except TaskExecutionError as exc:
            self._send_failure(task_id, str(exc), exc.logs)
            print(
                f"[Worker {self.worker_id}] Task {task_id} failed: {exc}",
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self._send_failure(task_id, str(exc), "")
            print(
                f"[Worker {self.worker_id}] Task {task_id} failed with unexpected error: {exc}",
            )

    def _send_completion(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]],
        logs: str,
    ) -> None:
        url = f"{self.master_url}/api/tasks/{task_id}/complete"
        response = self.session.post(
            url,
            json={
                "worker_id": self.worker_id,
                "result": result,
                "logs": logs,
            },
        )
        response.raise_for_status()

    def _send_failure(self, task_id: str, error: str, logs: str) -> None:
        url = f"{self.master_url}/api/tasks/{task_id}/fail"
        response = self.session.post(
            url,
            json={
                "worker_id": self.worker_id,
                "error": error,
                "logs": logs,
            },
        )
        response.raise_for_status()
