"""Tests for the distributed StateStore."""

from easy_mlops.distributed.state import StateStore


def test_state_store_workflow_lifecycle(tmp_path):
    state_path = tmp_path / "state.json"
    store = StateStore(state_path=str(state_path))

    workflow = store.create_workflow("train", {"foo": "bar"})
    workflow_id = workflow["id"]

    # Pending workflow should be retrievable
    fetched = store.get_workflow(workflow_id)
    assert fetched is not None
    assert fetched["status"] == "pending"

    # Worker claims the task
    task = store.assign_next_task("worker-1")
    assert task is not None
    task_id = task["task_id"]

    # Completing the task should mark workflow as completed
    completed = store.complete_task(
        task_id=task_id,
        worker_id="worker-1",
        result={"status": "ok"},
        logs="done",
    )
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == {"status": "ok"}

    final = store.get_workflow(workflow_id)
    assert final["status"] == "completed"
