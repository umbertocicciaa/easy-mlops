"""Command-line interface for Make MLOps Easy."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import requests
import uvicorn
from requests import RequestException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from easy_mlops.distributed.master import create_app
from easy_mlops.distributed.worker import WorkerAgent

DEFAULT_MASTER_URL = os.environ.get("EASY_MLOPS_MASTER_URL", "http://127.0.0.1:8000")
DEFAULT_POLL_INTERVAL = float(os.environ.get("EASY_MLOPS_POLL_INTERVAL", "2.0"))

WORKFLOW_ENDPOINTS = {
    "train": "/api/workflows/train",
    "predict": "/api/workflows/predict",
    "status": "/api/workflows/status",
    "observe": "/api/workflows/observe",
}

console = Console()


# ---------------------------------------------------------------------- #
# Utility helpers                                                        #
# ---------------------------------------------------------------------- #
def _require_existing_file(path: str, description: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        console.print(f"[bold red]✗ {description} not found:[/bold red] {path}")
        sys.exit(1)
    return str(resolved)


def _require_existing_dir(path: str, description: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        console.print(f"[bold red]✗ {description} not found:[/bold red] {path}")
        sys.exit(1)
    return str(resolved)


def _optional_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        console.print(f"[bold red]✗ Configuration file not found:[/bold red] {path}")
        sys.exit(1)
    return str(resolved)


def _absolute_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def _submit_workflow(master_url: str, operation: str, payload: Dict[str, Any]) -> str:
    endpoint = WORKFLOW_ENDPOINTS[operation]
    url = f"{master_url.rstrip('/')}{endpoint}"

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail: Optional[str] = None
        if exc.response is not None:
            try:
                data = exc.response.json()
                detail = data.get("detail") if isinstance(data, dict) else None
            except ValueError:
                detail = exc.response.text
        message = f"{exc}"
        if detail:
            message = f"{message}\n  → {detail}"
        console.print(f"[bold red]✗ Failed to submit workflow:[/bold red] {message}")
        sys.exit(1)
    except RequestException as exc:
        console.print(f"[bold red]✗ Failed to submit workflow:[/bold red] {exc}")
        sys.exit(1)

    data = response.json()
    workflow_id = data.get("workflow_id")
    if not workflow_id:
        console.print("[bold red]✗ Master did not return a workflow id.[/bold red]")
        sys.exit(1)
    return workflow_id


def _fetch_workflow(master_url: str, workflow_id: str) -> Dict[str, Any]:
    url = f"{master_url.rstrip('/')}/api/workflows/{workflow_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _wait_for_completion(
    master_url: str, workflow_id: str, poll_interval: float
) -> Dict[str, Any]:
    with console.status(
        f"[cyan]Workflow {workflow_id} pending...[/cyan]", spinner="dots"
    ) as status:
        while True:
            try:
                workflow = _fetch_workflow(master_url, workflow_id)
            except RequestException as exc:
                status.update(f"[yellow]Reconnecting to master: {exc}[/yellow]")
                time.sleep(min(poll_interval, 5))
                continue

            current_status = workflow.get("status", "unknown")
            status.update(
                f"[cyan]Workflow {workflow_id} status[/cyan]: [bold]{current_status}[/bold]"
            )

            if current_status in {"completed", "failed"}:
                return workflow

            time.sleep(poll_interval)


def _display_workflow_result(workflow: Dict[str, Any]) -> None:
    operation = workflow.get("operation", "unknown")
    status = workflow.get("status")

    if status == "failed":
        _print_failure(workflow)
        _print_logs(workflow)
        sys.exit(1)

    if operation == "train":
        _print_train_result(workflow)
    elif operation == "predict":
        _print_predict_result(workflow)
    elif operation == "status":
        _print_status_result(workflow)
    elif operation == "observe":
        _print_observe_result(workflow)
    else:
        console.print(
            f"[bold green]✓ Workflow completed:[/bold green] {workflow.get('id')}"
        )

    _print_logs(workflow)


def _print_failure(workflow: Dict[str, Any]) -> None:
    error = workflow.get("error")
    if not error:
        tasks = workflow.get("tasks", {})
        for task in tasks.values():
            if task.get("error"):
                error = task["error"]
                break

    console.print(f"[bold red]✗ Workflow {workflow.get('id')} failed.[/bold red]")
    if error:
        console.print(f"[bold]Reason:[/bold] {error}")


def _print_logs(workflow: Dict[str, Any]) -> None:
    tasks = workflow.get("tasks", {})
    for task_id, task in tasks.items():
        logs = (task.get("logs") or "").strip()
        if not logs:
            continue
        panel_title = f"Logs • {task.get('operation', task_id)}"
        console.print(
            Panel(
                logs,
                title=panel_title,
                title_align="left",
                border_style="dim",
            )
        )


def _print_train_result(workflow: Dict[str, Any]) -> None:
    result = workflow.get("result") or {}
    data = result.get("data", {})
    training = data.get("training", {})
    deployment = data.get("deployment", {})

    console.print(
        f"[bold green]✓ Training workflow {workflow.get('id')} completed[/bold green]"
    )

    model_name = training.get("model_name")
    model_type = training.get("model_type")
    if model_name or model_type:
        console.print(f"  Model: {model_name or 'N/A'} ({model_type or 'N/A'})")

    metrics = training.get("metrics", {})
    if metrics:
        table = Table(title="Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value")
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.4f}")
            else:
                table.add_row(metric, str(value))
        console.print(table)

    if deployment:
        console.print("\n[bold]Deployment Artifacts:[/bold]")
        for key, value in deployment.items():
            console.print(f"  {key}: {value}")


def _print_predict_result(workflow: Dict[str, Any]) -> None:
    result = workflow.get("result") or {}
    predictions = result.get("predictions") or []
    output_path = result.get("output_path")

    console.print(
        f"[bold green]✓ Prediction workflow {workflow.get('id')} completed[/bold green]"
    )
    console.print(f"  Total predictions: {len(predictions)}")

    if predictions:
        sample = predictions[:10]
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Index")
        table.add_column("Prediction")
        for idx, pred in enumerate(sample):
            table.add_row(str(idx), str(pred))
        console.print(table)

        if len(predictions) > len(sample):
            console.print(f"  ... and {len(predictions) - len(sample)} more")

    if output_path:
        console.print(f"  Saved to: {output_path}")


def _print_status_result(workflow: Dict[str, Any]) -> None:
    result = workflow.get("result") or {}
    status_info = result.get("status", {})

    console.print(
        f"[bold green]✓ Status workflow {workflow.get('id')} completed[/bold green]"
    )

    metadata = status_info.get("metadata", {})
    if metadata:
        console.print("[bold]Model Information:[/bold]")
        console.print(f"  Version: {metadata.get('version', 'N/A')}")
        console.print(f"  Status: {metadata.get('status', 'N/A')}")
        console.print(f"  Deployed: {metadata.get('deployment_time', 'N/A')}")

    metrics = status_info.get("metrics", {})
    if metrics:
        console.print("\n[bold]Metrics:[/bold]")
        for metric, value in metrics.items():
            if isinstance(value, float):
                console.print(f"  {metric}: {value:.4f}")
            else:
                console.print(f"  {metric}: {value}")

    monitoring = status_info.get("monitoring", {})
    metrics_summary = monitoring.get("metrics_summary", {})
    predictions_summary = monitoring.get("predictions_summary", {})

    if metrics_summary:
        console.print("\n[bold]Metrics Summary:[/bold]")
        for key, value in metrics_summary.items():
            console.print(f"  {key}: {value}")

    if predictions_summary:
        console.print("\n[bold]Predictions Summary:[/bold]")
        for key, value in predictions_summary.items():
            console.print(f"  {key}: {value}")


def _print_observe_result(workflow: Dict[str, Any]) -> None:
    result = workflow.get("result") or {}
    report = result.get("report", "")
    output_path = result.get("output_path")

    console.print(
        f"[bold green]✓ Observe workflow {workflow.get('id')} completed[/bold green]"
    )

    if report:
        console.print(Panel(report, title="Observability Report", border_style="cyan"))

    if output_path:
        console.print(f"  Saved to: {output_path}")


# ---------------------------------------------------------------------- #
# CLI                                                                   #
# ---------------------------------------------------------------------- #
@click.group()
@click.version_option(version="0.4.0")
def main():
    """Make MLOps Easy - orchestrate pipelines using the distributed runtime."""


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "-t",
    default=None,
    help="Name of the target column. If not specified, uses the last column.",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=str,
    help="Path to configuration file (YAML).",
)
@click.option(
    "--no-deploy",
    is_flag=True,
    help="Skip deployment step (only train the model).",
)
@click.option(
    "--master-url",
    default=DEFAULT_MASTER_URL,
    show_default=True,
    help="URL of the master service.",
)
@click.option(
    "--poll-interval",
    default=DEFAULT_POLL_INTERVAL,
    show_default=True,
    help="Seconds between status checks.",
)
def train(data_path, target, config, no_deploy, master_url, poll_interval):
    """Train a machine learning model on your data via the distributed runtime."""
    console.print("\n[bold blue]Submitting training workflow[/bold blue]\n")

    payload = {
        "data_path": _require_existing_file(data_path, "Training data"),
        "target": target,
        "config_path": _optional_file(config),
        "deploy": not no_deploy,
    }
    workflow_id = _submit_workflow(master_url, "train", payload)
    console.print(f"[dim]Workflow id:[/dim] {workflow_id}")

    workflow = _wait_for_completion(master_url, workflow_id, poll_interval)
    _display_workflow_result(workflow)


@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("model_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=str,
    help="Path to configuration file (YAML).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Optional path to store predictions as JSON.",
)
@click.option(
    "--master-url",
    default=DEFAULT_MASTER_URL,
    show_default=True,
    help="URL of the master service.",
)
@click.option(
    "--poll-interval",
    default=DEFAULT_POLL_INTERVAL,
    show_default=True,
    help="Seconds between status checks.",
)
def predict(data_path, model_dir, config, output, master_url, poll_interval):
    """Make predictions using a deployed model via the distributed runtime."""
    console.print("\n[bold blue]Submitting prediction workflow[/bold blue]\n")

    payload = {
        "data_path": _require_existing_file(data_path, "Prediction data"),
        "model_dir": _require_existing_dir(model_dir, "Model directory"),
        "config_path": _optional_file(config),
        "output_path": _absolute_path(output),
    }
    workflow_id = _submit_workflow(master_url, "predict", payload)
    console.print(f"[dim]Workflow id:[/dim] {workflow_id}")

    workflow = _wait_for_completion(master_url, workflow_id, poll_interval)
    _display_workflow_result(workflow)


@main.command()
@click.argument("model_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=str,
    help="Path to configuration file (YAML).",
)
@click.option(
    "--master-url",
    default=DEFAULT_MASTER_URL,
    show_default=True,
    help="URL of the master service.",
)
@click.option(
    "--poll-interval",
    default=DEFAULT_POLL_INTERVAL,
    show_default=True,
    help="Seconds between status checks.",
)
def status(model_dir, config, master_url, poll_interval):
    """Get status and metrics for a deployed model via the distributed runtime."""
    console.print("\n[bold blue]Submitting status workflow[/bold blue]\n")

    payload = {
        "model_dir": _require_existing_dir(model_dir, "Model directory"),
        "config_path": _optional_file(config),
    }
    workflow_id = _submit_workflow(master_url, "status", payload)
    console.print(f"[dim]Workflow id:[/dim] {workflow_id}")

    workflow = _wait_for_completion(master_url, workflow_id, poll_interval)
    _display_workflow_result(workflow)


@main.command()
@click.argument("model_dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=str,
    help="Path to configuration file (YAML).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=str,
    help="Optional path to save the observability report.",
)
@click.option(
    "--master-url",
    default=DEFAULT_MASTER_URL,
    show_default=True,
    help="URL of the master service.",
)
@click.option(
    "--poll-interval",
    default=DEFAULT_POLL_INTERVAL,
    show_default=True,
    help="Seconds between status checks.",
)
def observe(model_dir, config, output, master_url, poll_interval):
    """Generate an observability report via the distributed runtime."""
    console.print("\n[bold blue]Submitting observability workflow[/bold blue]\n")

    payload = {
        "model_dir": _require_existing_dir(model_dir, "Model directory"),
        "config_path": _optional_file(config),
        "output_path": _absolute_path(output),
    }
    workflow_id = _submit_workflow(master_url, "observe", payload)
    console.print(f"[dim]Workflow id:[/dim] {workflow_id}")

    workflow = _wait_for_completion(master_url, workflow_id, poll_interval)
    _display_workflow_result(workflow)


@main.group()
def master():
    """Manage the master orchestration service."""


@master.command("start")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8000, show_default=True, help="Listen port.")
@click.option(
    "--state-path",
    default=None,
    type=str,
    help="Optional path for persisting workflow state.",
)
def master_start(host, port, state_path):
    """Start the master service."""
    console.print(
        f"[bold blue]Starting master service on {host}:{port}[/bold blue]\n"
        "[dim]Press CTRL+C to stop.[/dim]"
    )
    app = create_app(state_path=state_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.group()
def worker():
    """Manage worker agents."""


@worker.command("start")
@click.option(
    "--master-url",
    default=DEFAULT_MASTER_URL,
    show_default=True,
    help="URL of the master service to connect to.",
)
@click.option(
    "--worker-id",
    default=None,
    type=str,
    help="Optional worker identifier.",
)
@click.option(
    "--poll-interval",
    default=DEFAULT_POLL_INTERVAL,
    show_default=True,
    help="Seconds between task polling attempts.",
)
@click.option(
    "--capability",
    "capabilities",
    multiple=True,
    help="Optional capability tag for task routing. Provide multiple times as needed.",
)
def worker_start(master_url, worker_id, poll_interval, capabilities):
    """Start a worker agent that executes remote workflows."""
    agent = WorkerAgent(
        master_url=master_url,
        worker_id=worker_id,
        poll_interval=poll_interval,
        capabilities=list(capabilities),
    )
    agent.start()


@main.command()
@click.option(
    "--output",
    "-o",
    default="mlops-config.yaml",
    help="Output path for the configuration file.",
)
def init(output):
    """Initialize a new MLOps project with default configuration."""
    console.print("\n[bold blue]Make MLOps Easy - Initialize Project[/bold blue]\n")

    try:
        from easy_mlops.config import Config

        config = Config()
        config.save(output)

        console.print(
            f"[bold green]✓ Configuration file created:[/bold green] {output}"
        )
        console.print(
            "\n[dim]You can now customize the configuration and use it with:[/dim]"
        )
        console.print(f"[dim]  make-mlops-easy train data.csv -c {output}[/dim]")

    except Exception as e:  # pragma: no cover - runtime error handling
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
