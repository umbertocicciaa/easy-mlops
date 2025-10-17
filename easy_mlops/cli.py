"""Command-line interface for Make MLOps Easy."""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from easy_mlops.pipeline import MLOpsPipeline

console = Console()


@click.group()
@click.version_option(version="0.3.0")
def main():
    """Make MLOps Easy - Make MLOps easier with automated pipelines.

    A comprehensive framework that abstracts data preprocessing, model training,
    deployment, and observability into simple CLI commands.
    """
    pass


@main.command()
@click.argument("data-path", type=click.Path(exists=True))
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
    type=click.Path(exists=True),
    help="Path to configuration file (YAML).",
)
@click.option(
    "--no-deploy",
    is_flag=True,
    help="Skip deployment step (only train the model).",
)
def train(data_path, target, config, no_deploy):
    """Train a machine learning model on your data.

    This command runs the complete MLOps pipeline:
    - Data preprocessing (handling missing values, encoding, scaling)
    - Model training (automatic model selection)
    - Model deployment (saving model artifacts)
    - Observability setup (metrics logging)

    Example:
        make-mlops-easy train data.csv --target price
        make-mlops-easy train data.json -t label -c config.yaml
    """
    console.print("\n[bold blue]Make MLOps Easy - Training Pipeline[/bold blue]\n")

    try:
        pipeline = MLOpsPipeline(config_path=config)
        results = pipeline.run(
            data_path=data_path,
            target_column=target,
            deploy=not no_deploy,
        )

        # Display results in a nice format
        console.print("\n[bold green]✓ Training completed successfully![/bold green]\n")

        if "deployment" in results:
            console.print("[bold]Deployment Information:[/bold]")
            console.print(f"  Location: {results['deployment']['deployment_dir']}")
            console.print(f"  Model: {results['deployment']['model_path']}")

            if "endpoint_path" in results["deployment"]:
                console.print(f"  Endpoint: {results['deployment']['endpoint_path']}")
                console.print("\n[dim]To make predictions, use:[/dim]")
                console.print(
                    f"[dim]  python {results['deployment']['endpoint_path']} <data_path>[/dim]"
                )

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


@main.command()
@click.argument("data-path", type=click.Path(exists=True))
@click.argument("model-dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="Path to configuration file (YAML).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file for predictions (JSON format).",
)
def predict(data_path, model_dir, config, output):
    """Make predictions using a deployed model.

    Load a previously deployed model and make predictions on new data.

    Example:
        make-mlops-easy predict new_data.csv models/deployment_20240101_120000
        make-mlops-easy predict data.json models/latest -o predictions.json
    """
    console.print("\n[bold blue]Make MLOps Easy - Prediction[/bold blue]\n")

    try:
        pipeline = MLOpsPipeline(config_path=config)
        predictions = pipeline.predict(data_path, model_dir)

        console.print(f"[bold green]✓ Predictions completed![/bold green]")
        console.print(f"  Total predictions: {len(predictions)}\n")

        # Display first few predictions
        console.print("[bold]Sample predictions:[/bold]")
        for i, pred in enumerate(predictions[:10]):
            console.print(f"  [{i}]: {pred}")

        if len(predictions) > 10:
            console.print(f"  ... and {len(predictions) - 10} more")

        # Save to file if requested
        if output:
            import json

            with open(output, "w") as f:
                json.dump({"predictions": predictions.tolist()}, f, indent=2)
            console.print(f"\n[bold]Predictions saved to:[/bold] {output}")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


@main.command()
@click.argument("model-dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="Path to configuration file (YAML).",
)
def status(model_dir, config):
    """Get status and metrics for a deployed model.

    Display information about a deployed model including metrics,
    deployment information, and monitoring statistics.

    Example:
        make-mlops-easy status models/deployment_20240101_120000
    """
    console.print("\n[bold blue]Make MLOps Easy - Model Status[/bold blue]\n")

    try:
        pipeline = MLOpsPipeline(config_path=config)
        status_info = pipeline.get_status(model_dir)

        # Display metadata
        console.print("[bold]Model Information:[/bold]")
        metadata = status_info.get("metadata", {})
        console.print(f"  Version: {metadata.get('version', 'N/A')}")
        console.print(f"  Status: {metadata.get('status', 'N/A')}")
        console.print(f"  Deployed: {metadata.get('deployment_time', 'N/A')}")
        console.print(f"  Type: {status_info.get('model_type', 'N/A')}")

        # Display metrics
        console.print("\n[bold]Model Metrics:[/bold]")
        metrics = status_info.get("metrics", {})
        if metrics:
            for metric, value in metrics.items():
                console.print(f"  {metric}: {value:.4f}")
        else:
            console.print("  No metrics available")

        # Display monitoring info
        console.print("\n[bold]Monitoring Statistics:[/bold]")
        monitoring = status_info.get("monitoring", {})

        metrics_summary = monitoring.get("metrics_summary", {})
        if "message" not in metrics_summary:
            console.print(f"  Metric logs: {metrics_summary.get('total_logs', 0)}")

        pred_summary = monitoring.get("predictions_summary", {})
        if "message" not in pred_summary:
            console.print(
                f"  Total predictions: {pred_summary.get('total_predictions', 0)}"
            )

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


@main.command()
@click.argument("model-dir", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="Path to configuration file (YAML).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file for the report.",
)
def observe(model_dir, config, output):
    """Generate observability report for a deployed model.

    Create a detailed monitoring report with metrics history,
    prediction logs, and performance trends.

    Example:
        make-mlops-easy observe models/deployment_20240101_120000
        make-mlops-easy observe models/latest -o report.txt
    """
    console.print("\n[bold blue]Make MLOps Easy - Observability Report[/bold blue]\n")

    try:
        pipeline = MLOpsPipeline(config_path=config)
        report = pipeline.observe(model_dir)

        console.print(report)

        # Save to file if requested
        if output:
            with open(output, "w") as f:
                f.write(report)
            console.print(f"\n[bold]Report saved to:[/bold] {output}")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


@main.command()
@click.option(
    "--output",
    "-o",
    default="mlops-config.yaml",
    help="Output path for the configuration file.",
)
def init(output):
    """Initialize a new MLOps project with default configuration.

    Creates a configuration file with default settings that you can
    customize for your specific use case.

    Example:
        make-mlops-easy init
        make-mlops-easy init -o my-config.yaml
    """
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

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
