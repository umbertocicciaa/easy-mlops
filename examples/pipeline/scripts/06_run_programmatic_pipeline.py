#!/usr/bin/env python3
"""Programmatic pipeline example.

This script mirrors the CLI but drives the pipeline directly from Python so you
can integrate Make MLOps Easy into notebooks or bespoke automation.
"""

from __future__ import annotations

import json
from pathlib import Path

from easy_mlops.pipeline import MLOpsPipeline


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "examples" / "data" / "house_prices.csv"
    config_path = repo_root / "examples" / "pipeline" / "configs" / "regression_neural_network.yaml"

    print("[python] Loading configuration:", config_path)
    pipeline = MLOpsPipeline(config_path=str(config_path))

    print("[python] Running training pipeline…")
    results = pipeline.run(
        data_path=str(data_path),
        target_column="price",
        deploy=True,
    )

    deployment = results.get("deployment", {})
    deployment_dir = deployment.get("deployment_dir")
    if deployment_dir:
        print(f"[python] Model deployed to: {deployment_dir}")

        print("[python] Generating predictions with the deployed artifacts…")
        predictions = pipeline.predict(
            data_path=str(data_path),
            model_dir=deployment_dir,
        )

        sample = predictions[:5].tolist()
        print("[python] Sample predictions:", json.dumps(sample, indent=2))
    else:
        print("[python] Deployment skipped; no predictions were generated.")


if __name__ == "__main__":
    main()
