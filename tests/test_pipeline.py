from pathlib import Path

from easy_mlops.pipeline import MLOpsPipeline


def test_pipeline_end_to_end(
    pipeline_training_csv, pipeline_prediction_csv, pipeline_config_file
):
    pipeline = MLOpsPipeline(config_path=str(pipeline_config_file))

    results = pipeline.run(
        str(pipeline_training_csv),
        target_column="target",
        deploy=True,
    )

    assert "training" in results
    assert "deployment" in results
    assert results["preprocessing"]["n_samples"] == 90
    assert results["logs"]["metrics"].endswith("metrics_history.json")

    deployment_dir = Path(results["deployment"]["deployment_dir"])
    assert deployment_dir.exists()
    assert (deployment_dir / "model.joblib").exists()

    predictions = pipeline.predict(
        str(pipeline_prediction_csv),
        results["deployment"]["deployment_dir"],
    )
    assert len(predictions) == 30

    status = pipeline.get_status(results["deployment"]["deployment_dir"])
    assert status["metadata"]["status"] == "deployed"
    assert "metrics" in status

    report = pipeline.observe(results["deployment"]["deployment_dir"])
    assert "MODEL MONITORING REPORT" in report

    log_dir = deployment_dir / "logs"
    assert (log_dir / "predictions_log.json").exists()
    assert (log_dir / "metrics_history.json").exists()


def test_pipeline_run_without_deployment(pipeline_training_csv, pipeline_config_file):
    pipeline = MLOpsPipeline(config_path=str(pipeline_config_file))
    results = pipeline.run(
        str(pipeline_training_csv),
        target_column="target",
        deploy=False,
    )

    assert "deployment" not in results
    assert "logs" not in results
    assert results["training"]["model_type"] == "classification"
