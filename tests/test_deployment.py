import os
from pathlib import Path

from easy_mlops.deployment import ModelDeployer


def test_save_deployment_artifacts_creates_expected_files(
    tmp_path, trained_trainer, fitted_preprocessor
):
    trainer, training_results = trained_trainer
    deployer = ModelDeployer({"output_dir": str(tmp_path)})

    info = deployer.save_deployment_artifacts(
        trainer, fitted_preprocessor, training_results
    )

    for key in ["deployment_dir", "model_path", "preprocessor_path", "metadata_path"]:
        assert key in info
        assert Path(info[key]).exists()


def test_deploy_with_endpoint_creates_script(
    tmp_path, trained_trainer, fitted_preprocessor
):
    trainer, training_results = trained_trainer
    deployer = ModelDeployer({"output_dir": str(tmp_path), "create_endpoint": False})

    info = deployer.deploy(
        trainer,
        fitted_preprocessor,
        training_results,
        create_endpoint=True,
    )

    assert "endpoint_path" in info
    endpoint_path = Path(info["endpoint_path"])
    assert endpoint_path.exists()
    assert os.access(endpoint_path, os.X_OK)


def test_load_deployed_model_roundtrip(
    tmp_path, trained_trainer, fitted_preprocessor, classification_data
):
    trainer, training_results = trained_trainer
    deployer = ModelDeployer({"output_dir": str(tmp_path)})
    info = deployer.deploy(trainer, fitted_preprocessor, training_results)

    model_data, loaded_preprocessor, metadata = deployer.load_deployed_model(
        info["deployment_dir"]
    )

    X, _ = classification_data
    preds = model_data["model"].predict(X.head(5))

    assert len(preds) == 5
    assert metadata["status"] == "deployed"
    assert loaded_preprocessor.target_column == "target"
