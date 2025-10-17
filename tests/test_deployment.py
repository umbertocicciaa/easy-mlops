import os
from pathlib import Path

from easy_mlops.deployment import ModelDeployer, DeploymentStep


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


def test_steps_configuration_customizes_pipeline(
    tmp_path, trained_trainer, fitted_preprocessor
):
    trainer, training_results = trained_trainer
    deployer = ModelDeployer(
        {
            "output_dir": str(tmp_path),
            "steps": [
                {"type": "create_directory", "params": {"prefix": "release"}},
                "save_model",
                "save_preprocessor",
                {"type": "save_metadata", "params": {"filename": "info.json"}},
                {
                    "type": "endpoint_script",
                    "params": {"enabled": True, "filename": "serve.py"},
                },
            ],
        }
    )

    info = deployer.save_deployment_artifacts(
        trainer, fitted_preprocessor, training_results
    )

    assert Path(info["deployment_dir"]).name.startswith("release_")
    assert info["metadata_path"].endswith("info.json")
    assert info["endpoint_path"].endswith("serve.py")
    assert Path(info["endpoint_path"]).exists()


def test_custom_deployment_step_registration(
    tmp_path, trained_trainer, fitted_preprocessor
):
    class MarkerStep(DeploymentStep):
        name = "marker_step"

        def run(self, context):
            context.artifacts["marker"] = "present"

    original_registry = ModelDeployer.STEP_REGISTRY.copy()
    try:
        ModelDeployer.register_step(MarkerStep)
        trainer, training_results = trained_trainer
        deployer = ModelDeployer(
            {
                "output_dir": str(tmp_path),
                "steps": [
                    "create_directory",
                    "save_model",
                    "save_preprocessor",
                    "save_metadata",
                    "marker_step",
                ],
            }
        )

        info = deployer.save_deployment_artifacts(
            trainer, fitted_preprocessor, training_results
        )

        assert info["marker"] == "present"
    finally:
        ModelDeployer.STEP_REGISTRY = original_registry
