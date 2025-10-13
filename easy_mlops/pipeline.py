"""Main MLOps pipeline orchestrator."""

from typing import Dict, Any, Optional
from pathlib import Path

from easy_mlops.config import Config
from easy_mlops.preprocessing import DataPreprocessor
from easy_mlops.training import ModelTrainer
from easy_mlops.deployment import ModelDeployer
from easy_mlops.observability import ModelMonitor


class MLOpsPipeline:
    """Main pipeline orchestrator for Make MLOps Easy."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MLOps pipeline.

        Args:
            config_path: Path to configuration file. If None, uses defaults.
        """
        self.config = Config(config_path)
        self.preprocessor = None
        self.trainer = None
        self.deployer = None
        self.monitor = None
        self.results = {}

    def run(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        deploy: bool = True,
    ) -> Dict[str, Any]:
        """Run complete MLOps pipeline.

        Args:
            data_path: Path to training data.
            target_column: Name of target column. If None, assumes last column.
            deploy: Whether to deploy the model after training.

        Returns:
            Dictionary with pipeline results.
        """
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config.get("preprocessing"))
        self.trainer = ModelTrainer(self.config.get("training"))
        self.deployer = ModelDeployer(self.config.get("deployment"))
        self.monitor = ModelMonitor(self.config.get("observability"))

        # Step 1: Data Preprocessing
        print("Step 1/4: Data Preprocessing...")
        X, y = self.preprocessor.preprocess(data_path, target_column)
        print(f"  ✓ Processed {X.shape[0]} samples with {X.shape[1]} features")

        # Step 2: Model Training
        print("\nStep 2/4: Model Training...")
        training_results = self.trainer.train(X, y)
        print(f"  ✓ Trained {training_results['model_name']}")
        print(f"  ✓ Problem type: {training_results['model_type']}")
        print("  ✓ Metrics:")
        for metric, value in training_results["metrics"].items():
            print(f"      {metric}: {value:.4f}")

        # Log metrics
        self.monitor.log_metrics(training_results["metrics"])

        # Step 3: Model Deployment
        if deploy:
            print("\nStep 3/4: Model Deployment...")
            deployment_info = self.deployer.deploy(
                self.trainer, self.preprocessor, training_results
            )
            print(f"  ✓ Model deployed to: {deployment_info['deployment_dir']}")
            print(f"  ✓ Model path: {deployment_info['model_path']}")
            print(f"  ✓ Preprocessor path: {deployment_info['preprocessor_path']}")

            if "endpoint_path" in deployment_info:
                print(f"  ✓ Endpoint script: {deployment_info['endpoint_path']}")

            self.results["deployment"] = deployment_info
        else:
            print("\nStep 3/4: Model Deployment... SKIPPED")

        # Step 4: Setup Observability
        print("\nStep 4/4: Setup Observability...")
        print("  ✓ Monitoring configured")
        print("  ✓ Metrics logging enabled")

        # Save monitoring logs if deployment happened
        if deploy and "deployment" in self.results:
            log_dir = Path(self.results["deployment"]["deployment_dir"]) / "logs"
            log_files = self.monitor.save_logs(str(log_dir))
            print(f"  ✓ Logs saved to: {log_dir}")
            self.results["logs"] = log_files

        # Compile results
        self.results.update(
            {
                "training": training_results,
                "preprocessing": {
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                    "feature_columns": self.preprocessor.feature_columns,
                },
            }
        )

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)

        return self.results

    def predict(
        self, data_path: str, model_dir: str, log_predictions: bool = True
    ) -> Any:
        """Make predictions using a deployed model.

        Args:
            data_path: Path to data for prediction.
            model_dir: Path to deployment directory.
            log_predictions: Whether to log predictions.

        Returns:
            Predictions array.
        """
        # Initialize deployer and load model
        deployer = ModelDeployer(self.config.get("deployment"))
        model_data, preprocessor, metadata = deployer.load_deployed_model(model_dir)

        # Load and preprocess data
        df = preprocessor.load_data(data_path)

        # Remove target column if it exists in the data
        if preprocessor.target_column and preprocessor.target_column in df.columns:
            df = df.drop(columns=[preprocessor.target_column])

        X, _ = preprocessor.prepare_data(df, target_column=None, fit=False)

        # Make predictions
        predictions = model_data["model"].predict(X)

        # Log predictions if monitoring is enabled
        if log_predictions:
            monitor = ModelMonitor(self.config.get("observability"))
            for pred in predictions:
                monitor.log_prediction(
                    input_data=None,
                    prediction=pred,
                    model_version=metadata.get("version", "1.0.0"),
                )

            # Save logs
            log_dir = Path(model_dir) / "logs"
            monitor.save_logs(str(log_dir))

        return predictions

    def get_status(self, model_dir: str) -> Dict[str, Any]:
        """Get status and information about a deployed model.

        Args:
            model_dir: Path to deployment directory.

        Returns:
            Dictionary with model status and metrics.
        """
        deployer = ModelDeployer(self.config.get("deployment"))
        model_data, preprocessor, metadata = deployer.load_deployed_model(model_dir)

        # Load monitoring logs
        monitor = ModelMonitor(self.config.get("observability"))
        log_dir = Path(model_dir) / "logs"
        if log_dir.exists():
            monitor.load_logs(str(log_dir))

        return {
            "metadata": metadata,
            "model_type": model_data.get("model_type"),
            "metrics": model_data.get("metrics", {}),
            "monitoring": {
                "metrics_summary": monitor.get_metrics_summary(),
                "predictions_summary": monitor.get_predictions_summary(),
            },
        }

    def observe(self, model_dir: str) -> str:
        """Generate observability report for a deployed model.

        Args:
            model_dir: Path to deployment directory.

        Returns:
            Formatted monitoring report.
        """
        monitor = ModelMonitor(self.config.get("observability"))
        log_dir = Path(model_dir) / "logs"

        if log_dir.exists():
            monitor.load_logs(str(log_dir))

        return monitor.generate_report()
