"""Configuration management for Make MLOps Easy."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for Make MLOps Easy pipeline."""

    DEFAULT_CONFIG = {
        "preprocessing": {
            "handle_missing": "drop",
            "scale_features": True,
            "encode_categorical": True,
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "model_type": "auto",
        },
        "deployment": {
            "output_dir": "./models",
            "save_format": "joblib",
            "create_endpoint": False,
        },
        "observability": {
            "track_metrics": True,
            "log_predictions": True,
            "alert_threshold": 0.8,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            if user_config:
                self._merge_configs(self.config, user_config)

    def _merge_configs(self, default: Dict, user: Dict) -> None:
        """Recursively merge user config into default config.

        Args:
            default: Default configuration dictionary.
            user: User configuration dictionary.
        """
        for key, value in user.items():
            if (
                key in default
                and isinstance(default[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_configs(default[key], value)
            else:
                default[key] = value

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            section: Configuration section name.
            key: Configuration key within section. If None, returns entire section.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        if section not in self.config:
            return default

        if key is None:
            return self.config[section]

        return self.config[section].get(key, default)

    def save(self, output_path: str) -> None:
        """Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
