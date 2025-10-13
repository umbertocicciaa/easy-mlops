import yaml

from easy_mlops.config import Config


def test_config_provides_defaults():
    cfg = Config()
    assert cfg.get("training", "test_size") == 0.2
    assert cfg.get("preprocessing")["encode_categorical"] is True


def test_config_merges_user_overrides(tmp_path):
    custom_cfg = {
        "training": {"test_size": 0.4},
        "observability": {"alert_threshold": 0.4},
    }
    cfg_path = tmp_path / "custom.yaml"
    with open(cfg_path, "w") as fp:
        yaml.safe_dump(custom_cfg, fp)

    cfg = Config(str(cfg_path))
    assert cfg.get("training", "test_size") == 0.4
    # Ensure untouched defaults remain
    assert cfg.get("deployment", "output_dir") == "./models"
    assert cfg.get("observability", "alert_threshold") == 0.4


def test_config_get_with_missing_returns_default():
    cfg = Config()
    assert cfg.get("nonexistent", default="fallback") == "fallback"
    assert cfg.get("training", "missing", default=123) == 123


def test_config_save_roundtrip(tmp_path):
    cfg = Config()
    cfg.config["training"]["test_size"] = 0.33

    out_path = tmp_path / "saved.yaml"
    cfg.save(str(out_path))

    loaded = Config(str(out_path))
    assert loaded.get("training", "test_size") == 0.33
