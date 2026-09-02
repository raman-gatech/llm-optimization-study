from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "Model_Optimizations" / "configs"


def test_all_experiment_configs_are_nonempty_and_portable() -> None:
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(config_paths) == 8

    for path in config_paths:
        config = yaml.safe_load(path.read_text())
        assert isinstance(config, dict), f"empty or invalid config: {path.name}"

        output_dir = Path(config["output_dir"])
        assert not output_dir.is_absolute(), f"absolute output path in {path.name}"
        assert output_dir.parts[0] == "outputs"

        assert config["seed"] >= 0
        assert config["max_length"] > 0
        assert config["max_steps"] > 0
        assert 0 < config["val_ratio"] < 1
        assert config["save_every"] > 0

        if "method" in config:
            assert config["student_model_name"]
            assert config["teacher_model_name"]
            assert 0 <= config["kd_alpha"] <= 1
            assert config["kd_temperature"] > 0
        else:
            assert config["model_name"]

        if config.get("method") == "method2_topk_kd":
            assert config["top_k"] > 0
