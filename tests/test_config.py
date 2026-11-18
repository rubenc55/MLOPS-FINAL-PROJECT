"""Tests for src/config.py: loading config.yaml into SETTINGS, and dvc.yaml agreeing with it."""

import pytest
import yaml

from src.config import CONFIG_PATH, PROJECT_ROOT, SETTINGS, load_config


def test_unit_load_config_resolves_paths_against_project_root(tmp_path):
    """Relative paths in the YAML file become absolute paths under the repo root; other values stay as written."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  model: models/model.joblib\nquality_gate:\n  mape: 0.18\n", encoding="utf-8")

    settings = load_config(config_path)

    assert settings["paths"]["model"] == PROJECT_ROOT / "models" / "model.joblib"
    assert settings["quality_gate"] == {"mape": 0.18}


def test_unit_load_config_missing_file_raises(tmp_path):
    """A config file that does not exist stops the run with a FileNotFoundError that names the file."""
    with pytest.raises(FileNotFoundError, match="missing_config.yaml"):
        load_config(tmp_path / "missing_config.yaml")


@pytest.mark.parametrize("content", ["", "- just\n- a list\n", "quality_gate:\n  mape: 0.18\n"])
def test_unit_load_config_without_paths_section_raises(tmp_path, content):
    """An empty file, a non-mapping or a file without 'paths' stops with a clear ValueError."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="'paths' section"):
        load_config(config_path)


def test_integration_project_config_has_every_setting_the_pipeline_reads():
    """The repo's config.yaml loads every section and path main uses, with the agreed gate mape <= 0.18."""
    settings = load_config(CONFIG_PATH)

    assert set(settings) == {
        "paths", "logging", "problem", "features", "validation", "split", "model_params", "quality_gate", "tracking"
    }
    assert set(settings["paths"]) == {
        "raw_data", "inference_data", "clean_data", "model", "predictions", "log_file"
    }
    assert all(path.is_absolute() for path in settings["paths"].values())
    assert settings["paths"]["log_file"] == PROJECT_ROOT / "logs" / "pipeline.log"
    assert settings["quality_gate"] == {"mape": 0.18}
    assert settings["tracking"]["prod_alias"] == "prod"
    assert settings == SETTINGS


def test_integration_dvc_stage_matches_config_paths():
    """dvc.yaml runs python -m src.main with the config.yaml inputs and outputs, so dvc.lock tracks the real files."""
    stage = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]["pipeline"]
    paths = {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in SETTINGS["paths"].items()}
    outs = [out if isinstance(out, str) else next(iter(out)) for out in stage["outs"]]

    assert stage["cmd"] == "python -m src.main"
    assert {"src", paths["raw_data"], paths["inference_data"]} == set(stage["deps"])
    assert stage["params"] == [{"config.yaml": None}]
    assert set(outs) == {paths["clean_data"], paths["model"], paths["predictions"], paths["log_file"]}
