"""Load config.yaml once into SETTINGS, the single source of every non-secret setting.

Inputs: config.yaml at the repo root (paths, problem, features, validation, split, model_params,
quality_gate, logging).
Outputs: SETTINGS, a dict with the same sections; every entry of "paths" is an absolute Path.
Used by: main (the pipeline) and, in stage 3, the API.
Why: one loader for every entry point, so the pipeline and the API always read the same settings.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Read the YAML settings file and turn its relative paths into absolute ones.

    Inputs: config_path, the YAML file (config.yaml at the repo root).
    Outputs: the settings dict with the same sections as the file; every entry of "paths" becomes
    an absolute Path under PROJECT_ROOT.
    Why: one source of settings for every run; paths relative to the repo root work from any folder.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Create config.yaml at the repo root."
        )
    with config_path.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict) or "paths" not in settings:
        raise ValueError(
            f"Config file {config_path} must be a YAML mapping with a 'paths' section."
        )
    settings["paths"] = {name: PROJECT_ROOT / path for name, path in settings["paths"].items()}
    return settings


SETTINGS = load_config(CONFIG_PATH)
