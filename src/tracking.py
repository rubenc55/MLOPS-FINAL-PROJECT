"""Record every pipeline run in MLflow on DagsHub and register the model that passes the quality gate.

Inputs: SETTINGS["tracking"] (mode, tracking_uri, experiment, model_name, prod_alias), the run settings,
metrics from evaluate and the saved model file; MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
(DagsHub user and token) from .env or GitHub Secrets, never in config.yaml.
Outputs: one MLflow run per pipeline run (settings, data hash, git commit, metrics, gate result) and,
only after the gate passes, a new candidate version of the registered model.
Used by: main, which owns the run (start, log, end); scripts/fetch_prod_model.py (deploy), which
downloads the prod model for the release image.
Why: every model can be traced to its run, data version and commit; the gate decides which models
can be candidates and a person decides which one is prod; without credentials the pipeline still runs,
with tracking disabled.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

MODES = ("online", "disabled")


def file_md5(filepath: Path) -> str:
    """Return the md5 hash of a file, the same value DVC stores in its .dvc pointer.

    Inputs: filepath, e.g. the raw training data.
    Outputs: the hex md5 string.
    Why: the run records which data version trained the model, matching data/raw/shipments.csv.dvc.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}. Get the data first: dvc pull")
    digest = hashlib.md5()
    with filepath.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the commit of the code that runs, or "unknown" outside a Git checkout.

    Inputs: none (GITHUB_SHA in CI, otherwise git rev-parse HEAD).
    Outputs: the commit hash, or "unknown".
    Why: a model in the registry points back to the exact code that trained it.
    """
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Git commit not found; recording 'unknown'")
        return "unknown"
    return result.stdout.strip()


def _connect(tracking: dict) -> None:
    """Point MLflow runs and the model registry at the tracking server in config.yaml."""
    mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_registry_uri(tracking["tracking_uri"])


def _flatten(settings: dict, prefix: str = "") -> dict:
    """Turn nested settings into flat MLflow parameters, e.g. {"split": {"seed": 42}} -> {"split.seed": 42}."""
    flat = {}
    for key, value in settings.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def start_run(tracking: dict, run_config: dict):
    """Open an MLflow run for one pipeline run and record its settings and lineage.

    Inputs: tracking (SETTINGS["tracking"]), run_config (settings plus data_md5 and git_commit).
    Outputs: the active MLflow run, or None when tracking is disabled.
    Why: online without MLFLOW_TRACKING_PASSWORD would stop the pipeline, so it runs disabled with a warning.
    """
    mode = tracking["mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown tracking mode: {mode!r}. Use one of {MODES} in config.yaml (tracking.mode).")
    if mode == "online" and not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        logger.warning("MLFLOW_TRACKING_PASSWORD is not set, so MLflow tracking is disabled for this run "
                       "(see .env.example)")
        mode = "disabled"
    if mode == "disabled":
        return None
    _connect(tracking)
    mlflow.set_experiment(tracking["experiment"])
    run = mlflow.start_run()
    lineage = {name: str(run_config[name]) for name in ("data_md5", "git_commit") if name in run_config}
    mlflow.set_tags(lineage)
    mlflow.log_params(_flatten({k: v for k, v in run_config.items() if k not in ("paths", *lineage)}))
    mlflow.log_dict(run_config, "run_config.json")
    logger.info("MLflow run started: %s (experiment %s)", run.info.run_id, tracking["experiment"])
    return run


def log_metrics(run, metrics: dict[str, float], split: str) -> None:
    """Log metrics of one split with the split as prefix, e.g. val_mape; nothing when tracking is off.

    Inputs: the MLflow run (or None), metrics from evaluate_model or evaluate_baseline, split name.
    Outputs: None; the values appear on the run page.
    Why: validation, baseline and test metrics stay apart, next to the gate result.
    """
    if run is not None:
        mlflow.log_metrics({f"{split}_{name}": value for name, value in metrics.items()})


def log_model(run, model_path: Path, tracking: dict, metadata: dict) -> None:
    """Upload the model that passed the gate and register it as a new candidate version.

    Inputs: the MLflow run (or None), model_path (saved by save_model), tracking settings, metadata
    (data hash, commit and metrics).
    Outputs: None; a new version of tracking["model_name"]. The prod alias does not move.
    Why: main calls this only after the gate passes, so every version met it; a person compares the
    runs and moves prod (README section 4), so a training run never changes what the next release ships.
    """
    if run is None:
        logger.warning("MLflow tracking is disabled, so the model is not registered (saved only at %s)", model_path)
        return
    mlflow.log_artifact(str(model_path), artifact_path="model")
    mlflow.log_dict(metadata, "model/model_metadata.json")
    client, name = MlflowClient(), tracking["model_name"]
    try:
        client.get_registered_model(name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        client.create_registered_model(name, description="Carrier cost in USD per shipment (README section 11)")
    tags = {key: str(value) for key, value in metadata.items() if key in ("data_md5", "git_commit")}
    version = client.create_model_version(name, f"{run.info.artifact_uri}/model", run_id=run.info.run_id, tags=tags)
    logger.info("Model registered as %s version %s (candidate). To ship it, compare the runs and move %s "
                "to this version (README section 4)", name, version.version, tracking["prod_alias"])


def fetch_prod_model(tracking: dict, model_path: Path, info_path: Path) -> dict:
    """Download the model aliased prod from the registry and write its lineage next to it.

    Inputs: tracking settings, model_path (where the API loads the model), info_path (lineage JSON).
    Outputs: the lineage dict (model version, MLflow run, git commit, data md5, metrics), also
    written to info_path; the model file is replaced.
    Why: the release image serves the registered prod model, and /health shows where it came from.
    """
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError("MLFLOW_TRACKING_PASSWORD is not set, so the prod model cannot be downloaded "
                           "(see .env.example)")
    _connect(tracking)
    client, name, alias = MlflowClient(), tracking["model_name"], tracking["prod_alias"]
    version = client.get_model_version_by_alias(name, alias)
    model_path, info_path = Path(model_path), Path(info_path)
    with tempfile.TemporaryDirectory() as download_dir:
        folder = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}@{alias}", dst_path=download_dir
        ))
        downloaded = folder / model_path.name
        if not downloaded.is_file():
            raise FileNotFoundError(f"{name}@{alias} has no file {model_path.name}; it was not logged by log_model")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, model_path)
    run = client.get_run(version.run_id)
    info = {
        "model": f"{name}@{alias}", "version": str(version.version), "run_id": version.run_id,
        "git_commit": run.data.tags.get("git_commit"), "data_md5": run.data.tags.get("data_md5"),
        "metrics": run.data.metrics,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Downloaded %s version %s to %s", name, version.version, model_path)
    return info


def end_run(run, error: Exception | None = None) -> None:
    """Close the MLflow run as finished, or as failed with the error message; nothing when tracking is off.

    Inputs: the MLflow run (or None), error (None when the pipeline succeeded).
    Outputs: None.
    Why: a failed gate or bad data shows up in MLflow as a failed run with its reason.
    """
    if run is None:
        return
    if error is not None:
        mlflow.set_tag("error", str(error)[:5000])
    mlflow.end_run(status="FINISHED" if error is None else "FAILED")
