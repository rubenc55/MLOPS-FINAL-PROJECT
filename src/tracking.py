"""Record every pipeline run in W&B and register the model that passes the quality gate.

Inputs: SETTINGS["tracking"] (mode, project, entity, model_name, prod_alias), the run settings,
metrics from evaluate and the saved model file; WANDB_API_KEY from .env (never in config.yaml).
Outputs: one W&B run per pipeline run (settings, data hash, git commit, metrics, gate result) and,
only after the gate passes, a model artifact version with the prod alias.
Used by: main, which owns the run (start, log, end); the API and deploy later load the prod model.
Why: every model can be traced to its run, data version and commit, and the gate decides which one
is prod; with no API key the pipeline still runs, with tracking disabled.
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

MODES = ("online", "offline", "disabled")


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


def start_run(tracking: dict, run_config: dict):
    """Open a W&B run for one pipeline run.

    Inputs: tracking (SETTINGS["tracking"]), run_config (settings, data hash and commit to record).
    Outputs: the W&B run; a no-op run when the mode is disabled.
    Why: online without WANDB_API_KEY would stop the pipeline, so it falls back to disabled with a warning.
    """
    mode = tracking["mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown tracking mode: {mode!r}. Use one of {MODES} in config.yaml (tracking.mode).")
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set, so W&B tracking is disabled for this run (see .env.example)")
        mode = "disabled"
    run = wandb.init(
        project=tracking["project"], entity=tracking["entity"], job_type="train", config=run_config, mode=mode
    )
    logger.info("W&B run started (mode %s): %s", mode, run.url or run.id)
    return run


def log_metrics(run, metrics: dict[str, float], split: str) -> None:
    """Log metrics of one split with the split as prefix, e.g. val_mape.

    Inputs: the W&B run, metrics from evaluate_model, split ("val" or "test").
    Outputs: None; the values appear in the run summary.
    Why: validation and test metrics stay apart, and the run page shows them next to the gate result.
    """
    run.log({f"{split}_{name}": value for name, value in metrics.items()})


def log_model(run, model_path: Path, tracking: dict, metadata: dict) -> None:
    """Upload the released model as a new artifact version and move the prod alias to it.

    Inputs: the W&B run, model_path (saved by save_model), tracking settings, metadata (metrics,
    data hash, commit).
    Outputs: None; the artifact is uploaded with the aliases latest and prod.
    Why: main calls this only after the gate passes, so prod always points to a model that met it.
    """
    if run.disabled:
        logger.warning("W&B tracking is disabled, so the model is not registered (saved only at %s)", model_path)
        return
    artifact = wandb.Artifact(tracking["model_name"], type="model", metadata=metadata)
    artifact.add_file(str(model_path))
    run.log_artifact(artifact, aliases=["latest", tracking["prod_alias"]])
    logger.info("Model logged to W&B as %s:%s", tracking["model_name"], tracking["prod_alias"])


def end_run(run, error: Exception | None = None) -> None:
    """Close the W&B run as finished, or as failed with the error message.

    Inputs: the W&B run, error (None when the pipeline succeeded).
    Outputs: None.
    Why: a failed gate or bad data shows up in W&B as a failed run with its reason.
    """
    if error is not None:
        run.summary["error"] = str(error)
    run.finish(exit_code=0 if error is None else 1)
