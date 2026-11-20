"""Run the whole shipping cost pipeline with one command: python -m src.main

Inputs: SETTINGS from src/config.py (config.yaml: paths, columns, rules, split, model parameters,
quality gate, logging, tracking); WANDB_API_KEY from .env.
Outputs: the clean training table, the model (only if the quality gate passes), the
predictions and the run log (logs/pipeline.log), written to the paths set in config.yaml;
one W&B run per pipeline run and, after the gate passes, the model registered as prod.
Used by: the team, CI and the stage 3 API;
it is the only place where the modules are wired together.
Why: one entry point and one source of settings, so every run follows the same steps.
"""

import logging

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.config import PROJECT_ROOT, SETTINGS
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logger import setup_logging
from src.tracking import end_run, file_md5, git_commit, log_metrics, log_model, start_run
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)


# Six arguments is the course contract for split_data, so its signature must not change.
# X keeps the contract's uppercase name (X is the feature matrix, as in scikit-learn).
def split_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    X: pd.DataFrame, y: pd.Series,  # pylint: disable=invalid-name
    test_size: float, val_size: float, random_state: int, stratify: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split features and target into train, validation and test parts, before anything is fitted.

    Inputs: X (feature columns) and y (target) of the clean, validated table; test and validation
    shares of the WHOLE table; seed; stratify (True keeps the class shares of y in every part).
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test) with no shared rows.
    Why: the model learns only from train, the gate uses validation, and test stays sealed.
    """
    logger.info("Splitting %d rows: test %s, validation %s", len(X), test_size, val_size)
    x_rest, x_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    # val_size is a share of the whole table, so rescale it to the rows left after removing test
    val_share = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=val_share, random_state=random_state,
        stratify=y_rest if stratify else None,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def check_quality_gate(metrics: dict[str, float], quality_gate: dict[str, float]) -> None:
    """Stop the release if any validation metric is above its agreed limit (lower is better).

    Inputs: validation metrics from evaluate_model; limits from SETTINGS["quality_gate"].
    Outputs: None if every metric is at or below its limit; otherwise raises ValueError.
    Why: a model that misses the business case (README) must never be saved or used for quotes.
    """
    logger.info("Checking quality gate %s against validation metrics %s", quality_gate, metrics)
    missing = [name for name in quality_gate if name not in metrics]
    if missing:
        raise ValueError(
            f"Metrics missing for the quality gate: {missing}. Check evaluate_model output keys"
        )
    failures = [
        f"{name} = {metrics[name]:.4f} > {limit}"
        for name, limit in quality_gate.items()
        if metrics[name] > limit
    ]
    if failures:
        message = f"Model not released, quality gate failed: {'; '.join(failures)}"
        logger.error(message)  # keep the decision in logs/pipeline.log, not only in the traceback
        raise ValueError(message)
    logger.info("Quality gate met: every validation metric is within its limit")


def _train_and_release(settings: dict, run) -> tuple:
    """Train on the train split, release the model only if the validation gate passes, then save it.

    Inputs: SETTINGS (paths, problem, features, validation, split, model_params, quality_gate)
    and the W&B run, which receives the validation and test metrics.
    Outputs: (fitted Pipeline, validation metrics, test metrics).
    Why: one place wires load, clean, validate, save the clean table, split, recipe, fit, gate
    and save, in the agreed order.
    """
    problem, features = settings["problem"], settings["features"]
    feature_columns = features["categorical"] + features["numeric"]
    target = problem["target_column"]

    # 1. Load, clean and validate the training table (target required and > 0), then save it
    df = clean_dataframe(load_raw_data(settings["paths"]["raw_data"]), target_column=target)
    validate_dataframe(
        df, [problem["id_column"]] + feature_columns, target_column=target, **settings["validation"]
    )
    save_csv(df, settings["paths"]["clean_data"])

    # 2. Split before anything is fitted; the test part stays sealed
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        df[feature_columns], df[target], **settings["split"]
    )

    # 3. Build the unfitted recipe and fit it, with the model, on the train split only
    model = train_model(
        x_train, y_train, get_feature_preprocessor(features["categorical"], features["numeric"]),
        problem["problem_type"],
        model_params=settings["model_params"], random_state=settings["split"]["random_state"],
    )

    # 4. Quality gate on validation: metrics go to W&B first, so a failed run still shows them;
    #    if the gate fails, the error stops here and nothing is saved
    val_metrics = evaluate_model(model, x_val, y_val, problem["problem_type"])
    log_metrics(run, val_metrics, "val")
    check_quality_gate(val_metrics, settings["quality_gate"])

    # 5. Test is scored once, after the release decision, only to report it
    test_metrics = evaluate_model(model, x_test, y_test, problem["problem_type"])
    log_metrics(run, test_metrics, "test")
    logger.info(
        "Validation metrics: %s | Test metrics (report only): %s", val_metrics, test_metrics
    )

    save_model(model, settings["paths"]["model"])
    return model, val_metrics, test_metrics


def _predict_new_shipments(settings: dict, model) -> pd.DataFrame:
    """Predict the cost of each new shipment and save it next to its ID.

    Inputs: SETTINGS (paths, problem, features, validation) and the released Pipeline.
    Outputs: DataFrame with the ID column and "prediction",
    also saved to SETTINGS["paths"]["predictions"].
    Why: same cleaning, rules and recipe as training (no skew);
    the ID keeps every estimate traceable.
    """
    paths, problem, features = settings["paths"], settings["problem"], settings["features"]
    feature_columns = features["categorical"] + features["numeric"]
    id_column = problem["id_column"]

    # 1. Same cleaning and rules as training, without the target; one bad shipment stops the batch
    df = clean_dataframe(load_raw_data(paths["inference_data"]))
    validate_dataframe(df, [id_column] + feature_columns, **settings["validation"])

    # 2. Predict from the feature columns only, then put the ID back as the first column
    predictions = run_inference(model, df[feature_columns])
    output = pd.concat([df[[id_column]], predictions], axis=1)

    save_csv(output, paths["predictions"])
    logger.info("Saved %d predictions to %s", len(output), paths["predictions"])
    return output


def _remove_previous_outputs(paths: dict) -> None:
    """Delete the model and predictions left by an earlier run, before anything is trained.

    Inputs: SETTINGS["paths"] (uses "model" and "predictions").
    Outputs: None; the files are removed if they exist.
    Why: if this run fails the gate, no older model or predictions stay on disk looking like its result;
    released models are kept in the model registry, not in models/.
    """
    for name in ("model", "predictions"):
        if paths[name].exists():
            logger.info("Removing %s from an earlier run: %s", name, paths[name])
            paths[name].unlink()


def run_pipeline(settings: dict) -> dict[str, float]:
    """Run training, the release gate and batch inference with the given settings.

    Inputs: a settings dict shaped like SETTINGS (tests pass a copy with temporary paths).
    Outputs: validation metrics of the released model.
    Why: settings as a parameter lets tests run the whole pipeline without touching real folders;
    main owns the W&B run: it records the data hash and commit, registers the model as prod only
    after the gate and the predictions succeed, and closes the run as failed on any error.
    """
    logger.info("Starting the shipping cost pipeline")
    _remove_previous_outputs(settings["paths"])
    lineage = {"data_md5": file_md5(settings["paths"]["raw_data"]), "git_commit": git_commit()}
    run_config = {**settings, "paths": {name: str(path) for name, path in settings["paths"].items()}, **lineage}
    run = start_run(settings["tracking"], run_config)
    try:
        model, val_metrics, test_metrics = _train_and_release(settings, run)
        _predict_new_shipments(settings, model)
        log_model(
            run, settings["paths"]["model"], settings["tracking"],
            metadata={**lineage, "val_metrics": val_metrics, "test_metrics": test_metrics},
        )
    except Exception as error:
        end_run(run, error)
        raise
    end_run(run)
    logger.info(
        "Done. Model: %s | Predictions: %s",
        settings["paths"]["model"], settings["paths"]["predictions"],
    )
    return val_metrics


def main() -> None:
    """Entry point for python -m src.main: run the pipeline with the project SETTINGS.

    Inputs: none (uses SETTINGS; secrets such as WANDB_API_KEY from .env).
    Outputs: model, predictions and log files on disk; the W&B run.
    Why: one command for the team, CI and later the stage 3 API.
    """
    # Secrets come from .env (ignored by Git); variables already set, as in CI, are kept
    load_dotenv(PROJECT_ROOT / ".env")
    # Only the entry point configures logging; every module just writes to its own logger
    setup_logging(SETTINGS["paths"]["log_file"], **SETTINGS["logging"])
    logger.info("Running the pipeline with the project SETTINGS")
    run_pipeline(SETTINGS)


if __name__ == "__main__":  # pragma: no cover
    main()
