"""Run the whole shipping cost pipeline with one command: python -m src.main

Inputs: SETTINGS below (paths, columns, rules, split, model parameters, quality gate).
Outputs: models/model.joblib (only if the quality gate passes) and reports/predictions.csv.
Used by: the team, CI and the stage 3 API; it is the only place where the modules are wired together.
Why: one entry point and one source of settings, so every run follows the same steps.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_data
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SETTINGS = {
    "paths": {
        "raw_data": PROJECT_ROOT / "data" / "raw" / "shipments.csv",
        "inference_data": PROJECT_ROOT / "data" / "inference" / "new_shipments.csv",
        "model": PROJECT_ROOT / "models" / "model.joblib",
        "predictions": PROJECT_ROOT / "reports" / "predictions.csv",
    },
    "problem": {"target_column": "cost_usd", "problem_type": "regression", "id_column": "shipment_id"},
    "features": {
        "categorical": ["transport_mode", "cargo_type"],
        "numeric": ["distance_km", "weight_kg", "volume_m3", "urgent", "month"],
    },
    # Keys match validate_dataframe's extra parameters, so main passes them with **settings["validation"]
    "validation": {
        "allowed_values": {
            "transport_mode": ["Road", "Sea", "Air"],
            "cargo_type": ["General", "Refrigerated", "Hazardous"],
            "urgent": [0, 1],
        },
        # Historical min-max (TEAM_BRIEF); business still has to confirm these limits
        "numeric_ranges": {
            "distance_km": (50, 12000),
            "weight_kg": (10, 9900),
            "volume_m3": (0.028, 93.213),
            "month": (1, 12),
        },
    },
    # test_size and val_size are fractions of the whole dataset
    "split": {"test_size": 0.15, "val_size": 0.15, "random_state": 42},
    # No random_state here: train_model receives the seed from SETTINGS["split"]
    "model_params": {"n_estimators": 200, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 5},
    # Lower is better: mape on validation must be <= its limit; mae is reported but not gated (README)
    "quality_gate": {"mape": 0.18},
}


def split_data(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the clean table into train, validation and test parts, before anything is fitted.

    Inputs: clean, validated table; test and validation shares of the WHOLE table; seed.
    Outputs: (train, val, test) DataFrames with no shared rows.
    Why: the model learns only from train, the gate uses validation, and test stays sealed.
    """
    # TODO: replace print with logging (Session 10)
    print(f"Splitting {len(df)} rows: test {test_size}, validation {val_size}")
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    # val_size is a share of the whole table, so rescale it to the rows left after removing test
    val_share = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_share, random_state=random_state)
    return train, val, test


def check_quality_gate(metrics: dict[str, float], quality_gate: dict[str, float]) -> None:
    """Stop the release if any validation metric is above its agreed limit (lower is better).

    Inputs: validation metrics from evaluate_model; limits from SETTINGS["quality_gate"].
    Outputs: None if every metric is at or below its limit; otherwise raises ValueError.
    Why: a model that misses the business case (README) must never be saved or used for quotes.
    """
    # TODO: replace print with logging (Session 10)
    print(f"Checking quality gate {quality_gate} against validation metrics {metrics}")
    missing = [name for name in quality_gate if name not in metrics]
    if missing:
        raise ValueError(f"Metrics missing for the quality gate: {missing}. Check evaluate_model output keys")
    failures = [
        f"{name} = {metrics[name]:.4f} > {limit}" for name, limit in quality_gate.items() if metrics[name] > limit
    ]
    if failures:
        raise ValueError(f"Model not released, quality gate failed: {'; '.join(failures)}")
    print("Quality gate met: every validation metric is within its limit")


def _train_and_release(settings: dict) -> tuple:
    """Train on the train split, release the model only if the validation gate passes, then save it.

    Inputs: SETTINGS (paths, problem, features, validation, split, model_params, quality_gate).
    Outputs: (fitted Pipeline, validation metrics).
    Why: one place wires load, clean, validate, split, recipe, fit, gate and save, in the agreed order.
    """
    paths, problem, features = settings["paths"], settings["problem"], settings["features"]
    feature_columns = features["categorical"] + features["numeric"]
    target = problem["target_column"]

    # 1. Load, clean and validate the training table (target required and > 0)
    df = clean_data(load_raw_data(paths["raw_data"]))
    validate_dataframe(
        df, [problem["id_column"]] + feature_columns, target_column=target, **settings["validation"]
    )

    # 2. Split before anything is fitted; the test part stays sealed
    train, val, test = split_data(df, **settings["split"])

    # 3. Build the unfitted recipe and fit it, with the model, on the train split only
    preprocessor = get_feature_preprocessor(features["categorical"], features["numeric"])
    model = train_model(
        train[feature_columns], train[target], preprocessor, problem["problem_type"],
        model_params=settings["model_params"], random_state=settings["split"]["random_state"],
    )

    # 4. Quality gate on validation: if it fails, the error stops here and nothing is saved
    val_metrics = evaluate_model(model, val[feature_columns], val[target], problem["problem_type"])
    check_quality_gate(val_metrics, settings["quality_gate"])

    # 5. Test is scored once, after the release decision, only to report it
    test_metrics = evaluate_model(model, test[feature_columns], test[target], problem["problem_type"])
    # TODO: replace print with logging (Session 10)
    print(f"Validation metrics: {val_metrics} | Test metrics (report only): {test_metrics}")

    save_model(model, paths["model"])
    return model, val_metrics


def _predict_new_shipments(settings: dict, model) -> pd.DataFrame:
    """Predict the cost of each new shipment and save it next to its ID.

    Inputs: SETTINGS (paths, problem, features, validation) and the released Pipeline.
    Outputs: DataFrame with the ID column and "prediction", also saved to SETTINGS["paths"]["predictions"].
    Why: same cleaning, rules and recipe as training (no skew); the ID keeps every estimate traceable.
    """
    paths, problem, features = settings["paths"], settings["problem"], settings["features"]
    feature_columns = features["categorical"] + features["numeric"]
    id_column = problem["id_column"]

    # 1. Same cleaning and rules as training, without the target; one bad shipment stops the batch
    df = clean_data(load_raw_data(paths["inference_data"]))
    validate_dataframe(df, [id_column] + feature_columns, **settings["validation"])

    # 2. Predict from the feature columns only, then put the ID back as the first column
    predictions = run_inference(model, df[feature_columns])
    output = pd.concat([df[[id_column]], predictions], axis=1)

    save_csv(output, paths["predictions"])
    # TODO: replace print with logging (Session 10)
    print(f"Saved {len(output)} predictions to {paths['predictions']}")
    return output


def run_pipeline(settings: dict) -> dict[str, float]:
    """Run training, the release gate and batch inference with the given settings.

    Inputs: a settings dict shaped like SETTINGS (tests pass a copy with temporary paths).
    Outputs: validation metrics of the released model.
    Why: settings as a parameter lets tests run the whole pipeline without touching real folders.
    """
    # TODO: replace print with logging (Session 10)
    print("Starting the shipping cost pipeline")
    model, val_metrics = _train_and_release(settings)
    _predict_new_shipments(settings, model)
    print(f"Done. Model: {settings['paths']['model']} | Predictions: {settings['paths']['predictions']}")
    return val_metrics


def main() -> None:
    """Entry point for python -m src.main: run the pipeline with the project SETTINGS.

    Inputs: none (uses SETTINGS). Outputs: model and predictions files on disk.
    Why: one command for the team, CI and later the stage 3 API.
    """
    # TODO: replace print with logging (Session 10)
    print("Running the pipeline with the project SETTINGS")
    run_pipeline(SETTINGS)


if __name__ == "__main__":
    main()
