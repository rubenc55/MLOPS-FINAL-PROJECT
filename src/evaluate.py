"""Score a trained model on unseen data with the business metrics.

Inputs: a fitted Pipeline, evaluation features and true costs
(the validation split, passed by main).
Outputs: {"mae": float, "mape": float}: MAE in USD, MAPE as a fraction, not a percentage.
Used by: main, which compares these values with the quality gate in config.yaml.
Why: the gate reads these exact keys and units;
a wrong key or unit would block or release a model wrongly.
"""

import logging

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_inputs(x_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> None:
    """Fail fast with a clear message before any scoring happens."""
    if problem_type != "regression":
        raise ValueError(
            f"Unsupported problem_type: {problem_type!r}. "
            "evaluate_model only supports 'regression' "
            "(see TEAM_BRIEF.md). Check SETTINGS['problem']['problem_type']."
        )
    if len(x_eval) == 0 or len(y_eval) == 0:
        raise ValueError(
            "Evaluation data is empty. Check the validation split sizes in SETTINGS['split']."
        )
    if len(x_eval) != len(y_eval):
        raise ValueError(
            "X_eval and y_eval must have the same number of rows, "
            f"got {len(x_eval)} and {len(y_eval)}. "
            "Check the split in main."
        )


# X_eval keeps the contract's uppercase name (X is the feature matrix, as in scikit-learn).
def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series,  # pylint: disable=invalid-name
                   problem_type: str) -> dict[str, float]:
    """Score a fitted model on unseen data with the metrics of the quality gate.

    Inputs: fitted Pipeline from train, X_eval and y_eval (validation split),
    problem_type ("regression").
    Outputs: {"mae": ..., "mape": ...} as plain Python floats; MAPE is a fraction, not a percentage.
    Why: main's quality gate reads these keys; scoring only unseen rows keeps the numbers honest.
    """
    logger.info("Evaluating model on %d rows", len(X_eval))
    _check_inputs(X_eval, y_eval, problem_type)
    y_pred = model.predict(X_eval)
    return {
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_eval, y_pred)),
    }
