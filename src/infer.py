"""Turn new, unseen shipments into cost predictions with the saved Pipeline.

Inputs: the fitted Pipeline (saved at paths.model in config.yaml) and new rows
with the same feature columns, no target.
Outputs: a DataFrame with a fixed "prediction" column (+ "probability" if asked),
same index as the input.
Used by: main (batch, writes paths.predictions) and the API in stage 3 (live requests).
Why: no training happens here, so batch and live predictions match; keeping the index lets each
prediction join back to its ID column.
"""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_inputs(model: Pipeline, x_infer: pd.DataFrame, include_proba: bool) -> None:
    """Fail fast with a clear message before any prediction happens."""
    if not isinstance(x_infer, pd.DataFrame):
        raise TypeError(
            f"X_infer must be a pandas DataFrame, got {type(x_infer).__name__}. "
            "Pass the cleaned and validated inference rows from main."
        )
    if len(x_infer) == 0:
        raise ValueError(
            "Inference data is empty. Check the paths.inference_data file in config.yaml."
        )
    if include_proba and not hasattr(model, "predict_proba"):
        raise ValueError(
            "include_proba=True but the model cannot return probabilities (regression model). "
            "Use include_proba=False for regression predictions."
        )


# X_infer keeps the contract's uppercase name (X is the feature matrix, as in scikit-learn).
def run_inference(model: Pipeline, X_infer: pd.DataFrame,  # pylint: disable=invalid-name
                  include_proba: bool = False) -> pd.DataFrame:
    """Predict with a fitted Pipeline and return the results with the input index.

    Inputs: fitted Pipeline from train, X_infer (new rows with the feature columns),
    include_proba (adds the positive-class "probability" column; classifiers only).
    Outputs: DataFrame with column "prediction" (+ "probability"), same index as X_infer.
    Why: a fixed output schema lets main and the API rely on it; the input is never modified.
    """
    _check_inputs(model, X_infer, include_proba)
    logger.info("Running inference on %d rows", len(X_infer))
    predictions = pd.DataFrame({"prediction": model.predict(X_infer)}, index=X_infer.index)
    if include_proba:
        predictions["probability"] = model.predict_proba(X_infer)[:, 1]
    return predictions
