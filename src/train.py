"""Fit the feature recipe and the model together as one Pipeline on the train split.

Inputs: X_train and y_train (train split, passed by main), the unfitted preprocessor from features,
problem_type, model_params from SETTINGS and the random seed.
Outputs: one fitted Pipeline (preprocessor + RandomForestRegressor),
saved by main as models/model.joblib.
Used by: main; the fitted Pipeline then goes to evaluate and infer.
Why: fitting on the train split only keeps validation and test metrics honest, and one artifact
means what we evaluate is exactly what we deploy.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def _check_inputs(x_train: pd.DataFrame, y_train: pd.Series, problem_type: str,
                  model_params: dict) -> None:
    """Fail fast with a clear message before any fitting happens."""
    if problem_type != "regression":
        raise ValueError(
            f"Unsupported problem_type: {problem_type!r}. train_model only supports 'regression' "
            "(see TEAM_BRIEF.md). Check SETTINGS['problem']['problem_type']."
        )
    if len(x_train) == 0 or len(y_train) == 0:
        raise ValueError("Train data is empty. Check the split sizes in SETTINGS['split'].")
    if len(x_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have the same number of rows, "
            f"got {len(x_train)} and {len(y_train)}. "
            "Check the split in main."
        )
    if "random_state" in model_params:
        raise ValueError(
            "random_state must not be inside model_params; pass it as the random_state argument "
            "(SETTINGS['split']['random_state']) so the pipeline has one seed."
        )


# Six arguments is the agreed team contract for train_model, so its signature must not change.
def train_model(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
    problem_type: str, model_params: dict, random_state: int
) -> Pipeline:
    """Build preprocessor + RandomForestRegressor as one Pipeline and fit it on the train split.

    Inputs: X_train with raw feature columns, y_train (cost_usd),
    the unfitted preprocessor from features,
    problem_type ("regression"), model_params (RandomForestRegressor settings), random_state (seed).
    Outputs: the fitted Pipeline with steps "preprocessor" and "model".
    Why: the preprocessor is cloned, so the recipe from features stays unfitted and reusable.
    """
    print(f"Training model on {len(x_train)} rows")
    model_params = model_params or {}
    _check_inputs(x_train, y_train, problem_type, model_params)
    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(**model_params, random_state=random_state)),
    ])
    model.fit(x_train, y_train)
    return model
