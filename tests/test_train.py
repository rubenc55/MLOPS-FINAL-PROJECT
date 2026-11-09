"""Tests for src/train.py: one fitted Pipeline, trained reproducibly, with fail-fast input checks."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from src.train import train_model


CATEGORICAL = ["transport_mode", "cargo_type"]
NUMERIC = ["distance_km", "weight_kg", "volume_m3", "urgent", "month"]
PARAMS = {"n_estimators": 10, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1}


def _make_preprocessor() -> ColumnTransformer:
    """Unfitted recipe like the one agreed in TEAM_BRIEF.md, built here so train tests do not depend on features."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])


def _make_data() -> tuple[pd.DataFrame, pd.Series]:
    """Tiny train split with raw columns, as main passes it after validate and the split."""
    X = pd.DataFrame({
        "transport_mode": ["Road", "Sea", "Air", "Road"] * 5,
        "cargo_type": ["General", "Hazardous", "Refrigerated", "General"] * 5,
        "distance_km": [100.0 * i for i in range(1, 21)],
        "weight_kg": [50.0 * i for i in range(1, 21)],
        "volume_m3": [0.5 * i for i in range(1, 21)],
        "urgent": [0, 1] * 10,
        "month": [(i % 12) + 1 for i in range(20)],
    })
    y = pd.Series([200.0 + 10.0 * i for i in range(20)])
    return X, y


def test_unit_train_returns_fitted_pipeline_with_preprocessor_and_model():
    """train_model returns one fitted Pipeline that takes raw columns and predicts one cost per row."""
    X, y = _make_data()

    model = train_model(X, y, _make_preprocessor(), problem_type="regression", model_params=PARAMS, random_state=42)

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocessor", "model"]
    assert isinstance(model.named_steps["model"], RandomForestRegressor)
    check_is_fitted(model)
    assert len(model.predict(X)) == len(X)


def test_unit_train_applies_model_params_and_random_state():
    """Hyperparameters from SETTINGS and the seed reach the RandomForestRegressor unchanged."""
    X, y = _make_data()

    model = train_model(X, y, _make_preprocessor(), problem_type="regression", model_params=PARAMS, random_state=7)

    params = model.named_steps["model"].get_params()
    for name, value in PARAMS.items():
        assert params[name] == value
    assert params["random_state"] == 7


def test_unit_train_same_seed_gives_same_predictions():
    """Same data and same seed give identical predictions: training is reproducible."""
    X, y = _make_data()

    first = train_model(X, y, _make_preprocessor(), problem_type="regression", model_params=PARAMS, random_state=42)
    second = train_model(X, y, _make_preprocessor(), problem_type="regression", model_params=PARAMS, random_state=42)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_unit_train_leaves_the_given_preprocessor_unfitted():
    """The recipe from features stays unfitted, so main can reuse it without hidden state."""
    X, y = _make_data()
    preprocessor = _make_preprocessor()

    train_model(X, y, preprocessor, problem_type="regression", model_params=PARAMS, random_state=42)

    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)


def test_unit_train_raises_on_empty_or_mismatched_inputs():
    """Empty or misaligned train data stops before fitting, with a clear message."""
    X, y = _make_data()

    with pytest.raises(ValueError, match="empty"):
        train_model(X.iloc[0:0], y.iloc[0:0], _make_preprocessor(), problem_type="regression",
                    model_params=PARAMS, random_state=42)

    with pytest.raises(ValueError, match="same number of rows"):
        train_model(X, y.iloc[:5], _make_preprocessor(), problem_type="regression",
                    model_params=PARAMS, random_state=42)


def test_unit_train_rejects_unknown_problem_type():
    """Only regression is supported, as agreed in TEAM_BRIEF.md."""
    X, y = _make_data()

    with pytest.raises(ValueError, match="problem_type"):
        train_model(X, y, _make_preprocessor(), problem_type="classification", model_params=PARAMS, random_state=42)


def test_unit_train_rejects_random_state_inside_model_params():
    """The seed has one source (the random_state argument), so two conflicting seeds cannot exist."""
    X, y = _make_data()

    with pytest.raises(ValueError, match="random_state"):
        train_model(X, y, _make_preprocessor(), problem_type="regression",
                    model_params={**PARAMS, "random_state": 1}, random_state=42)
