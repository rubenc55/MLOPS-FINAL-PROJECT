"""Tests for src/evaluate.py: metric keys, known values and fail-fast input checks."""

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_baseline, evaluate_model


def _make_model_and_data() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Tiny fitted Pipeline that always predicts 100 USD, so the expected metrics are easy to compute by hand."""
    X = pd.DataFrame({"distance_km": [100.0, 200.0, 300.0, 400.0]})
    y = pd.Series([100.0, 200.0, 50.0, 100.0])
    model = Pipeline([("model", DummyRegressor(strategy="constant", constant=100.0))])
    model.fit(X, y)
    return model, X, y


def test_unit_evaluate_returns_mae_and_mape_as_floats():
    model, X, y = _make_model_and_data()

    metrics = evaluate_model(model, X, y, problem_type="regression")

    assert set(metrics) == {"mae", "mape"}
    assert all(type(value) is float for value in metrics.values())
    assert metrics["mae"] == pytest.approx(37.5)
    assert metrics["mape"] == pytest.approx(0.375)


def test_unit_evaluate_raises_on_empty_or_mismatched_inputs():
    model, X, y = _make_model_and_data()

    with pytest.raises(ValueError, match="empty"):
        evaluate_model(model, X.iloc[0:0], y.iloc[0:0], problem_type="regression")

    with pytest.raises(ValueError, match="same number of rows"):
        evaluate_model(model, X, y.iloc[:3], problem_type="regression")


def test_unit_evaluate_rejects_unknown_problem_type():
    model, X, y = _make_model_and_data()

    with pytest.raises(ValueError, match="problem_type"):
        evaluate_model(model, X, y, problem_type="classification")


def test_integration_evaluate_scores_output_of_train_model():
    from src.features import get_feature_preprocessor
    from src.train import train_model

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
    preprocessor = get_feature_preprocessor(
        categorical_columns=["transport_mode", "cargo_type"],
        numeric_columns=["distance_km", "weight_kg", "volume_m3", "urgent", "month"],
    )
    model = train_model(
        X, y, preprocessor, problem_type="regression", model_params={"n_estimators": 10}, random_state=42
    )

    metrics = evaluate_model(model, X, y, problem_type="regression")

    assert set(metrics) == {"mae", "mape"}
    assert metrics["mae"] >= 0.0
    assert 0.0 <= metrics["mape"] < 1.0


def test_unit_evaluate_baseline_uses_train_median_per_group():
    """Each eval row gets the median train cost of its group: Road 100 and Sea 300, so the errors are known."""
    x_train = pd.DataFrame({"transport_mode": ["Road", "Road", "Road", "Sea", "Sea"]})
    y_train = pd.Series([90.0, 100.0, 500.0, 200.0, 400.0])
    x_eval = pd.DataFrame({"transport_mode": ["Road", "Sea"]})
    y_eval = pd.Series([200.0, 300.0])

    metrics = evaluate_baseline(x_train, y_train, x_eval, y_eval, "transport_mode")

    assert metrics == pytest.approx({"mae": 50.0, "mape": 0.25})  # errors 100 and 0; 100/200 and 0/300


def test_unit_evaluate_baseline_unseen_group_uses_overall_median():
    """A group that never appears in train (here Air) gets the overall train median instead of no estimate."""
    x_train = pd.DataFrame({"transport_mode": ["Road", "Sea", "Sea"]})
    y_train = pd.Series([100.0, 200.0, 400.0])

    metrics = evaluate_baseline(
        x_train, y_train, pd.DataFrame({"transport_mode": ["Air"]}), pd.Series([200.0]), "transport_mode"
    )

    assert metrics == pytest.approx({"mae": 0.0, "mape": 0.0})


def test_unit_evaluate_baseline_missing_group_column_raises():
    """A baseline column that is not a feature stops the run with a message naming the setting."""
    x = pd.DataFrame({"distance_km": [100.0]})

    with pytest.raises(ValueError, match="Baseline column 'transport_mode'"):
        evaluate_baseline(x, pd.Series([1.0]), x, pd.Series([1.0]), "transport_mode")
