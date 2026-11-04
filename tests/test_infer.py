"""Tests for src/infer.py: output schema, index kept, input untouched and fail-fast input checks."""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.infer import run_inference


def _make_model_and_data() -> tuple[Pipeline, pd.DataFrame]:
    """Tiny fitted Pipeline that always predicts 100 USD, and new rows indexed by shipment_id."""
    X_fit = pd.DataFrame({"distance_km": [100.0, 200.0, 300.0]})
    model = Pipeline([("model", DummyRegressor(strategy="constant", constant=100.0))])
    model.fit(X_fit, pd.Series([90.0, 100.0, 110.0]))
    X_infer = pd.DataFrame(
        {"distance_km": [150.0, 250.0]},
        index=pd.Index(["SHP-001", "SHP-002"], name="shipment_id"),
    )
    return model, X_infer


def test_unit_infer_returns_prediction_column_with_same_index():
    model, X_infer = _make_model_and_data()

    predictions = run_inference(model, X_infer)

    assert list(predictions.columns) == ["prediction"]
    assert predictions.index.equals(X_infer.index)
    assert predictions["prediction"].tolist() == pytest.approx([100.0, 100.0])


def test_unit_infer_does_not_modify_input():
    model, X_infer = _make_model_and_data()
    before = X_infer.copy()

    run_inference(model, X_infer)

    pd.testing.assert_frame_equal(X_infer, before)


def test_unit_infer_raises_on_empty_data():
    model, X_infer = _make_model_and_data()

    with pytest.raises(ValueError, match="empty"):
        run_inference(model, X_infer.iloc[0:0])


def test_unit_infer_raises_on_non_dataframe_input():
    model, X_infer = _make_model_and_data()

    with pytest.raises(TypeError, match="DataFrame"):
        run_inference(model, X_infer.to_numpy())


def test_unit_infer_raises_when_probability_requested_from_regressor():
    model, X_infer = _make_model_and_data()

    with pytest.raises(ValueError, match="probabilit"):
        run_inference(model, X_infer, include_proba=True)


def test_unit_infer_adds_probability_column_for_classifier():
    X_fit = pd.DataFrame({"distance_km": [100.0, 200.0, 300.0, 400.0]})
    model = Pipeline([("model", DummyClassifier(strategy="prior"))])
    model.fit(X_fit, pd.Series([0, 1, 1, 1]))
    X_infer = pd.DataFrame({"distance_km": [150.0, 250.0]}, index=[10, 20])

    predictions = run_inference(model, X_infer, include_proba=True)

    assert list(predictions.columns) == ["prediction", "probability"]
    assert predictions.index.equals(X_infer.index)
    assert predictions["probability"].tolist() == pytest.approx([0.75, 0.75])


def test_integration_infer_predicts_with_output_of_train_model():
    from src.train import train_model

    X = pd.DataFrame({
        "transport_mode": ["Road", "Sea", "Air", "Road"] * 5,
        "cargo_type": ["General", "Hazardous", "Refrigerated", "General"] * 5,
        "distance_km": [100.0 * i for i in range(1, 21)],
        "weight_kg": [50.0 * i for i in range(1, 21)],
        "volume_m3": [0.5 * i for i in range(1, 21)],
        "urgent": [0, 1] * 10,
        "month": [(i % 12) + 1 for i in range(20)],
    }, index=pd.Index([f"SHP-{i:03d}" for i in range(20)], name="shipment_id"))
    y = pd.Series([200.0 + 10.0 * i for i in range(20)], index=X.index)
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["transport_mode", "cargo_type"]),
        ("numeric", "passthrough", ["distance_km", "weight_kg", "volume_m3", "urgent", "month"]),
    ])
    model = train_model(
        X, y, preprocessor, problem_type="regression", model_params={"n_estimators": 10}, random_state=42
    )
    X_new = X.iloc[:5]

    predictions = run_inference(model, X_new)

    assert list(predictions.columns) == ["prediction"]
    assert predictions.index.equals(X_new.index)
    assert predictions["prediction"].notna().all()
    assert (predictions["prediction"] > 0).all()
