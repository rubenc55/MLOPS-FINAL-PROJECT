"""Tests for src/main.py: the split, the quality gate and the full pipeline run."""

import copy

import pandas as pd
import pytest

from src.main import SETTINGS, check_quality_gate, run_pipeline, split_data


def _make_raw_df(n_rows: int = 60) -> pd.DataFrame:
    """Build synthetic shipments that pass every validation rule, with a simple known cost."""
    ids = list(range(1, n_rows + 1))
    df = pd.DataFrame(
        {
            "shipment_id": ids,
            "transport_mode": [["Road", "Sea", "Air"][i % 3] for i in ids],
            "cargo_type": [["General", "Refrigerated", "Hazardous"][i % 3] for i in ids],
            "distance_km": [100 + i * 50 for i in ids],
            "weight_kg": [50 + i * 20 for i in ids],
            "volume_m3": [1 + i * 0.5 for i in ids],
            "urgent": [i % 2 for i in ids],
            "month": [i % 12 + 1 for i in ids],
        }
    )
    df["cost_usd"] = 0.5 * df["distance_km"] + 0.2 * df["weight_kg"] + 50
    return df


def _make_settings(tmp_path, quality_gate: dict) -> dict:
    """Copy SETTINGS, point every path to tmp_path, write the input CSVs and use a small, fast model."""
    settings = copy.deepcopy(SETTINGS)
    raw = _make_raw_df()
    raw_path = tmp_path / "raw" / "shipments.csv"
    inference_path = tmp_path / "inference" / "new_shipments.csv"
    raw_path.parent.mkdir()
    inference_path.parent.mkdir()
    raw.to_csv(raw_path, index=False)
    raw.drop(columns="cost_usd").head(3).to_csv(inference_path, index=False)
    settings["paths"] = {
        "raw_data": raw_path,
        "inference_data": inference_path,
        "model": tmp_path / "models" / "model.joblib",
        "predictions": tmp_path / "reports" / "predictions.csv",
    }
    settings["model_params"] = {"n_estimators": 10}
    settings["quality_gate"] = quality_gate
    return settings


def test_unit_split_data_sizes_and_no_overlap():
    """60 rows split 70/15/15 give 42, 9 and 9 rows, and no shipment appears in two parts."""
    df = _make_raw_df()

    train, val, test = split_data(df, test_size=0.15, val_size=0.15, random_state=42)

    assert (len(train), len(val), len(test)) == (42, 9, 9)
    ids = [set(part["shipment_id"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_unit_check_quality_gate_passes_at_limit_and_fails_above():
    """Metrics equal to the limits pass; a metric above its limit raises an error that names it."""
    gate = {"mae": 200.0, "mape": 0.18}

    check_quality_gate({"mae": 200.0, "mape": 0.18}, gate)

    with pytest.raises(ValueError, match="mape"):
        check_quality_gate({"mae": 150.0, "mape": 0.25}, gate)


def test_integration_run_pipeline_end_to_end(tmp_path):
    """With a passing gate, the pipeline saves the model and one prediction per new shipment, with its ID."""
    settings = _make_settings(tmp_path, quality_gate={"mae": 1_000_000.0, "mape": 1_000_000.0})

    metrics = run_pipeline(settings)

    assert set(metrics) == {"mae", "mape"}
    assert settings["paths"]["model"].is_file()
    predictions = pd.read_csv(settings["paths"]["predictions"])
    assert list(predictions.columns) == ["shipment_id", "prediction"]
    assert predictions["shipment_id"].tolist() == [1, 2, 3]


def test_integration_run_pipeline_failed_gate_saves_no_model(tmp_path):
    """With an impossible gate (mae <= 0), the pipeline stops and writes no model or predictions."""
    settings = _make_settings(tmp_path, quality_gate={"mae": 0.0, "mape": 0.0})

    with pytest.raises(ValueError, match="quality gate"):
        run_pipeline(settings)

    assert not settings["paths"]["model"].exists()
    assert not settings["paths"]["predictions"].exists()
