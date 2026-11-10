"""Tests for src/main.py: the split, the quality gate and the full pipeline run."""

import copy

import pandas as pd
import pytest

from src.main import (
    PROJECT_ROOT, SETTINGS, check_quality_gate, load_config, main, run_pipeline, split_data
)


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
        "clean_data": tmp_path / "processed" / "clean.csv",
        "model": tmp_path / "models" / "model.joblib",
        "predictions": tmp_path / "reports" / "predictions.csv",
    }
    settings["model_params"] = {"n_estimators": 10}
    settings["quality_gate"] = quality_gate
    return settings


def test_unit_split_data_sizes_and_no_overlap():
    """60 rows split 70/15/15 give 42, 9 and 9 rows, X and y stay aligned, and no shipment appears in two parts."""
    df = _make_raw_df()
    X, y = df.drop(columns="cost_usd"), df["cost_usd"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.15, val_size=0.15, random_state=42, stratify=False
    )

    assert (len(X_train), len(X_val), len(X_test)) == (42, 9, 9)
    for X_part, y_part in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert X_part.index.equals(y_part.index)
    ids = [set(part["shipment_id"]) for part in (X_train, X_val, X_test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_unit_split_data_stratify_keeps_class_shares():
    """With stratify=True every part keeps the class share of y (here 20% of ones)."""
    X = _make_raw_df()
    y = pd.Series([1 if i % 5 == 0 else 0 for i in range(len(X))], index=X.index)

    _, _, _, y_train, y_val, y_test = split_data(
        X, y, test_size=0.15, val_size=0.15, random_state=42, stratify=True
    )

    for y_part in (y_train, y_val, y_test):
        assert y_part.mean() == pytest.approx(0.2, abs=0.05)


def test_unit_check_quality_gate_passes_at_limit_and_fails_above():
    """Metrics equal to the limits pass; a metric above its limit raises an error that names it."""
    gate = {"mae": 200.0, "mape": 0.18}

    check_quality_gate({"mae": 200.0, "mape": 0.18}, gate)

    with pytest.raises(ValueError, match="mape"):
        check_quality_gate({"mae": 150.0, "mape": 0.25}, gate)


def test_integration_run_pipeline_end_to_end(tmp_path):
    """With a passing gate, the pipeline saves the clean table, the model and one prediction per new shipment."""
    settings = _make_settings(tmp_path, quality_gate={"mae": 1_000_000.0, "mape": 1_000_000.0})

    metrics = run_pipeline(settings)

    assert set(metrics) == {"mae", "mape"}
    assert len(pd.read_csv(settings["paths"]["clean_data"])) == 60
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


def test_unit_check_quality_gate_missing_metric_raises():
    """A gate metric that evaluate_model did not return stops the release, and the message names it."""
    with pytest.raises(ValueError, match="mape"):
        check_quality_gate({"mae": 150.0}, {"mape": 0.18})


def test_unit_main_runs_pipeline_with_project_settings(monkeypatch):
    """main() hands the project SETTINGS to run_pipeline, so python -m src.main has one source of settings."""
    calls = []
    monkeypatch.setattr("src.main.run_pipeline", calls.append)

    main()

    assert len(calls) == 1 and calls[0] is SETTINGS


def test_integration_main_runs_full_pipeline(tmp_path, monkeypatch):
    """main() really runs the pipeline with SETTINGS (pointed to tmp_path) and writes the model and predictions."""
    settings = _make_settings(tmp_path, quality_gate={"mape": 1_000_000.0})
    monkeypatch.setattr("src.main.SETTINGS", settings)

    main()

    assert settings["paths"]["model"].is_file()
    assert settings["paths"]["predictions"].is_file()


def test_unit_load_config_resolves_paths_against_project_root(tmp_path):
    """Relative paths in the YAML file become absolute paths under the repo root; other values stay as written."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  model: models/model.joblib\nquality_gate:\n  mape: 0.18\n", encoding="utf-8")

    settings = load_config(config_path)

    assert settings["paths"]["model"] == PROJECT_ROOT / "models" / "model.joblib"
    assert settings["quality_gate"] == {"mape": 0.18}


def test_unit_load_config_missing_file_raises(tmp_path):
    """A config file that does not exist stops the run with a FileNotFoundError that names the file."""
    with pytest.raises(FileNotFoundError, match="missing_config.yaml"):
        load_config(tmp_path / "missing_config.yaml")


def test_integration_project_config_has_every_setting_the_pipeline_reads():
    """The repo's config.yaml loads every section and path main uses, with the agreed gate mape <= 0.18."""
    settings = load_config(PROJECT_ROOT / "config.yaml")

    assert set(settings) == {"paths", "problem", "features", "validation", "split", "model_params", "quality_gate"}
    assert set(settings["paths"]) == {"raw_data", "inference_data", "clean_data", "model", "predictions"}
    assert all(path.is_absolute() for path in settings["paths"].values())
    assert settings["quality_gate"] == {"mape": 0.18}
