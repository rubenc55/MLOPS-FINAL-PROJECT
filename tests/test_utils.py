"""Tests for src/utils.py: CSV and model read and write helpers."""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.utils import load_csv, load_model, save_csv, save_model


def _make_df() -> pd.DataFrame:
    """Build a tiny DataFrame with text, integer and decimal columns."""
    return pd.DataFrame(
        {
            "shipment_id": [1, 2, 3],
            "transport_mode": ["Road", "Sea", "Air"],
            "weight_kg": [120.5, 3400.0, 15.25],
        }
    )


def test_unit_save_then_load_csv_returns_same_data(tmp_path):
    """save_csv then load_csv gives back the same rows, columns and values."""
    df = _make_df()
    filepath = tmp_path / "nested" / "shipments.csv"

    save_csv(df, filepath)
    loaded = load_csv(filepath)

    pd.testing.assert_frame_equal(loaded, df)


def test_unit_load_csv_missing_file_raises(tmp_path):
    """load_csv on a path that does not exist fails fast with FileNotFoundError."""
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing)


def test_unit_load_csv_handles_excel_bom(tmp_path):
    """A CSV exported by Excel as CSV UTF-8 starts with an invisible mark; column names stay clean."""
    filepath = tmp_path / "excel_export.csv"
    filepath.write_text("shipment_id,transport_mode\n1,Road\n", encoding="utf-8-sig")

    loaded = load_csv(filepath)

    assert list(loaded.columns) == ["shipment_id", "transport_mode"]


def test_integration_save_then_load_model_gives_same_predictions(tmp_path):
    """A model saved with save_model and reloaded with load_model predicts exactly the same values."""
    X = pd.DataFrame({"distance_km": [100.0, 500.0, 1000.0, 2000.0]})
    y = pd.Series([50.0, 210.0, 400.0, 790.0])
    model = LinearRegression().fit(X, y)
    filepath = tmp_path / "models" / "model.joblib"

    save_model(model, filepath)
    reloaded = load_model(filepath)

    assert list(reloaded.predict(X)) == list(model.predict(X))