"""Tests for src/clean_data.py: fixed, stateless cleaning rules."""

import pandas as pd

from src.clean_data import clean_data
from src.load_data import load_raw_data


def test_unit_clean_data_strips_spaces_in_text_columns():
    """Leading and trailing spaces in text columns are removed; the text itself stays the same."""
    df = pd.DataFrame(
        {
            "transport_mode": [" Road", "Sea ", " Air "],
            "cargo_type": ["General", " Hazardous", "Refrigerated "],
        }
    )

    cleaned = clean_data(df)

    assert cleaned["transport_mode"].tolist() == ["Road", "Sea", "Air"]
    assert cleaned["cargo_type"].tolist() == ["General", "Hazardous", "Refrigerated"]


def test_unit_clean_data_keeps_original_numbers_and_rows():
    """clean_data returns a new table: the input stays untouched, and numbers and row count do not change."""
    df = pd.DataFrame({"transport_mode": [" Road", "Sea"], "weight_kg": [120.5, 3400.0]})

    cleaned = clean_data(df)

    assert df["transport_mode"].tolist() == [" Road", "Sea"]
    assert cleaned["weight_kg"].tolist() == [120.5, 3400.0]
    assert len(cleaned) == len(df)


def test_integration_clean_data_after_load_data(tmp_path):
    """A raw CSV read by load_raw_data and cleaned by clean_data has clean text values."""
    filepath = tmp_path / "shipments.csv"
    filepath.write_text("shipment_id,transport_mode,weight_kg\n1, Road ,120.5\n2,Air,15.25\n", encoding="utf-8")

    cleaned = clean_data(load_raw_data(filepath))

    assert cleaned["transport_mode"].tolist() == ["Road", "Air"]
    assert cleaned["weight_kg"].tolist() == [120.5, 15.25]
