"""Tests for src/clean_data.py: fixed, stateless cleaning rules."""

import pandas as pd

from src.clean_data import clean_dataframe
from src.load_data import load_raw_data


def test_unit_clean_data_strips_spaces_in_text_columns():
    """Leading and trailing spaces in text columns are removed; the text itself stays the same."""
    df = pd.DataFrame(
        {
            "transport_mode": [" Road", "Sea ", " Air "],
            "cargo_type": ["General", " Hazardous", "Refrigerated "],
        }
    )

    cleaned = clean_dataframe(df)

    assert cleaned["transport_mode"].tolist() == ["Road", "Sea", "Air"]
    assert cleaned["cargo_type"].tolist() == ["General", "Hazardous", "Refrigerated"]


def test_unit_clean_data_keeps_original_numbers_and_rows():
    """clean_dataframe returns a new table: the input stays untouched, and numbers and row count do not change."""
    df = pd.DataFrame({"transport_mode": [" Road", "Sea"], "weight_kg": [120.5, 3400.0]})

    cleaned = clean_dataframe(df)

    assert df["transport_mode"].tolist() == [" Road", "Sea"]
    assert cleaned["weight_kg"].tolist() == [120.5, 3400.0]
    assert len(cleaned) == len(df)


def test_integration_clean_data_after_load_data(tmp_path):
    """A raw CSV read by load_raw_data and cleaned by clean_dataframe has clean text values."""
    filepath = tmp_path / "shipments.csv"
    filepath.write_text("shipment_id,transport_mode,weight_kg\n1, Road ,120.5\n2,Air,15.25\n", encoding="utf-8")

    cleaned = clean_dataframe(load_raw_data(filepath))

    assert cleaned["transport_mode"].tolist() == ["Road", "Air"]
    assert cleaned["weight_kg"].tolist() == [120.5, 15.25]


def test_unit_clean_dataframe_training_mode_drops_rows_without_target():
    """With target_column (training), only rows whose target is missing are dropped; the index is kept."""
    df = pd.DataFrame({"transport_mode": ["Road", "Sea", "Air"], "cost_usd": [250.5, None, 90.0]})

    cleaned = clean_dataframe(df, target_column="cost_usd")

    assert cleaned.index.tolist() == [0, 2]
    assert cleaned["cost_usd"].tolist() == [250.5, 90.0]
    assert len(df) == 3


def test_unit_clean_dataframe_inference_mode_never_drops_rows():
    """Without target_column (inference), every row is kept, even one with missing values."""
    df = pd.DataFrame({"transport_mode": ["Road", None], "cost_usd": [250.5, None]})

    cleaned = clean_dataframe(df)

    assert len(cleaned) == len(df)


def test_unit_clean_dataframe_leaves_absent_target_column_to_validate():
    """A target column that is not in the table drops nothing, so validate can report it as missing."""
    df = pd.DataFrame({"transport_mode": ["Road", "Air"]})

    cleaned = clean_dataframe(df, target_column="cost_usd")

    assert len(cleaned) == len(df)
