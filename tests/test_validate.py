"""Tests for src/validate.py: the data quality gate."""

import pandas as pd
import pytest

from src.clean_data import clean_data
from src.validate import validate_dataframe


REQUIRED = ["transport_mode", "weight_kg", "month"]
ALLOWED = {"transport_mode": ["Road", "Sea", "Air"]}
RANGES = {"weight_kg": (10, 9900), "month": (1, 12)}


def _make_df() -> pd.DataFrame:
    """Build a tiny table that passes every rule, including a positive target."""
    return pd.DataFrame(
        {
            "transport_mode": ["Road", "Air"],
            "weight_kg": [120.5, 15.25],
            "month": [1, 12],
            "cost_usd": [250.5, 90.0],
        }
    )


def test_unit_validate_dataframe_valid_table_passes():
    """A table that meets every rule passes, in training (with target) and inference (without target)."""
    df = _make_df()

    validate_dataframe(df, REQUIRED, target_column="cost_usd", allowed_values=ALLOWED, numeric_ranges=RANGES)
    validate_dataframe(df.drop(columns="cost_usd"), REQUIRED, allowed_values=ALLOWED, numeric_ranges=RANGES)


def test_unit_validate_dataframe_missing_column_raises():
    """A missing required column stops the pipeline, and the message names it."""
    df = _make_df().drop(columns="weight_kg")

    with pytest.raises(ValueError, match="weight_kg"):
        validate_dataframe(df, REQUIRED, allowed_values=ALLOWED, numeric_ranges=RANGES)


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("transport_mode", "Rail"),
        ("weight_kg", 15000.0),
        ("month", 13),
    ],
)
def test_unit_validate_dataframe_bad_value_raises(column, bad_value):
    """An unknown category or a value outside its range stops the pipeline, and the message names the column."""
    df = _make_df()
    df.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=column):
        validate_dataframe(df, REQUIRED, allowed_values=ALLOWED, numeric_ranges=RANGES)


def test_unit_validate_dataframe_lists_every_problem_in_one_error():
    """Several problems, including a non-positive target, are all reported in a single error."""
    df = _make_df()
    df.loc[0, "transport_mode"] = "Rail"
    df.loc[1, "cost_usd"] = -5.0

    with pytest.raises(ValueError) as error:
        validate_dataframe(df, REQUIRED, target_column="cost_usd", allowed_values=ALLOWED, numeric_ranges=RANGES)

    message = str(error.value)
    assert "transport_mode" in message
    assert "cost_usd" in message


def test_integration_validate_after_clean_data():
    """A value with stray spaces (" Road ") passes validation once clean_data has trimmed it."""
    df = _make_df()
    df.loc[0, "transport_mode"] = " Road "

    cleaned = clean_data(df)

    validate_dataframe(cleaned, REQUIRED, target_column="cost_usd", allowed_values=ALLOWED, numeric_ranges=RANGES)
