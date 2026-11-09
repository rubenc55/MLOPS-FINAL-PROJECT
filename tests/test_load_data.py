"""Tests for src/load_data.py: read the raw shipments CSV and fail fast on bad files."""

import pandas as pd
import pytest

from src.load_data import load_raw_data
from src.utils import save_csv


def test_unit_load_raw_data_reads_rows_and_columns(tmp_path):
    """A valid CSV comes back with its exact columns and values."""
    filepath = tmp_path / "shipments.csv"
    filepath.write_text("shipment_id,cost_usd\n1,250.5\n2,90.0\n", encoding="utf-8")

    df = load_raw_data(filepath)

    assert list(df.columns) == ["shipment_id", "cost_usd"]
    assert df["cost_usd"].tolist() == [250.5, 90.0]


def test_unit_load_raw_data_missing_file_raises(tmp_path):
    """A raw data path that does not exist stops the pipeline with FileNotFoundError."""
    missing = tmp_path / "shipments.csv"

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing)


@pytest.mark.parametrize("content", ["", "shipment_id,cost_usd\n"])
def test_unit_load_raw_data_empty_file_raises(tmp_path, content):
    """An empty file, or one with headers but no rows, stops the pipeline with ValueError."""
    filepath = tmp_path / "shipments.csv"
    filepath.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_raw_data(filepath)


def test_integration_load_raw_data_reads_file_saved_by_utils(tmp_path):
    """A table saved with utils.save_csv is read back by load_raw_data unchanged."""
    df = pd.DataFrame({"shipment_id": [1, 2], "transport_mode": ["Road", "Air"], "cost_usd": [250.5, 90.0]})
    filepath = tmp_path / "raw" / "shipments.csv"

    save_csv(df, filepath)
    loaded = load_raw_data(filepath)

    pd.testing.assert_frame_equal(loaded, df)


def test_unit_load_raw_data_folder_path_raises(tmp_path):
    """A folder given as the raw data path stops the pipeline with a clear ValueError."""
    with pytest.raises(ValueError, match="folder"):
        load_raw_data(tmp_path)
