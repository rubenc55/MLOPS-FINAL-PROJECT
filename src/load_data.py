"""Read the raw shipments data for the pipeline, without changing it.

Inputs: raw_data_path, the CSV file set in config.yaml (paths.raw_data).
Outputs: a DataFrame with every raw row and column.
Used by: main, as the first pipeline step; its output goes to clean_data.
Why: stop the pipeline early, with a clear message, when the raw file is missing, a folder or empty.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Read the raw shipments CSV exactly as it is.

    Inputs: raw_data_path, path to the raw CSV file.
    Outputs: a DataFrame with all raw rows and columns, unchanged.
    Why: a missing, folder or empty input must stop the pipeline here,
    not produce a useless model later.
    """
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_data_path}. Check SETTINGS['paths']."
        )
    if raw_data_path.is_dir():
        raise ValueError(f"Raw data path is a folder, not a CSV file: {raw_data_path}")
    if raw_data_path.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty: {raw_data_path}")

    df = load_csv(raw_data_path)
    if df.empty:
        raise ValueError(f"Raw data file has headers but no rows: {raw_data_path}")
    logger.info("Loaded raw data: %d rows, %d columns", len(df), df.shape[1])
    return df
