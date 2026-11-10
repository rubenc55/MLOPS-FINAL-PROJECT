"""Generic read and write helpers for CSV files and fitted models.

Inputs: file paths (pathlib.Path), DataFrames and fitted models.
Outputs: loaded DataFrames or models; files written with their parent folders created.
Used by: load_data, main and infer.
Why: one shared way to read and write files, so every module behaves the same.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Inputs: filepath, the CSV file to read.
    Outputs: a DataFrame with the file's rows and columns.
    Why: fail fast if the file is missing,
    and keep column names clean in CSV files exported by Excel.
    """
    logger.info("Loading CSV: %s", filepath)
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {filepath}. Check the path in SETTINGS['paths']."
        )
    return pd.read_csv(filepath, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to a CSV file.

    Inputs: df, the table to save; filepath, where to write it.
    Outputs: none; the file is written and missing parent folders are created.
    Why: index=False keeps the file identical to the data, so load_csv returns the same table.
    """
    logger.info("Saving CSV: %s", filepath)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """Save a fitted model or Pipeline to disk with joblib.

    Inputs: model, the fitted object; filepath, where to write it (e.g. paths.model in config.yaml).
    Outputs: none; the file is written and missing parent folders are created.
    Why: the saved Pipeline keeps its preprocessing, so inference uses exactly the trained recipe.
    """
    logger.info("Saving model: %s", filepath)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """Load a model or Pipeline saved with save_model.

    Inputs: filepath, the .joblib file to read.
    Outputs: the fitted model, ready to predict.
    Why: fail fast with a clear message if the model was not trained yet or the path is wrong.
    """
    logger.info("Loading model: %s", filepath)
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"Model file not found: {filepath}. Run the pipeline first: python -m src.main"
        )
    return joblib.load(filepath)
