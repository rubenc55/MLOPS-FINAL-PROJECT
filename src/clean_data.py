"""Apply fixed cleaning rules that learn nothing from the data.

Inputs: a DataFrame from load_data (training) or new shipments (inference).
Outputs: a new DataFrame with the same columns and text values trimmed; in training, rows
without a target are dropped.
Used by: main, between load_data and validate.
Why: stray spaces typed in the TMS (" Road") must not make validate reject a valid shipment.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Trim leading and trailing spaces in every text column; in training, drop rows with no target.

    Inputs: df_raw, the table to clean (it is not modified); target_column, the target name in
    training mode, or None in inference mode.
    Outputs: a copy with the same columns; numbers are unchanged. Training mode drops rows whose
    target is missing; inference mode never drops rows.
    Why: fixed rules, applied the same way in training and inference, learn nothing from the data;
    a row without a cost cannot teach the model, but every new shipment needs a prediction.
    """
    cleaned = df_raw.copy()
    text_columns = [
        column
        for column in cleaned.columns
        if pd.api.types.is_object_dtype(cleaned[column])
        or pd.api.types.is_string_dtype(cleaned[column])
    ]
    for column in text_columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    logger.info("Cleaned text columns: %s", text_columns)
    # A missing target column is left for validate to report with its clear message
    if target_column is not None and target_column in cleaned.columns:
        rows_before = len(cleaned)
        cleaned = cleaned.dropna(subset=[target_column])
        logger.info("Dropped %d rows with a missing %s", rows_before - len(cleaned), target_column)
    return cleaned
