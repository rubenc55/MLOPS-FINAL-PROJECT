"""Apply fixed cleaning rules that learn nothing from the data.

Inputs: a DataFrame from load_data (training) or new shipments (inference).
Outputs: a new DataFrame with the same rows and columns, with text values trimmed.
Used by: main, between load_data and validate.
Why: stray spaces typed in the TMS (" Road") must not make validate reject a valid shipment.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading and trailing spaces in every text column.

    Inputs: df, the table to clean (it is not modified).
    Outputs: a copy with the same rows and columns; numbers are unchanged.
    Why: fixed rules, applied the same way in training and inference, learn nothing from the data.
    """
    cleaned = df.copy()
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
    print(f"Cleaned text columns: {text_columns}")
    return cleaned
