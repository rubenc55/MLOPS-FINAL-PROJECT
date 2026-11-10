"""Data quality gate: stop the pipeline when the data breaks the agreed rules.

Inputs: a cleaned DataFrame and the rules from SETTINGS
(required columns, allowed values, numeric ranges).
Outputs: nothing when the data is valid; a ValueError listing every problem otherwise.
Used by: main, after clean_data, in training (with the target) and inference (without it).
Why: wrong units or unknown categories must stop the pipeline instead of producing an invented cost.
"""

import pandas as pd


def _find_value_problems(df: pd.DataFrame, allowed_values: dict, numeric_ranges: dict) -> list:
    """List unknown categories and out-of-range numbers, skipping columns that are missing."""
    problems = []
    for column, allowed in allowed_values.items():
        if column in df.columns:
            unknown = sorted(set(df[column].dropna()) - set(allowed))
            if unknown:
                problems.append(f"{column}: unknown values {unknown}, allowed {allowed}")
    for column, (low, high) in numeric_ranges.items():
        if column in df.columns:
            outside = df[(df[column] < low) | (df[column] > high)]
            if not outside.empty:
                problems.append(f"{column}: {len(outside)} rows outside the range {low} to {high}")
    return problems


def _find_column_problems(df: pd.DataFrame, columns_to_check: list) -> list:
    """List columns that are missing, then present columns that have missing values."""
    problems = [
        f"missing column: {column}" for column in columns_to_check if column not in df.columns
    ]
    for column in [column for column in columns_to_check if column in df.columns]:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            problems.append(f"{column}: {missing_count} missing values")
    return problems


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    target_column: str | None = None,
    allowed_values: dict | None = None,
    numeric_ranges: dict | None = None,
) -> None:
    """Check the data against the agreed rules and raise one ValueError that lists every problem.

    Inputs: df; required_columns; target_column (training only, must be greater than 0);
    allowed_values {column: list of values}; numeric_ranges {column: (min, max)}, limits included.
    Outputs: None when every rule passes.
    Why: the gate runs before training and inference, so bad data never reaches the model.
    """
    columns_to_check = list(required_columns) + ([target_column] if target_column else [])
    problems = _find_column_problems(df, columns_to_check)

    if target_column in df.columns and (df[target_column] <= 0).any():
        problems.append(f"{target_column}: target must be greater than 0")

    problems += _find_value_problems(df, allowed_values or {}, numeric_ranges or {})
    if problems:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(problems))
    print(f"Data validation passed: {len(df)} rows")
