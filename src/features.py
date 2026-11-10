"""Build the shipment feature-preprocessing recipe.

Inputs: categorical and numeric column-name lists.
Outputs: an unfitted ColumnTransformer.
Used by: the training orchestration.
Why: keep feature preparation consistent in training and inference.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _validate_column_list(
    columns: list[str],
    argument_name: str,
) -> None:
    """Reject empty, invalid or duplicate column-name lists."""
    if not isinstance(columns, list) or not columns:
        raise ValueError(
            f"{argument_name} must be a non-empty list of column names."
        )

    if any(
        not isinstance(column, str) or not column.strip()
        for column in columns
    ):
        raise ValueError(
            f"{argument_name} must contain non-empty string column names."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            f"{argument_name} must not contain duplicate column names."
        )


def get_feature_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> ColumnTransformer:
    """Create the unfitted feature-preprocessing recipe.

    Inputs: categorical and numeric column-name lists.
    Outputs: an unfitted ColumnTransformer.
    Why: reuse the same preparation during training and inference.
    """
    print("Building feature preprocessor")

    _validate_column_list(categorical_columns, "categorical_columns")
    _validate_column_list(numeric_columns, "numeric_columns")

    if set(categorical_columns) & set(numeric_columns):
        raise ValueError(
            "Categorical and numeric column lists must not overlap."
        )

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns.copy(),
            ),
            (
                "num",
                "passthrough",
                numeric_columns.copy(),
            ),
        ],
        remainder="drop",
    )


# Recommendations for later:
# - Monitor category frequencies before revising validation rules.
# - Add tests for missing columns and different training category subsets.
