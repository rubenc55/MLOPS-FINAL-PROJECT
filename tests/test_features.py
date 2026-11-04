"""Tests for the shipment feature-preprocessing recipe."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.features import get_feature_preprocessor


CATEGORICAL_COLUMNS = ["transport_mode", "cargo_type"]
NUMERIC_COLUMNS = [
    "distance_km",
    "weight_kg",
    "volume_m3",
    "urgent",
    "month",
]


def _make_df() -> pd.DataFrame:
    """Create small synthetic shipment data for the tests."""
    return pd.DataFrame(
        {
            "shipment_id": [101, 102, 103, 104],
            "transport_mode": ["Road", "Sea", "Air", "Road"],
            "cargo_type": [
                "General",
                "Hazardous",
                "Refrigerated",
                "General",
            ],
            "distance_km": [900, 9000, 4500, 1500],
            "weight_kg": [1200, 8000, 300, 500],
            "volume_m3": [5.0, 30.0, 2.5, 4.0],
            "urgent": [0, 0, 1, 0],
            "month": [3, 6, 12, 9],
            "cost_usd": [400.0, 8000.0, 2000.0, 600.0],
        }
    )


def test_unit_returns_unfitted_preprocessor() -> None:
    preprocessor = get_feature_preprocessor(
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )

    assert isinstance(preprocessor, ColumnTransformer)

    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)


@pytest.mark.parametrize(
    "categorical_columns,numeric_columns",
    [
        ([], ["distance_km"]),
        (["transport_mode"], []),
        ("transport_mode", ["distance_km"]),
        (["transport_mode"], "distance_km"),
        ([42], ["distance_km"]),
        (["transport_mode"], [""]),
        (["transport_mode", "transport_mode"], ["distance_km"]),
        (["distance_km"], ["distance_km"]),
    ],
)
def test_unit_rejects_invalid_column_groups(
    categorical_columns,
    numeric_columns,
) -> None:
    with pytest.raises(ValueError, match="column"):
        get_feature_preprocessor(
            categorical_columns,
            numeric_columns,
        )


def test_unit_preprocessing_preserves_values_and_excludes_extra_columns() -> None:
    data = _make_df()
    original = data.copy(deep=True)
    preprocessor = get_feature_preprocessor(
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )

    values = preprocessor.fit_transform(data)
    if hasattr(values, "toarray"):
        values = values.toarray()

    names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(values, columns=names, index=data.index)

    assert result.loc[0, "cat__transport_mode_Road"] == 1.0
    assert result.loc[0, "cat__transport_mode_Air"] == 0.0

    numeric_names = [f"num__{column}" for column in NUMERIC_COLUMNS]
    np.testing.assert_allclose(
        result[numeric_names].to_numpy(),
        data[NUMERIC_COLUMNS].to_numpy(),
    )

    assert all(
        "shipment_id" not in name and "cost_usd" not in name
        for name in names
    )
    pd.testing.assert_frame_equal(data, original)


def test_integration_fitted_recipe_transforms_new_shipments() -> None:
    training = _make_df()
    preprocessor = get_feature_preprocessor(
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )
    preprocessor.fit(training)

    new_shipments = training.iloc[[0]].copy()
    new_shipments["transport_mode"] = "Rail"

    values = preprocessor.transform(new_shipments)
    if hasattr(values, "toarray"):
        values = values.toarray()

    assert values.shape == (1, 11)
    assert not np.isnan(values).any()

    names = preprocessor.get_feature_names_out()
    mode_positions = [
        position
        for position, name in enumerate(names)
        if name.startswith("cat__transport_mode_")
    ]

    assert len(mode_positions) == 3
    np.testing.assert_array_equal(
        values[:, mode_positions],
        np.zeros((1, 3)),
    )
