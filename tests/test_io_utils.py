"""Unit tests for io_utils column detection and Excel I/O."""

import pandas as pd
import pytest

from warranty.io_utils import ColumnMap, detect_columns, load_column_aliases


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "VIN": ["A" * 17],
        "Repair Date": ["2024-06-01"],
        "Mileage": [25000],
        "Customer Complaint": ["Engine noise"],
        "Technician Comment": ["Replaced coil"],
        "Model Year": [2023],
        "Model": ["Sedan A"],
    })


@pytest.fixture
def aliases():
    return load_column_aliases()


def test_detect_standard_headers(sample_df, aliases):
    cmap = detect_columns(sample_df, aliases)
    assert cmap.vin == "VIN"
    assert cmap.repair_date == "Repair Date"
    assert cmap.customer_complaint == "Customer Complaint"
    assert cmap.tech_comment == "Technician Comment"
    assert cmap.model_year == "Model Year"


def test_detect_underscore_headers(aliases):
    df = pd.DataFrame({
        "vin_no": ["X" * 17],
        "repair_date": ["2024-01-01"],
        "tech_comment": ["Fixed it"],
    })
    cmap = detect_columns(df, aliases)
    assert cmap.vin == "vin_no"
    assert cmap.repair_date == "repair_date"
    assert cmap.tech_comment == "tech_comment"


def test_unresolved_roles(sample_df, aliases):
    cmap = detect_columns(sample_df, aliases)
    unresolved = cmap.unresolved_roles()
    # Many optional roles will be unresolved in a minimal df.
    assert "engine_type" in unresolved
    assert "vin" not in unresolved
