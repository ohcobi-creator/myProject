"""Unit tests for statistical analysis functions."""

import numpy as np
import pandas as pd
import pytest

from warranty.stats_trends import (
    _mann_kendall,
    frequency_analysis,
    pareto_analysis,
    select_variables,
)
from warranty.io_utils import ColumnMap


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "vin": [f"VIN{i:04d}" for i in range(50)],
        "model": np.random.choice(["Sedan A", "SUV B"], 50),
        "model_year": np.random.choice([2021, 2022, 2023], 50),
        "mileage_at_repair": np.random.randint(5000, 80000, 50),
        "repair_date": pd.date_range("2024-01-01", periods=50, freq="3D"),
        "in_service_date": pd.date_range("2022-01-01", periods=50, freq="7D"),
        "customer_complaint": ["complaint"] * 50,
        "tech_comment": ["tech note"] * 50,
        "repair_cost": np.random.uniform(100, 1500, 50).round(2),
        "labor_hours": np.random.uniform(0.5, 4.0, 50).round(1),
        "region": np.random.choice(["US-CA", "US-TX", "US-NY"], 50),
    })


@pytest.fixture
def col_map():
    return ColumnMap(
        vin="vin", model="model", model_year="model_year",
        mileage_at_repair="mileage_at_repair",
        repair_date="repair_date", in_service_date="in_service_date",
        customer_complaint="customer_complaint", tech_comment="tech_comment",
        repair_cost="repair_cost", labor_hours="labor_hours",
        region="region",
    )


def test_select_variables_roles(sample_df, col_map):
    report = select_variables(sample_df, col_map)
    input_names = [v.name for v in report.inputs]
    output_names = [v.name for v in report.outputs]
    excluded_names = [v.name for v in report.excluded]

    # VIN should be excluded (identifier).
    assert "vin" in excluded_names
    # Free-text should be excluded.
    assert "customer_complaint" in excluded_names
    assert "tech_comment" in excluded_names
    # Repair cost should be output (post-failure).
    assert "repair_cost" in output_names
    # Model should be input.
    assert "model" in input_names
    # Derived outputs should be present.
    assert any("is_defect" in v.name for v in report.outputs)
    assert any("days_to_failure" in v.name for v in report.outputs)


def test_frequency_analysis(sample_df):
    result_df, _, chart_path = frequency_analysis(sample_df, "model")
    assert len(result_df) == 2  # Sedan A, SUV B
    assert "ci_low" in result_df.columns
    assert "ci_high" in result_df.columns
    assert chart_path.endswith(".png")


def test_pareto_80_cutoff(sample_df):
    pareto_df, chart_path = pareto_analysis(sample_df, "region")
    assert "cum_pct" in pareto_df.columns
    # At least one row should reach >= 80%.
    assert (pareto_df["cum_pct"] >= 0.8).any()


def test_mann_kendall_no_trend():
    y = np.array([5, 5, 5, 5, 5], dtype=float)
    s, p = _mann_kendall(y)
    assert s == 0
    assert p > 0.05


def test_mann_kendall_clear_trend():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    s, p = _mann_kendall(y)
    assert s > 0
    assert p < 0.05
