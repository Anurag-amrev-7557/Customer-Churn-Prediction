"""Unit tests for data loading and cleaning."""

import pytest
import pandas as pd
from src.data_loader import load_and_clean_data, validate_raw_data


def test_data_loader_shape_and_columns():
    df, report = load_and_clean_data()
    assert len(df) == 7043
    assert "Churn_Binary" in df.columns
    assert "TotalCharges" in df.columns
    assert df["TotalCharges"].dtype in ["float64", "float32"]


def test_total_charges_nulls_handled():
    df, report = load_and_clean_data()
    assert df["TotalCharges"].isna().sum() == 0


def test_zero_tenure_customers_charges():
    df, report = load_and_clean_data()
    zero_tenure = df[df["tenure"] == 0]
    assert len(zero_tenure) > 0
    assert (zero_tenure["TotalCharges"] == 0.0).all()


def test_churn_binary_values():
    df, report = load_and_clean_data()
    unique_vals = set(df["Churn_Binary"].unique())
    assert unique_vals == {0, 1}
    assert 0.25 < df["Churn_Binary"].mean() < 0.28
