"""Unit tests for business impact, deciles, and retention ROI simulation."""

import pandas as pd
import pytest
from src.business_impact import calculate_decile_table, simulate_retention_campaign


def test_decile_table_properties():
    df_preds = pd.read_csv("models/test_predictions.csv")
    deciles = calculate_decile_table(df_preds)

    assert len(deciles) == 10
    # Decile 1 should have substantially higher churn rate than base
    assert deciles.iloc[0]["churn_rate"] > 0.60
    assert deciles.iloc[0]["decile_lift"] > 2.0
    # Cumulative gain at decile 10 must be 100%
    assert abs(deciles.iloc[-1]["cumulative_gain_pct"] - 100.0) < 0.1


def test_simulate_retention_roi():
    df_preds = pd.read_csv("models/test_predictions.csv")
    sim = simulate_retention_campaign(df_preds, target_top_pct=0.20, offer_cost=40.0, save_rate=0.30)

    assert sim["customers_targeted"] == int(len(df_preds) * 0.20)
    assert sim["churners_captured_pct"] > 45.0
    assert sim["net_profit"] > 0
    assert sim["roi_pct"] > 100.0
