"""Data loader and validation module for IBM Telco Customer Churn dataset."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_raw_data(df: pd.DataFrame) -> Dict[str, any]:
    """Validate raw dataframe structure and return a diagnostic report."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    blank_total_charges = (
        (df["TotalCharges"].astype(str).str.strip() == "")
        | (df["TotalCharges"].astype(str).str.lower() == "nan")
    ).sum()

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated(subset=["customerID"]).sum()),
        "blank_total_charges_count": int(blank_total_charges),
        "churn_values": df["Churn"].dropna().unique().tolist(),
    }
    return report


def load_and_clean_data(
    raw_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    save_processed_path: str = "data/processed/telco_churn_cleaned.csv",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Loads raw CSV, validates, fixes whitespace/TotalCharges,
    creates clean binary target Churn_Binary, and optionally caches clean data.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Please place "
            f"WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/."
        )

    df = pd.read_csv(path)
    report = validate_raw_data(df)

    # Clean TotalCharges: convert whitespace to NaN, then impute tenure==0 with 0.0
    total_charges_raw = df["TotalCharges"].astype(str).str.strip()
    total_charges_numeric = pd.to_numeric(total_charges_raw, errors="coerce")

    # For customers with tenure == 0, TotalCharges is 0.0
    zero_tenure_mask = (df["tenure"] == 0) & (total_charges_numeric.isna())
    total_charges_numeric = total_charges_numeric.mask(zero_tenure_mask, 0.0)

    # In case there are any other remaining NaNs, fill with median
    if total_charges_numeric.isna().sum() > 0:
        median_val = total_charges_numeric.median()
        total_charges_numeric = total_charges_numeric.fillna(median_val)

    df_clean = df.copy()
    df_clean["TotalCharges"] = total_charges_numeric
    df_clean["Churn_Binary"] = (df_clean["Churn"].str.strip().str.lower() == "yes").astype(int)

    # Add tenure groups for segmentation clarity
    df_clean["TenureCohort"] = pd.cut(
        df_clean["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )

    report["clean_rows"] = len(df_clean)
    report["churn_rate"] = float(df_clean["Churn_Binary"].mean())

    if save_processed_path:
        out_path = Path(save_processed_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out_path, index=False)

    return df_clean, report


if __name__ == "__main__":
    df, rep = load_and_clean_data()
    print("Data successfully loaded & cleaned:")
    for k, v in rep.items():
        print(f"  {k}: {v}")
