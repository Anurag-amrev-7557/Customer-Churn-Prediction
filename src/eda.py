"""Exploratory Data Analysis (EDA) and Segment Profiling module."""

from pathlib import Path
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def calculate_cramers_v(cat_series: pd.Series, target_series: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical association with target."""
    confusion_matrix = pd.crosstab(cat_series, target_series)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    r_corr = max(0, r - ((r - 1) ** 2) / (n - 1))
    k_corr = max(0, k - ((k - 1) ** 2) / (n - 1))
    denom = min(k_corr - 1, r_corr - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt(phi2 / denom))


def compute_segment_profiles(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate churn rates across key demographic, account, and service segments."""
    segments = {}

    cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TenureCohort",
        "SeniorCitizen",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
        "Partner",
        "Dependents",
    ]

    for col in cat_cols:
        grouped = df.groupby(col).agg(
            total_customers=("customerID", "count"),
            churned_customers=("Churn_Binary", "sum"),
            churn_rate=("Churn_Binary", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_tenure=("tenure", "mean"),
        ).reset_index()
        grouped["churn_rate"] = grouped["churn_rate"].round(4)
        grouped["avg_monthly_charges"] = grouped["avg_monthly_charges"].round(2)
        grouped["avg_tenure"] = grouped["avg_tenure"].round(1)
        segments[col] = grouped.to_dict(orient="records")

    return segments


def compute_associations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute associations for numeric (Point-Biserial) and categorical (Cramér's V) variables."""
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    num_assoc = {}
    for col in numeric_cols:
        corr, pval = stats.pointbiserialr(df[col], df["Churn_Binary"])
        num_assoc[col] = {"correlation": round(float(corr), 4), "p_value": float(pval)}

    cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "DeviceProtection",
        "OnlineBackup",
        "MultipleLines",
        "PhoneService",
        "gender",
    ]
    cramers_dict = {}
    for col in cat_cols:
        v = calculate_cramers_v(df[col], df["Churn_Binary"])
        cramers_dict[col] = round(v, 4)

    # Sort descending
    cramers_sorted = dict(sorted(cramers_dict.items(), key=lambda x: x[1], reverse=True))

    return {"numeric_point_biserial": num_assoc, "categorical_cramers_v": cramers_sorted}


def generate_eda_figures(df: pd.DataFrame, figures_dir: str = "reports/figures") -> None:
    """Generate and save publication-grade EDA figures."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"No": "#2b5c8f", "Yes": "#d95f02"}

    # 1. Churn Distribution & Tenure Cohort Churn
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    churn_counts = df["Churn"].value_counts(normalize=True) * 100
    bars = axes[0].bar(churn_counts.index, churn_counts.values, color=["#2b5c8f", "#d95f02"], width=0.5)
    axes[0].set_title("Overall Churn Rate (IBM Telco)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Percentage (%)")
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    tenure_churn = df.groupby("TenureCohort", observed=False)["Churn_Binary"].mean() * 100
    axes[1].bar(tenure_churn.index.astype(str), tenure_churn.values, color="#e7298a", width=0.5)
    axes[1].set_title("Churn Rate by Tenure Cohort", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, 60)
    for i, v in enumerate(tenure_churn.values):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / "churn_distribution_and_tenure.png", dpi=200)
    plt.close(fig)

    # 2. Churn by Contract and Internet Service
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    contract_churn = df.groupby(["Contract", "Churn"], observed=False).size().unstack(fill_value=0)
    contract_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
    contract_pct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d95f02"], ax=axes[0], rot=0)
    axes[0].set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].legend(["Retained (No)", "Churned (Yes)"], loc="upper right")

    internet_churn = df.groupby(["InternetService", "Churn"], observed=False).size().unstack(fill_value=0)
    internet_pct = internet_churn.div(internet_churn.sum(axis=1), axis=0) * 100
    internet_pct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d95f02"], ax=axes[1], rot=0)
    axes[1].set_title("Churn Rate by Internet Service", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(["Retained (No)", "Churned (Yes)"], loc="upper right")
    plt.tight_layout()
    fig.savefig(fig_dir / "churn_by_contract_and_internet.png", dpi=200)
    plt.close(fig)

    # 3. Monthly Charges Distribution by Churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True,
                palette=palette, alpha=0.4, linewidth=2, ax=ax)
    ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    plt.tight_layout()
    fig.savefig(fig_dir / "monthly_charges_distribution.png", dpi=200)
    plt.close(fig)

    # 4. Association Ranking (Cramér's V)
    assoc = compute_associations(df)["categorical_cramers_v"]
    s = pd.Series(assoc)
    fig, ax = plt.subplots(figsize=(9, 6))
    s.plot(kind="barh", color="#1b9e77", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Categorical Characteristics Association with Churn (Cramér's V)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cramér's V Statistic (0 = No Association, 1 = Perfect Association)")
    for i, v in enumerate(s.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "cramers_v_association.png", dpi=200)
    plt.close(fig)


def run_full_eda(
    data_path: str = "data/processed/telco_churn_cleaned.csv",
    out_json_path: str = "reports/eda_summary.json",
) -> Dict[str, Any]:
    """Run full EDA pipeline, compute stats, export summary json, and plot charts."""
    df = pd.read_csv(data_path)
    segments = compute_segment_profiles(df)
    associations = compute_associations(df)
    generate_eda_figures(df)

    summary = {
        "dataset_rows": len(df),
        "overall_churn_rate": round(float(df["Churn_Binary"].mean()), 4),
        "segment_profiles": segments,
        "associations": associations,
    }

    out_file = Path(out_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_full_eda()
    print("EDA completed successfully. Associations summary:")
    print("Numeric correlations:", summary["associations"]["numeric_point_biserial"])
    print("Top Categorical Cramér's V:", list(summary["associations"]["categorical_cramers_v"].items())[:5])
