"""Business Impact, Decile Lift, and Customer Retention Prioritization module."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_decile_table(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Sort customers by predicted churn probability and compute decile lift and gains.
    Expects df_preds to contain 'Actual_Churn' and 'Predicted_Prob'.
    """
    df = df_preds.sort_values(by="Predicted_Prob", ascending=False).reset_index(drop=True).copy()
    total_customers = len(df)
    total_churners = df["Actual_Churn"].sum()
    base_churn_rate = total_churners / total_customers

    # Assign 10 deciles
    df["decile"] = pd.qcut(df.index, q=10, labels=[f"Decile {i}" for i in range(1, 11)])

    decile_summary = []
    cum_customers = 0
    cum_churners = 0

    for decile, group in df.groupby("decile", observed=False):
        n_cust = len(group)
        n_churn = int(group["Actual_Churn"].sum())
        churn_rate = n_churn / n_cust

        cum_customers += n_cust
        cum_churners += n_churn

        cum_gains_pct = (cum_churners / total_churners) * 100
        decile_lift = churn_rate / base_churn_rate
        cum_lift = (cum_churners / cum_customers) / base_churn_rate

        decile_summary.append({
            "decile": decile,
            "customers": n_cust,
            "churners": n_churn,
            "churn_rate": round(churn_rate, 4),
            "cum_customers": cum_customers,
            "cum_churners": cum_churners,
            "decile_lift": round(decile_lift, 2),
            "cumulative_lift": round(cum_lift, 2),
            "cumulative_gain_pct": round(cum_gains_pct, 2),
            "avg_monthly_charges": round(float(group["MonthlyCharges"].mean()), 2),
        })

    return pd.DataFrame(decile_summary)


def simulate_retention_campaign(
    df_preds: pd.DataFrame,
    target_top_pct: float = 0.20,
    offer_cost: float = 40.0,
    save_rate: float = 0.30,
    months_horizon: int = 12,
) -> Dict[str, Any]:
    """
    Simulate financial outcome of targeting the top X% highest-risk customers.
    """
    df = df_preds.sort_values(by="Predicted_Prob", ascending=False).reset_index(drop=True)
    n_target = int(len(df) * target_top_pct)
    targeted_group = df.iloc[:n_target]

    actual_churners_targeted = targeted_group["Actual_Churn"].sum()
    total_churners = df["Actual_Churn"].sum()
    churners_captured_pct = (actual_churners_targeted / total_churners) * 100

    # Financial calculations
    # Saved revenue: churners who accept the offer continue paying their monthly charge for horizon
    saved_monthly_revenue = (
        targeted_group[targeted_group["Actual_Churn"] == 1]["MonthlyCharges"].sum() * save_rate
    )
    saved_annual_revenue = saved_monthly_revenue * months_horizon

    campaign_cost = n_target * offer_cost
    net_profit = saved_annual_revenue - campaign_cost
    roi_pct = (net_profit / campaign_cost) * 100 if campaign_cost > 0 else 0.0

    # Baseline comparison: Random Outreach to same number of customers
    random_churners_targeted = (total_churners / len(df)) * n_target
    random_saved_annual = (
        random_churners_targeted * df["MonthlyCharges"].mean() * save_rate * months_horizon
    )
    random_net_profit = random_saved_annual - campaign_cost

    return {
        "target_fraction": target_top_pct,
        "customers_targeted": n_target,
        "total_test_customers": len(df),
        "actual_churners_captured": int(actual_churners_targeted),
        "total_churners": int(total_churners),
        "churners_captured_pct": round(float(churners_captured_pct), 2),
        "campaign_cost": round(float(campaign_cost), 2),
        "gross_revenue_saved": round(float(saved_annual_revenue), 2),
        "net_profit": round(float(net_profit), 2),
        "roi_pct": round(float(roi_pct), 2),
        "random_net_profit": round(float(random_net_profit), 2),
        "lift_over_random": round(float(net_profit - random_net_profit), 2),
    }


def generate_business_figures(df_deciles: pd.DataFrame, figures_dir: str = "reports/figures") -> None:
    """Generate Cumulative Gains and Lift Charts."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Cumulative Gains Chart
    pct_targeted = np.arange(10, 110, 10)
    gains = df_deciles["cumulative_gain_pct"].values

    axes[0].plot(pct_targeted, gains, marker="o", color="#2b5c8f", linewidth=2.5, label="Model (Random Forest)")
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random Baseline")
    axes[0].set_title("Cumulative Gains Chart (Retention Capture)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Percentage of Customers Contacted (%)")
    axes[0].set_ylabel("Percentage of Total Churners Captured (%)")
    axes[0].set_xlim(0, 105)
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right")

    # Annotate Top 20%
    top20_gain = gains[1]
    axes[0].annotate(
        f"Top 20% contacts\ncaptures {top20_gain:.1f}% churners",
        xy=(20, top20_gain),
        xytext=(35, top20_gain - 15),
        arrowprops=dict(facecolor="#d95f02", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#d95f02",
    )

    # 2. Decile Lift Chart
    axes[1].bar(df_deciles["decile"], df_deciles["decile_lift"], color="#1b9e77", width=0.55)
    axes[1].axhline(1.0, color="red", linestyle="--", label="Baseline Lift (1.0x)")
    axes[1].set_title("Lift by Decile", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Lift Factor (x)")
    axes[1].set_xticks(range(len(df_deciles)))
    axes[1].set_xticklabels(df_deciles["decile"], rotation=45, ha="right")
    axes[1].legend()

    for i, v in enumerate(df_deciles["decile_lift"]):
        axes[1].text(i, v + 0.05, f"{v:.1f}x", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "cumulative_gains_and_lift.png", dpi=200)
    plt.close(fig)


def run_business_impact_pipeline(
    predictions_path: str = "models/test_predictions.csv",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute retention business impact analysis and save decile and ROI reports."""
    df_preds = pd.read_csv(predictions_path)
    df_deciles = calculate_decile_table(df_preds)

    generate_business_figures(df_deciles)

    # Simulate standard campaigns at top 10%, 20%, 30%
    campaign_20 = simulate_retention_campaign(df_preds, target_top_pct=0.20, offer_cost=40.0, save_rate=0.30)
    campaign_10 = simulate_retention_campaign(df_preds, target_top_pct=0.10, offer_cost=40.0, save_rate=0.30)
    campaign_30 = simulate_retention_campaign(df_preds, target_top_pct=0.30, offer_cost=40.0, save_rate=0.30)

    # Playbook recommendations
    playbooks = [
        {
            "tier": "Tier 1: Critical Flight Risk (Deciles 1-2, Probability > 60%)",
            "profile": "Month-to-month contracts, fiber optic, high monthly charges ($75+), tenure < 12m",
            "action": "Immediate proactive outreach by senior retention specialist; offer 1-year contract lock-in with 20% discount and free tech support trial.",
            "target_save_rate": "30-35%",
        },
        {
            "tier": "Tier 2: Moderate Risk (Deciles 3-4, Probability 40-60%)",
            "profile": "Electronic check payers, lacking security/backup add-ons, moderate tenure",
            "action": "Automated in-app/email campaign offering $5/month autopay discount (transitioning from electronic check to credit card/bank transfer) and complimentary online backup.",
            "target_save_rate": "20-25%",
        },
        {
            "tier": "Tier 3: Low Risk / Healthy Accounts (Deciles 5-10, Probability < 40%)",
            "profile": "Longer tenure (> 24m), 1-2 year contracts, bundled services",
            "action": "Do NOT discount or contact with retention offers (avoids margin cannibalization). Focus on loyalty rewards and service upgrade cross-selling.",
            "target_save_rate": "Organic retention (>95%)",
        },
    ]

    summary = {
        "decile_table": df_deciles.to_dict(orient="records"),
        "campaign_simulation_top_20pct": campaign_20,
        "campaign_simulation_top_10pct": campaign_10,
        "campaign_simulation_top_30pct": campaign_30,
        "retention_playbooks": playbooks,
    }

    out_file = Path(reports_dir) / "retention_strategy_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    df_deciles.to_csv(Path(reports_dir) / "decile_analysis.csv", index=False)

    return summary


if __name__ == "__main__":
    summary = run_business_impact_pipeline()
    print("Business Impact analysis completed.")
    c20 = summary["campaign_simulation_top_20pct"]
    print(f"Top 20% Targeting captures {c20['churners_captured_pct']}% of all churners.")
    print(f"Gross Revenue Saved: ${c20['gross_revenue_saved']:,.2f} | Campaign Cost: ${c20['campaign_cost']:,.2f}")
    print(f"Net Profit: ${c20['net_profit']:,.2f} | ROI: {c20['roi_pct']:.1f}%")
    print(f"Lift over Random Contact: +${c20['lift_over_random']:,.2f}")
