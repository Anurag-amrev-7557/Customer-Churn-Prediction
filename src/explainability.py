"""Model Explainability and Feature Attribution module."""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib

from src.preprocessing import prepare_train_test_data, get_feature_names


def explain_logistic_regression(lr_pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Extract coefficients and odds ratios from fitted Logistic Regression pipeline."""
    classifier = lr_pipeline.named_steps["classifier"]
    coefs = classifier.coef_[0]
    odds_ratios = np.exp(coefs)

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": odds_ratios,
        "impact": ["Increases Churn" if c > 0 else "Reduces Churn" for c in coefs],
    })
    df_coef["abs_coef"] = df_coef["coefficient"].abs()
    return df_coef.sort_values(by="abs_coef", ascending=False).drop(columns=["abs_coef"])


def explain_tree_model(tree_pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Extract Gini feature importance from fitted Random Forest pipeline."""
    classifier = tree_pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)
    return df_imp


def compute_permutation_importance(pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    """Compute model-agnostic permutation importance on held-out test set."""
    perm_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    df_perm = pd.DataFrame({
        "feature_raw": X_test.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values(by="importance_mean", ascending=False)
    return df_perm


def run_explainability_analysis(
    data_path: str = "data/processed/telco_churn_cleaned.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute complete explainability workflow: odds ratios, Gini importance, and permutation importance."""
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data(data_path)

    # Train a baseline Logistic Regression for interpretable odds ratios
    from src.models import get_candidate_models
    candidates = get_candidate_models()
    lr_pipe = candidates["Logistic Regression (Baseline)"]
    lr_pipe.fit(X_train, y_train)

    rf_pipe = candidates["Random Forest"]
    rf_pipe.fit(X_train, y_train)

    preprocessor = lr_pipe.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    # 1. Odds Ratios (Logistic Regression)
    lr_explanation = explain_logistic_regression(lr_pipe, feature_names)

    # 2. Random Forest Importance
    rf_importance = explain_tree_model(rf_pipe, feature_names)

    # 3. Permutation Importance on Raw Features
    perm_importance = compute_permutation_importance(rf_pipe, X_test, y_test, feature_names)

    # Generate Explainability Visualizations
    fig_dir = Path(reports_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Top 15 Logistic Regression Coefficients (Risk vs Protective)
    top_lr = pd.concat([
        lr_explanation[lr_explanation["coefficient"] > 0].head(8),
        lr_explanation[lr_explanation["coefficient"] < 0].head(8),
    ]).sort_values(by="coefficient")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2b5c8f" if c < 0 else "#d95f02" for c in top_lr["coefficient"]]
    bars = ax.barh(top_lr["feature"], top_lr["coefficient"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Top Customer Churn Drivers & Protective Factors (Logistic Regression Log-Odds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient (Positive = Higher Risk, Negative = Lower Risk)")
    plt.tight_layout()
    fig.savefig(fig_dir / "logistic_regression_coefficients.png", dpi=200)
    plt.close(fig)

    # Plot 2: Random Forest Top 15 Feature Importances
    fig, ax = plt.subplots(figsize=(10, 6))
    top_rf = rf_importance.head(15).sort_values(by="importance")
    ax.barh(top_rf["feature"], top_rf["importance"], color="#7570b3")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gini Importance")
    plt.tight_layout()
    fig.savefig(fig_dir / "random_forest_feature_importances.png", dpi=200)
    plt.close(fig)

    # Plot 3: Permutation Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_perm = perm_importance.head(12).sort_values(by="importance_mean")
    ax.barh(top_perm["feature_raw"], top_perm["importance_mean"], xerr=top_perm["importance_std"], color="#1b9e77")
    ax.set_title("Permutation Importance on Test Set (Drop in ROC-AUC)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean ROC-AUC Decrease")
    plt.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.png", dpi=200)
    plt.close(fig)

    # Prepare summary dictionary
    summary = {
        "top_churn_risk_drivers": lr_explanation[lr_explanation["coefficient"] > 0].head(10).to_dict(orient="records"),
        "top_churn_protective_factors": lr_explanation[lr_explanation["coefficient"] < 0].head(10).to_dict(orient="records"),
        "top_rf_features": rf_importance.head(15).to_dict(orient="records"),
        "top_permutation_features": perm_importance.head(10).to_dict(orient="records"),
    }

    out_file = Path(reports_dir) / "explainability_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_explainability_analysis()
    print("Explainability analysis completed successfully.")
    print("\nTop 5 Risk Drivers (Increases Churn Odds):")
    for r in summary["top_churn_risk_drivers"][:5]:
        print(f"  {r['feature']}: Odds Ratio = {r['odds_ratio']:.3f} (+{((r['odds_ratio']-1)*100):.1f}% risk)")
    print("\nTop 5 Protective Factors (Reduces Churn Odds):")
    for r in summary["top_churn_protective_factors"][:5]:
        print(f"  {r['feature']}: Odds Ratio = {r['odds_ratio']:.3f} (-{((1-r['odds_ratio'])*100):.1f}% risk)")
