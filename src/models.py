import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import (
    build_preprocessor,
    prepare_train_test_data,
    get_feature_names,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate key classification performance metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate model pipelines covering linear, bagging, and boosting paradigms."""
    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Logistic Regression (Balanced)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
            )),
        ]),
        "Random Forest (Balanced)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=120, max_depth=6, min_samples_leaf=20, random_state=42
            )),
        ]),
        "Logistic Regression (SMOTE)": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
    return models


def train_and_evaluate_all(
    data_path: str = "data/processed/telco_churn_cleaned.csv",
    save_dir: str = "models",
) -> Tuple[str, Dict[str, Any]]:
    """
    Train and evaluate candidate models using both 5-fold CV and held-out test evaluation.
    Selects the champion model based on ROC-AUC and business suitability.
    """
    X_train, X_test, y_train, y_test, meta = prepare_train_test_data(data_path)
    models = get_candidate_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}
    curves_data = {}
    best_model_name = None
    best_roc_auc = -1.0

    print("--- Starting Model Benchmarking (5-Fold CV & Held-Out Test) ---")

    for name, pipeline in models.items():
        print(f"Training and evaluating: {name}...")

        # 5-Fold Cross Validation on Train Set
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        cv_summary = {
            f"cv_{metric}": round(float(np.mean(cv_scores[f"test_{metric}"])), 4)
            for metric in scoring
        }

        # Fit on entire Train Set and evaluate on Test Set
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_predictions(y_test.values, y_test_pred, y_test_prob)
        cm = confusion_matrix(y_test, y_test_pred).tolist()

        # Compute ROC and PR curve points for plotting
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_test_prob)

        curves_data[name] = {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]],
            "precision": [round(float(x), 4) for x in prec[::max(1, len(prec) // 50)]],
            "recall": [round(float(x), 4) for x in rec[::max(1, len(rec) // 50)]],
        }

        results[name] = {
            "cross_validation": cv_summary,
            "test_metrics": test_metrics,
            "confusion_matrix": cm,
        }

        if test_metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = test_metrics["roc_auc"]
            best_model_name = name

    # Save artifacts
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    champion_pipeline = models[best_model_name]
    champion_path = save_path / "best_model.joblib"
    joblib.dump(champion_pipeline, champion_path)

    # Save test dataset predictions for business impact modeling
    y_test_prob = champion_pipeline.predict_proba(X_test)[:, 1]
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Churn"] = y_test.values
    test_eval_df["Predicted_Prob"] = y_test_prob
    test_eval_df["Predicted_Class"] = (y_test_prob >= 0.5).astype(int)
    test_eval_df.to_csv(save_path / "test_predictions.csv", index=False)

    comparison_report = {
        "best_model_name": best_model_name,
        "best_roc_auc": best_roc_auc,
        "metadata": meta,
        "results": results,
        "curves_data": curves_data,
    }

    with open(save_path / "model_comparison.json", "w") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"\nChampion Model Selected: '{best_model_name}' with Test ROC-AUC = {best_roc_auc:.4f}")
    return best_model_name, comparison_report


if __name__ == "__main__":
    best_name, rep = train_and_evaluate_all()
