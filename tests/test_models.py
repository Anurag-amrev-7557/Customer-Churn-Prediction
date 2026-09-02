"""Unit tests for model loading and evaluation."""

from pathlib import Path
import pytest
import joblib
import numpy as np
from src.preprocessing import prepare_train_test_data


def test_saved_model_exists_and_predicts():
    model_path = Path("models/best_model.joblib")
    assert model_path.exists()

    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    assert len(preds) == len(X_test)
    assert len(probs) == len(X_test)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_roc_auc_benchmark():
    from sklearn.metrics import roc_auc_score
    model = joblib.load("models/best_model.joblib")
    _, X_test, _, y_test, _ = prepare_train_test_data()

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    assert auc >= 0.80, f"Expected ROC-AUC >= 0.80, got {auc:.4f}"
