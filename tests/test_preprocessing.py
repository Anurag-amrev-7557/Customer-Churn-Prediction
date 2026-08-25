"""Unit tests for preprocessing and feature encoding."""

import numpy as np
import pytest
from src.preprocessing import (
    build_preprocessor,
    prepare_train_test_data,
    get_feature_names,
)


def test_prepare_train_test_data_stratification():
    X_train, X_test, y_train, y_test, meta = prepare_train_test_data()
    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert abs(float(y_train.mean()) - float(y_test.mean())) < 0.01


def test_preprocessor_transformation_shape_and_nans():
    X_train, X_test, y_train, y_test, _ = prepare_train_test_data()
    prep = build_preprocessor()
    X_train_trans = prep.fit_transform(X_train)
    X_test_trans = prep.transform(X_test)

    assert X_train_trans.shape[0] == 5634
    assert X_test_trans.shape[0] == 1409
    assert X_train_trans.shape[1] == 46
    assert np.isnan(X_train_trans).sum() == 0
    assert np.isnan(X_test_trans).sum() == 0


def test_feature_names_length():
    X_train, _, _, _, _ = prepare_train_test_data()
    prep = build_preprocessor()
    prep.fit(X_train)
    names = get_feature_names(prep)
    assert len(names) == 46
    assert "tenure" in names
    assert "MonthlyCharges" in names
    assert "TotalCharges" in names
