"""Feature preprocessing pipeline module for Customer Churn modeling."""

from typing import Tuple, List, Dict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]

TARGET_COL = "Churn_Binary"


def build_preprocessor() -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve full list of transformed feature names after fitting ColumnTransformer."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERICAL_FEATURES + cat_feature_names


def prepare_train_test_data(
    data_path: str = "data/processed/telco_churn_cleaned.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, any]]:
    """
    Load cleaned dataset and produce a stratified train/test split.
    Guarantees no data leakage by separating before fitting any transformers.
    """
    df = pd.read_csv(data_path)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "feature_count_raw": len(feature_cols),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    return X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = prepare_train_test_data()
    print("Train-Test Split Summary:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    prep = build_preprocessor()
    X_train_trans = prep.fit_transform(X_train)
    feat_names = get_feature_names(prep)
    print(f"Transformed X_train shape: {X_train_trans.shape}")
    print(f"Number of encoded features: {len(feat_names)}")
