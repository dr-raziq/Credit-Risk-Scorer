import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "revol_util",
    "total_acc", "open_acc", "delinq_2yrs", "inq_last_6mths"
]

CATEGORICAL_FEATURES = [
    "emp_length", "home_ownership", "purpose"
]

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_names + cat_names