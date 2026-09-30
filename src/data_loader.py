import pandas as pd

REQUIRED_COLUMNS = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "emp_length",
    "home_ownership", "purpose", "revol_util",
    "total_acc", "open_acc", "delinq_2yrs", "inq_last_6mths",
    "loan_status"
]

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df

def filter_definitive_statuses(df: pd.DataFrame) -> pd.DataFrame:
    valid = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(valid)].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[REQUIRED_COLUMNS + ["target"]].copy()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["target"])
    numeric_cols = ["loan_amnt", "int_rate", "annual_inc", "dti",
                    "fico_range_low", "fico_range_high", "revol_util",
                    "total_acc", "open_acc", "delinq_2yrs", "inq_last_6mths"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    return df

def load_and_prepare(path: str) -> pd.DataFrame:
    df = load_raw_data(path)
    df = filter_definitive_statuses(df)
    df = select_features(df)
    df = clean_data(df)
    return df