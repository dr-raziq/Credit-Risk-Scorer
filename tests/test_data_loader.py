import pandas as pd
import pytest
from src.data_loader import filter_definitive_statuses, select_features, clean_data

def test_filter_definitive_statuses():
    df = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Current", "Late"],
        "other": [1, 2, 3, 4]
    })
    result = filter_definitive_statuses(df)
    assert len(result) == 2
    assert set(result["target"]) == {0, 1}

def test_select_features_missing_column():
    df = pd.DataFrame({"loan_amnt": [1000]})
    with pytest.raises(ValueError):
        select_features(df)

def test_clean_data_removes_nan_target():
    df = pd.DataFrame({
        "loan_amnt": [1000, 2000],
        "int_rate": [10.0, 12.0],
        "annual_inc": [50000, 60000],
        "dti": [10.0, 15.0],
        "fico_range_low": [680, 700],
        "fico_range_high": [684, 704],
        "emp_length": ["5 years", "10+ years"],
        "home_ownership": ["RENT", "MORTGAGE"],
        "purpose": ["debt_consolidation", "credit_card"],
        "revol_util": [30.0, 40.0],
        "total_acc": [10, 20],
        "open_acc": [5, 10],
        "delinq_2yrs": [0, 0],
        "inq_last_6mths": [1, 0],
        "target": [0, None]
    })
    result = clean_data(df)
    assert len(result) == 1