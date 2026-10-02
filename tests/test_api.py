from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_requires_valid_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_returns_expected_fields():
    payload = {
        "loan_amnt": 15000,
        "int_rate": 12.5,
        "annual_inc": 60000,
        "dti": 15.0,
        "fico_range_low": 680,
        "fico_range_high": 684,
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "revol_util": 45.0,
        "total_acc": 20,
        "open_acc": 10,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "credit_score" in data
    assert "risk_band" in data
    assert "decision" in data
    assert "explanation" in data