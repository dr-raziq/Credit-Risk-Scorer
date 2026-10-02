from src.scoring import probability_to_score, score_to_risk_band, get_decision

def test_probability_to_score_bounds():
    assert probability_to_score(0.0) == 850
    assert probability_to_score(1.0) == 300

def test_probability_to_score_monotonic():
    assert probability_to_score(0.1) > probability_to_score(0.9)

def test_score_to_risk_band():
    assert score_to_risk_band(800) == "Very Low Risk"
    assert score_to_risk_band(720) == "Low Risk"
    assert score_to_risk_band(670) == "Medium Risk"
    assert score_to_risk_band(620) == "High Risk"
    assert score_to_risk_band(500) == "Very High Risk"

def test_get_decision():
    assert get_decision(700) == "Approve"
    assert get_decision(620) == "Manual Review"
    assert get_decision(550) == "Decline"