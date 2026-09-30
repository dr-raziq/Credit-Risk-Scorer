def probability_to_score(probability: float) -> int:
    """
    Map default probability (0-1) to a credit score between 300 and 850.
    Higher probability of default -> lower score.
    """
    min_score = 300
    max_score = 850
    score = max_score - (max_score - min_score) * probability
    return int(round(score))

def score_to_risk_band(score: int) -> str:
    if score >= 750:
        return "Very Low Risk"
    elif score >= 700:
        return "Low Risk"
    elif score >= 650:
        return "Medium Risk"
    elif score >= 600:
        return "High Risk"
    else:
        return "Very High Risk"

def get_decision(score: int) -> str:
    if score >= 650:
        return "Approve"
    elif score >= 600:
        return "Manual Review"
    else:
        return "Decline"