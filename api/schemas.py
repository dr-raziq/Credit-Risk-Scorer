from pydantic import BaseModel, Field

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, description="Loan amount")
    int_rate: float = Field(..., gt=0, description="Interest rate percentage")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    dti: float = Field(..., ge=0, description="Debt-to-income ratio")
    fico_range_low: int = Field(..., ge=300, le=850)
    fico_range_high: int = Field(..., ge=300, le=850)
    emp_length: str = Field(..., description="Employment length")
    home_ownership: str = Field(..., description="Home ownership status")
    purpose: str = Field(..., description="Loan purpose")
    revol_util: float = Field(..., ge=0, description="Revolving line utilisation rate")
    total_acc: int = Field(..., ge=0, description="Total accounts")
    open_acc: int = Field(..., ge=0, description="Open accounts")
    delinq_2yrs: int = Field(..., ge=0, description="Delinquencies in last 2 years")
    inq_last_6mths: int = Field(..., ge=0, description="Inquiries in last 6 months")

class PredictionResponse(BaseModel):
    default_probability: float
    credit_score: int
    risk_band: str
    decision: str
    explanation: dict