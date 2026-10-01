import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import LoanApplication, PredictionResponse
from src.feature_engineer import NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_feature_names
from src.scoring import probability_to_score, score_to_risk_band, get_decision
from src.explain import get_shap_explainer, explain_single

app = FastAPI(title="Credit Risk Scorer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/xgb_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None
explainer = None
feature_names = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, explainer, feature_names
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    explainer = get_shap_explainer(model)
    feature_names = get_feature_names(preprocessor)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        input_df = pd.DataFrame([application.dict()])
        X = input_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_processed = preprocessor.transform(X)

        proba = float(model.predict_proba(X_processed)[0, 1])
        score = probability_to_score(proba)
        risk_band = score_to_risk_band(score)
        decision = get_decision(score)

        explanation = explain_single(
            explainer, X_processed, feature_names, top_n=5
        )

        return PredictionResponse(
            default_probability=round(proba, 4),
            credit_score=score,
            risk_band=risk_band,
            decision=decision,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))