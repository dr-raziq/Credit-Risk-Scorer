import os
import joblib
import numpy as np
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import LoanApplication, PredictionResponse
from src.feature_engineer import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    get_feature_names,
)
from src.scoring import (
    probability_to_score,
    score_to_risk_band,
    get_decision,
)
from src.explain import get_shap_explainer, explain_single


MODEL_PATH = "models/xgb_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
FRONTEND_DIR = "frontend"

app = FastAPI(
    title="Credit Risk Scorer",
    version="1.0.0",
    description=(
        "Explainable credit risk scoring API. Returns a default probability, "
        "a credit score between 300 and 850, a risk band, an automated decision, "
        "and a SHAP-based local explanation for each prediction."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessor = None
explainer = None
feature_names = None


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, explainer, feature_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model file not found at "
            + MODEL_PATH
            + ". Run: python -m src.train_model"
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(
            "Preprocessor file not found at "
            + PREPROCESSOR_PATH
            + ". Run: python -m src.train_model"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    explainer = get_shap_explainer(model)
    feature_names = get_feature_names(preprocessor)

    print("Model loaded. Feature count:", len(feature_names))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check the server startup logs.",
        )

    try:
        # Pydantic v2 uses model_dump, v1 uses dict. Support both.
        if hasattr(application, "model_dump"):
            payload = application.model_dump()
        else:
            payload = application.dict()

        input_df = pd.DataFrame([payload])
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
            explanation=explanation,
        )

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail="Missing or unexpected feature in payload: " + str(e),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend from the same origin to avoid CORS issues.
if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))