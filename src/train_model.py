import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from src.data_loader import load_and_prepare
from src.feature_engineer import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

DATA_PATH = "data/loan.csv"
MODEL_PATH = "models/xgb_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def train():
    df = load_and_prepare(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_processed, y_train)

    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    train()