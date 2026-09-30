import shap
import numpy as np
import pandas as pd

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def explain_single(explainer, X_processed, feature_names, top_n=5):
    shap_values = explainer.shap_values(X_processed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    values = shap_values[0]
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False)
    top = contributions.head(top_n)

    explanation = {
        "base_value": float(base_value),
        "top_features": top[["feature", "shap_value"]].to_dict(orient="records")
    }
    return explanation

def global_feature_importance(explainer, X_processed, feature_names):
    shap_values = explainer.shap_values(X_processed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    return importance