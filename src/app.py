import os
import pickle
import pandas as pd
from typing import Dict, Any, List, Union
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Employee Churn Prediction API")

# 1. Load model and scaler directly on startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Get reference feature columns for alignment
ref_df = pd.read_csv("data/processed/train_features.csv", nrows=1)
FEATURE_COLUMNS = [c for c in ref_df.columns if c != "Attrition"]


def transform_data(df: pd.DataFrame):
    """Simple feature transform & alignment."""
    # Create Domain interaction features
    df['TenurePerAge'] = df['YearsAtCompany'] / (df['Age'] + 1e-5)
    df['YearsInRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1e-5)
    df['YearsWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1e-5)
    df['IncomePerWorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1e-5)
    df['PromotionTenureRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1e-5)

    # Encode & Align columns
    encoded_df = pd.get_dummies(df)
    aligned_df = encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    # Scale
    return scaler.transform(aligned_df)


@app.get("/")
def home():
    return {"status": "online", "message": "Churn Prediction API is running. Go to /docs to test"}


@app.post("/predict")
def predict(data: Union[Dict[str, Any], List[Dict[str, Any]]]):
    """Accepts any single JSON object or a list of JSON objects without declaring schemas!"""
    try:
        # Convert incoming JSON directly into a pandas DataFrame
        df = pd.DataFrame([data] if isinstance(data, dict) else data)

        # Preprocess & Predict
        X = transform_data(df)
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X)[:, 1].round(4).tolist()

        return {
            "predictions": predictions,
            "churn_labels": ["Leave" if p == 1 else "Stay" for p in predictions],
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
