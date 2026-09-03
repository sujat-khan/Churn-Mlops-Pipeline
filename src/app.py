import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Union
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Employee Churn Prediction API")

# 1. Prometheus Metrics: Instruments app and exposes /metrics
Instrumentator().instrument(app).expose(app)

# 2. Path for storing production inference logs
LOG_DIR = "data/monitoring"
LOG_FILE = os.path.join(LOG_DIR, "production_inferences.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# 3. Load model and scaler
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

ref_df = pd.read_csv("data/processed/train_features.csv", nrows=1)
FEATURE_COLUMNS = [c for c in ref_df.columns if c != "Attrition"]


def transform_data(df: pd.DataFrame):
    df['TenurePerAge'] = df['YearsAtCompany'] / (df['Age'] + 1e-5)
    df['YearsInRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1e-5)
    df['YearsWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1e-5)
    df['IncomePerWorkingYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1e-5)
    df['PromotionTenureRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1e-5)

    encoded_df = pd.get_dummies(df)
    aligned_df = encoded_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return scaler.transform(aligned_df)


@app.get("/")
def home():
    return {"status": "online", "message": "Churn Prediction API running. Metrics available at /metrics"}


@app.post("/predict")
def predict(data: Union[Dict[str, Any], List[Dict[str, Any]]]):
    try:
        raw_records = [data] if isinstance(data, dict) else data
        df = pd.DataFrame(raw_records)

        # Generate predictions
        X = transform_data(df.copy())
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X)[:, 1].round(4).tolist()

        # Log inference payload for drift auditing
        log_df = df.copy()
        log_df["prediction"] = predictions
        log_df["probability"] = probabilities
        log_df["timestamp"] = datetime.utcnow().isoformat()

        file_exists = os.path.exists(LOG_FILE)
        log_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)

        return {
            "predictions": predictions,
            "churn_labels": ["Leave" if p == 1 else "Stay" for p in predictions],
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
