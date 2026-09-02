import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_predict_endpoint():
    sample_payload = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 19000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "churn_labels" in data
    assert "probabilities" in data
