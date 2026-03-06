from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict

app = FastAPI(
    title="Churn Prediction API",
    description="MLOps pipeline for customer churn prediction",
    version="1.0.0"
)

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        input_data = customer.model_dump()
        result = predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
