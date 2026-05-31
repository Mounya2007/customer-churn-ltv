# FastAPI - Customer Churn Prediction & LTV API
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction & LTV API",
    description="API for predicting customer churn and lifetime value",
    version="1.0.0"
)

# Load models
with open("../models/best_model.pkl", "rb") as f:
    churn_model = pickle.load(f)

with open("../models/ltv_model.pkl", "rb") as f:
    ltv_model = pickle.load(f)

with open("../models/feature_names_v2.pkl", "rb") as f:
    churn_features = pickle.load(f)

with open("../models/ltv_features.pkl", "rb") as f:
    ltv_features = pickle.load(f)

# Input schema for churn prediction
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    TotalServices: int
    ChargesPerMonth: float
    IsNewCustomer: int
    IsLongTermCustomer: int
    IsHighSpender: int
    SeniorCitizen: int

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction & LTV API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/ltv", "/health"]
    }

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "API is running!"}

# Churn prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Create input dataframe
    input_data = pd.DataFrame([customer.dict()])
    
    # Add missing features with 0
    for col in churn_features:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Select features in correct order
    input_data = input_data[churn_features]
    
    # Make prediction
    prediction = churn_model.predict(input_data)[0]
    probability = churn_model.predict_proba(input_data)[0]
    
    return {
        "churn_prediction": int(prediction),
        "churn_status": "Will Churn" if prediction == 1 else "Will Not Churn",
        "churn_probability": round(float(probability[1]) * 100, 2),
        "retention_probability": round(float(probability[0]) * 100, 2)
    }

# LTV prediction endpoint
@app.post("/ltv")
def predict_ltv(customer: CustomerData):
    try:
        # Create input dataframe
        customer_dict = {
            "tenure": customer.tenure,
            "MonthlyCharges": customer.MonthlyCharges,
            "TotalCharges": customer.TotalCharges,
            "TotalServices": customer.TotalServices,
            "ChargesPerMonth": customer.ChargesPerMonth,
            "IsNewCustomer": customer.IsNewCustomer,
            "IsLongTermCustomer": customer.IsLongTermCustomer,
            "IsHighSpender": customer.IsHighSpender
        }
        
        input_data = pd.DataFrame([customer_dict])
        ltv_prediction = ltv_model.predict(input_data)[0]
        ltv_value = float(ltv_prediction)
        
        if ltv_value < 500:
            segment = "Low"
        elif ltv_value < 2000:
            segment = "Medium"
        elif ltv_value < 5000:
            segment = "High"
        else:
            segment = "Premium"
        
        return {
            "predicted_ltv": round(ltv_value, 2),
            "ltv_segment": segment
        }
    except Exception as e:
        return {"error": str(e)}