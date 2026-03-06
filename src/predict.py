import pandas as pd
import joblib
import os

def load_model(model_path="models/churn_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model

def predict(input_data, model_path="models/churn_model.pkl"):
    model = load_model(model_path)

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "churn_probability": round(float(probability[0]), 4),
        "churn_label": "Yes" if prediction[0] == 1 else "No"
    }

if __name__ == "__main__":
    sample = {
        "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 2, "MonthlyCharges": 65.5, "TotalCharges": 786.0
    }

    result = predict(sample)
    print("--- Prediction Result ---")
    print(f"Churn: {result['churn_label']}")
    print(f"Churn Probability: {result['churn_probability']:.2%}")
