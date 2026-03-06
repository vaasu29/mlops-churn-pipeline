import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Fix TotalCharges column (has spaces instead of nulls)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop customerID (not useful for prediction)
    df.drop(columns=['customerID'], inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def encode_features(df):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_path = "data/processed/churn_processed.csv"

    df = load_data(raw_path)
    print(f"Raw data shape: {df.shape}")

    df = clean_data(df)
    df = encode_features(df)
    print(f"Processed data shape: {df.shape}")

    save_processed_data(df, processed_path)