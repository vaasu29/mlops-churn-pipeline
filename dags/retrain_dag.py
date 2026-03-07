from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

default_args = {
    'owner': 'vaasu',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def run_preprocessing():
    from src.preprocess import load_data, clean_data, encode_features, save_processed_data
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    save_processed_data(df, "data/processed/churn_processed.csv")
    print("Preprocessing complete")

def run_drift_detection():
    from src.drift_detection import load_data, simulate_drift, check_drift
    import pandas as pd
    df = load_data("data/processed/churn_processed.csv")
    split = int(len(df) * 0.7)
    reference_df = df.iloc[:split].drop(columns=['Churn'])
    current_df = simulate_drift(df.iloc[split:].drop(columns=['Churn']))
    drift_detected = check_drift(reference_df, current_df)
    if not drift_detected:
        raise ValueError("No drift detected - skipping retraining")
    print("Drift detected - proceeding to retrain")

def run_training():
    from src.train import train
    train()
    print("Retraining complete")

with DAG(
    dag_id='churn_retraining_pipeline',
    default_args=default_args,
    description='Auto-retraining pipeline triggered on data drift',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'churn', 'retraining']
) as dag:

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_preprocessing
    )

    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=run_drift_detection
    )

    retrain = PythonOperator(
        task_id='retrain_model',
        python_callable=run_training
    )

    preprocess >> detect_drift >> retrain
