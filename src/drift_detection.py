import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def simulate_drift(df):
    drifted_df = df.copy()
    drifted_df['tenure'] = drifted_df['tenure'] * np.random.uniform(1.5, 2.5, size=len(drifted_df))
    drifted_df['MonthlyCharges'] = drifted_df['MonthlyCharges'] * np.random.uniform(1.2, 1.8, size=len(drifted_df))
    drifted_df['TotalCharges'] = drifted_df['TotalCharges'] * np.random.uniform(1.3, 2.0, size=len(drifted_df))
    return drifted_df

def generate_drift_report(reference_df, current_df, report_path="reports/drift_report.html"):
    os.makedirs("reports", exist_ok=True)
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric()
    ])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(report_path)
    print(f"Drift report saved to {report_path}")

def check_drift(reference_df, current_df, threshold=0.1):
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
    drift_detected = drift_share > threshold
    print(f"Share of drifted columns: {drift_share:.2%}")
    print(f"Drift detected (threshold={threshold}): {drift_detected}")
    return drift_detected

if __name__ == "__main__":
    df = load_data("data/processed/churn_processed.csv")

    split = int(len(df) * 0.7)
    reference_df = df.iloc[:split].drop(columns=['Churn'])
    current_df = simulate_drift(df.iloc[split:].drop(columns=['Churn']))

    print("--- Checking for Data Drift ---")
    drift_detected = check_drift(reference_df, current_df, threshold=0.1)
    generate_drift_report(reference_df, current_df)

    if drift_detected:
        print("WARNING: Drift detected - model retraining recommended")
    else:
        print("No significant drift detected")
