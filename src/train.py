import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_processed_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def train(data_path="data/processed/churn_processed.csv"):
    X, y = load_processed_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("--- Training Results ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/churn_model.pkl")
        print("Model saved to models/churn_model.pkl")

if __name__ == "__main__":
    train()
