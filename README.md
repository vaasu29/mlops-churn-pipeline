# 🔁 MLOps Churn Prediction Pipeline

> An end-to-end MLOps pipeline that predicts customer churn, detects data drift automatically, and triggers retraining alerts — built with production-grade tools used at Snowflake, Databricks, and Cloudera.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-orange?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.33-red)](https://evidentlyai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project simulates a real-world ML system where a churn prediction model is trained, served via API, and continuously monitored for data drift. When drift is detected, a retraining alert is triggered — closing the MLOps loop.

**Business Problem:** Telecom companies lose millions every year to customer churn. This pipeline predicts which customers are likely to churn and monitors model health over time.

---

## ✨ Features

- 🧹 **Automated Preprocessing** — Handles missing values, encoding, and feature engineering
- 🤖 **Model Training** — Random Forest classifier with full hyperparameter logging
- 📊 **MLflow Tracking** — Every experiment run tracked with params, metrics, and artifacts
- 🔍 **Drift Detection** — Evidently AI monitors 19 features for distribution shifts
- ⚡ **FastAPI Serving** — REST API with Swagger UI for real-time predictions
- 🐳 **Docker Ready** — Fully containerized with docker-compose

---

## 🏗️ Architecture
```
Raw CSV Data
     ↓
Preprocessing (preprocess.py)
     ↓
Model Training + MLflow Logging (train.py)
     ↓
Saved Model (.pkl) ──────────────────────────────┐
     ↓                                           ↓
Drift Detection (drift_detection.py)     FastAPI Endpoint (app.py)
     ↓                                           ↓
Evidently HTML Report              /predict → Churn Probability
     ↓
⚠️ Retraining Alert if drift > 10%
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.13 | Core development |
| ML | Scikit-learn, XGBoost | Model training |
| Tracking | MLflow 3.10 | Experiment logging |
| Monitoring | Evidently AI | Drift detection |
| Serving | FastAPI + Uvicorn | REST API |
| Container | Docker + Compose | Deployment |
| Data | Pandas, NumPy | Data processing |

---

## 📁 Project Structure
```
mlops-churn-pipeline/
│
├── 📂 data/
│   ├── raw/                    # Original Telco dataset
│   └── processed/              # Cleaned & encoded data
│
├── 📂 src/
│   ├── preprocess.py           # Data cleaning & feature engineering
│   ├── train.py                # Model training + MLflow logging
│   ├── predict.py              # Inference logic
│   └── drift_detection.py      # Evidently AI drift reports
│
├── 📂 api/
│   └── app.py                  # FastAPI serving endpoint
│
├── 📂 reports/                 # HTML drift reports
├── 📂 models/                  # Saved model (.pkl)
├── 📂 mlruns/                  # MLflow experiment logs
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip
- Git

### 1. Clone the repo
```bash
git clone https://github.com/vaasu29/mlops-churn-pipeline.git
cd mlops-churn-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the CSV in `data/raw/`.

### 4. Run the full pipeline
```bash
# Step 1 - Preprocess
python src/preprocess.py

# Step 2 - Train + log to MLflow
python src/train.py

# Step 3 - Detect drift
python src/drift_detection.py

# Step 4 - Serve API
python -m uvicorn api.app:app --reload
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Accuracy | 80.27% |
| 🎯 ROC-AUC | 0.8641 |
| ⚖️ F1 Score | 0.5587 |
| 🔬 Precision | 68.48% |
| 📡 Recall | 47.18% |

---

## 🔍 Drift Detection Results

Evidently AI compares reference vs current data distributions across all 19 features.

| Column | Status | Stat Test | Drift Score |
|--------|--------|-----------|-------------|
| tenure | 🔴 Detected | Wasserstein | 1.317 |
| MonthlyCharges | 🔴 Detected | Wasserstein | 1.062 |
| TotalCharges | 🔴 Detected | Wasserstein | 0.667 |
| Other 16 cols | 🟢 Not Detected | Jensen-Shannon | < 0.03 |

> Drift threshold set at 10%. When more than 10% of columns drift, retraining is recommended.

---

## ⚡ API Overview

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Health check |
| GET | `/health` | System status |
| POST | `/predict` | Predict churn probability |

### Sample Request
```json
POST /predict
{
  "gender": 0,
  "SeniorCitizen": 0,
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  ...
}
```

### Sample Response
```json
{
  "prediction": 0,
  "churn_probability": 0.4166,
  "churn_label": "No"
}
```

---

## 🐳 Run with Docker
```bash
docker-compose up --build
```

- API → http://localhost:8000/docs
- MLflow → http://localhost:5000

---

## 📈 MLflow Dashboard
```bash
python -m mlflow ui
```
Visit http://127.0.0.1:5000 to compare experiment runs, view metrics, and manage model versions.

---

## 🗺️ Roadmap

- [x] Data preprocessing pipeline
- [x] Model training with MLflow tracking
- [x] Drift detection with Evidently AI
- [x] FastAPI serving endpoint
- [x] Docker containerization
- [ ] Airflow DAG for scheduled retraining
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment (AWS/Azure)

---

## 👨‍💻 Author

**Vaasu Chandra**
[![GitHub](https://img.shields.io/badge/GitHub-vaasu29-black?logo=github)](https://github.com/vaasu29)
