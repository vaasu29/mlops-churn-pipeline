# 🏗️ Architecture — MLOps Churn Prediction Pipeline

## Overview

This document describes the architecture of the end-to-end MLOps pipeline for customer churn prediction.

---

## 🔄 Pipeline Flow
```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│                                                             │
│   Raw CSV (Telco Dataset)  →  Preprocessing  →  Processed  │
│         7,043 records            (preprocess.py)    CSV     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                         │
│                                                             │
│   Processed Data  →  Random Forest  →  Model (.pkl)        │
│                        (train.py)                           │
│                            │                                │
│                            ▼                                │
│                     MLflow Tracking                         │
│               (params + metrics + artifacts)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                        │
│                                                             │
│   Reference Data  →  Evidently AI  →  Drift Report (HTML)  │
│   vs Current Data   (drift_detection.py)                    │
│                            │                                │
│                            ▼                                │
│              Drift > 10%? → Retraining Alert                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                          │
│                                                             │
│   Model (.pkl)  →  FastAPI  →  POST /predict                │
│                   (app.py)         │                        │
│                                   ▼                         │
│                         JSON Response                       │
│                  { prediction, probability, label }         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AUTOMATION LAYER                         │
│                                                             │
│   Apache Airflow DAG (Weekly Schedule)                      │
│                                                             │
│   preprocess_data → detect_drift → retrain_model           │
│                                                             │
│   Triggered automatically when drift is detected           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Breakdown

### 1. Data Layer
| File | Role |
|------|------|
| `data/raw/` | Original IBM Telco CSV (7,043 rows, 21 columns) |
| `data/processed/` | Cleaned, encoded, ready-to-train data |
| `src/preprocess.py` | Cleaning, encoding, missing value handling |

### 2. Training Layer
| File | Role |
|------|------|
| `src/train.py` | Random Forest training with MLflow logging |
| `mlruns/` | MLflow experiment store (params, metrics, models) |
| `models/churn_model.pkl` | Serialized trained model |

### 3. Monitoring Layer
| File | Role |
|------|------|
| `src/drift_detection.py` | Wasserstein + Jensen-Shannon drift tests |
| `reports/drift_report.html` | Interactive Evidently AI drift report |

### 4. Serving Layer
| File | Role |
|------|------|
| `src/predict.py` | Core inference logic |
| `api/app.py` | FastAPI REST endpoint with Swagger UI |

### 5. Automation Layer
| File | Role |
|------|------|
| `dags/retrain_dag.py` | Airflow DAG — weekly scheduled retraining |

### 6. Infrastructure
| File | Role |
|------|------|
| `Dockerfile` | Container image for API |
| `docker-compose.yml` | Multi-service orchestration (API + MLflow) |

---

## 🔁 Retraining Trigger Logic
```
Every Week (Airflow Scheduler)
        │
        ▼
Run Preprocessing
        │
        ▼
Check Drift (Evidently AI)
        │
   drift > 10%?
   ┌────┴────┐
  Yes        No
   │          │
   ▼          ▼
Retrain    Skip &
Model      Log Warning
   │
   ▼
Log to MLflow
   │
   ▼
Save New Model (.pkl)
```

---

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |
| max_depth | 6 |
| Train/Test Split | 80/20 |
| Accuracy | 80.27% |
| ROC-AUC | 0.8641 |
| F1 Score | 0.5587 |

---

## 🔍 Drift Detection Details

| Column | Drift Status | Stat Test | Score |
|--------|-------------|-----------|-------|
| tenure | 🔴 Detected | Wasserstein | 1.317 |
| MonthlyCharges | 🔴 Detected | Wasserstein | 1.062 |
| TotalCharges | 🔴 Detected | Wasserstein | 0.667 |
| Other 16 cols | 🟢 Not Detected | Jensen-Shannon | < 0.03 |

> Drift threshold: **10%** — if more than 10% of columns drift, retraining is triggered.

---

## 🐳 Deployment Architecture
```
Docker Compose
├── api (Port 8000)
│   ├── FastAPI + Uvicorn
│   └── Mounts: models/, data/, reports/
│
└── mlflow (Port 5000)
    └── MLflow Tracking Server
```
