
# 🏠 California Housing Price Predictor - MLOps Pipeline

This project demonstrates a complete MLOps pipeline using the **California Housing Dataset**. The pipeline covers all stages from development to deployment, with versioning, experiment tracking, Dockerization, CI/CD, and monitoring.

---

## 📌 Project Overview

- **Model Type**: Regression (Linear Regression & Decision Tree Regressor)
- **Dataset**: [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Frameworks**: Scikit-learn, FastAPI, Docker, GitHub Actions, MLflow

---

## 📁 Project Structure

```
mlops-housing/
│
├── data/                  # Raw dataset CSV
├── models/                # Saved models (.pkl)
├── logs/                  # Prediction logs & SQLite DB
├── src/
│   ├── train.py           # Model training with MLflow tracking
│   ├── evaluate.py        # Evaluation script
│   ├── api.py             # FastAPI app for prediction + metrics
│
├── Dockerfile             # Docker setup for API
├── requirements.txt       # Project dependencies
├── .github/workflows/     # GitHub Actions CI/CD workflow
└── README.md              # Project overview
```

---

## ⚙️ Part 1: Data Versioning

- GitHub repo initialized and connected.
- Dataset stored in `data/housing.csv`
- DVC initialized for dataset tracking.

---

## 🧪 Part 2: Model Development

- Trained two models: `LinearRegression` and `DecisionTreeRegressor`
- Tracked experiments using MLflow:
  - Metrics: RMSE, R²
  - Parameters: max_depth, random_state
- Best model auto-logged and saved to `models/model.pkl`

---

## 🌐 Part 3: API & Docker

- Developed REST API using **FastAPI**
- Accepts input JSON and returns price prediction
- Dockerized with:
  ```bash
  docker build -t housing-predictor .
  docker run -p 8000:8000 housing-predictor
  ```

---

## 🔁 Part 4: CI/CD with GitHub Actions

- On every push:
  - Lints Python code
  - Builds Docker image
  - Pushes to Docker Hub
- Secrets used:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

---

## 📈 Part 5: Logging & Monitoring

- Logs all predictions to `logs/predictions.db`
- `/metrics` endpoint shows total predictions served
- SQLite used for persistent logging
- Optionally view `predictions.log`

---

## 🧪 Example Usage

### 🚀 POST Prediction
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample.json
```

### 📊 GET Metrics
```bash
curl http://localhost:8000/metrics
```

---

## 🎁 Bonus Features

- ✅ Input validation with `pydantic`
- ✅ `/metrics` endpoint for monitoring
- 🛠️ Prometheus/Grafana support (optional setup)
- 🔁 Ready for model re-training with new data

---

## 🎥 Demo Recording

> 📹 See `demo.mp4` for 5-min walkthrough

---

## 📅 Last Updated
**August 06, 2025**

---

## 👨‍💻 Author
[SandeepEddula](https://github.com/SandeepAmruta)
