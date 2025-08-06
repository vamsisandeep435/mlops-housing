import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import logging
from datetime import datetime
import sqlite3

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

# Define input format
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Init app
app = FastAPI(title="California Housing Predictor")

# Logging config
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


@app.post("/predict")
def predict_price(input: HousingInput):
    df = pd.DataFrame([input.dict()])
    prediction = model.predict(df)[0]

    # Log input and prediction
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input": input.dict(),
        "prediction": round(prediction, 3)
    }

    logging.info(log_data)

    # Optional: Write to SQLite DB
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                        timestamp TEXT,
                        MedInc REAL, HouseAge REAL, AveRooms REAL, AveBedrms REAL,
                        Population REAL, AveOccup REAL, Latitude REAL, Longitude REAL,
                        prediction REAL
                      )''')

    cursor.execute('''INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (log_data["timestamp"], *df.iloc[0].tolist(), log_data["prediction"]))

    conn.commit()
    conn.close()

    return {"predicted_price": log_data["prediction"]}

@app.get("/metrics")
def get_metrics():
    try:
        conn = sqlite3.connect("logs/predictions.db")
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
        if cursor.fetchone() is None:
            return {
                "error": "No logs found yet. Hit /predict at least once."
            }

        # Proceed only if table exists
        cursor.execute("SELECT COUNT(*) FROM logs")
        total_requests = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(prediction) FROM logs")
        avg_pred = cursor.fetchone()[0]

        conn.close()

        return {
            "total_requests": total_requests,
            "average_prediction": round(avg_pred, 3) if avg_pred else None
        }

    except Exception as e:
        return {"error": str(e)}

# Entry point
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


# Ensure the table exists
def init_db():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()
