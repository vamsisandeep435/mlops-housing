import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Predictor API!"}

@app.post("/predict")
def predict_price(input: HousingInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_price": round(prediction, 3)}

# Entry point
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
