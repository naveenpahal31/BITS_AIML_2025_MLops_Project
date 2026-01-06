
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

model = joblib.load("model.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max()
    return {"prediction": int(pred), "confidence": float(prob)}
