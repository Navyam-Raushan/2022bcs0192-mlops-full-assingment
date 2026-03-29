from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("models/model.pkl")

# ✅ Define input schema
class InputData(BaseModel):
    data: list

@app.get("/")
def read_root():
    return {"name": "Raushan Kumar", "roll": "2022BCS0192"}

@app.post("/predict")
def predict(input: InputData):
    arr = np.array(input.data).reshape(1, -1)
    pred = model.predict(arr)
    return {
        "prediction": int(pred[0]),
        "name": "Raushan Kumar",
        "roll": "2022BCS0192"
    }