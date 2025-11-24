from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Iris ML Predictor")

# Load model (use env var for flexibility)
model_path = os.getenv("MODEL_PATH", "app/iris_model.pkl")
model = joblib.load(model_path)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris ML Predictor API is running!"}

@app.post("/predict")
def predict(data: IrisInput):
    input_data = np.array([[data.sepal_length, data.sepal_width, 
                            data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {"prediction": int(prediction), "species": species_map[int(prediction)]}