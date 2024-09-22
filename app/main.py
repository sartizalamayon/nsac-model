from fastapi import FastAPI 
from pydantic import BaseModel
from app.model.model import predict_pipeline

app = FastAPI()

# pydantic model for input
class PlantGrowthIn(BaseModel):
    soil_type: str
    water_frequency: str
    fertilizer_type: str
    sunlight_hours: float
    temperature: float
    humidity: float

# pydantic model for output
class PredictionOut(BaseModel):
    growth_stage: str

# Root endpoint
@app.get("/")
def home():
    return {"model_check": "OK"}

# Predict endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: PlantGrowthIn):
    growth_stage = predict_pipeline(
        payload.soil_type, 
        payload.water_frequency, 
        payload.fertilizer_type, 
        payload.sunlight_hours,
        payload.temperature,
        payload.humidity
    )
    return {"growth_stage": growth_stage}
