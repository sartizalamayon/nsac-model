from fastapi import FastAPI 
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.services.gemini_service import get_gemini_response


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
    suggestion: str

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
    data = {
        "soil_type": payload.soil_type,
        "water_frequency": payload.water_frequency,
        "fertilizer_type": payload.fertilizer_type,
        "sunlight_hours": payload.sunlight_hours,
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "growth_stage": growth_stage
    }
    suggestion = get_gemini_response(data)
    print(suggestion)
    return {"growth_stage": growth_stage, "suggestion": suggestion}


# soil_type: "Loamy"
# water_frequency: "Daily"
# fertilizer_type: "Organic"
# sunlight_hours: 8
# temperature: 25
# humidity: 60
# growth_stage: "Vegetative"


