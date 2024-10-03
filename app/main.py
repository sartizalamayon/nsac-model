from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.services.gemini_service import get_gemini_response
# from PIL import Image
# import io



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


# # Endpoint for soil type prediction
# @app.post("/predict-soil-type")
# async def predict_soil_type(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image file
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

#         # Classify the soil type
#         predicted_soil_type = classify_soil(image)
#         return {"soil_type": predicted_soil_type}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# # Endpoint for soil type prediction
# @app.post("/predict-soil-type")
# async def predict_soil_type(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image file
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

#         # Classify the soil type
#         predicted_soil_type = classify_soil(image)
#         return {"soil_type": predicted_soil_type}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))