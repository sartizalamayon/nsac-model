from fastapi import FastAPI 
from pydantic import BaseModel # pydantic models to validate request and response
from app.model.model import predict_pipeline # function to predict language
from app.model.model import __version__ as model_version # model version


app = FastAPI()

# pydantic models
class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str

# endpoints
@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}
