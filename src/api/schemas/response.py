# src/api/schemas/response.py
from pydantic import BaseModel


class PredictionResponse(BaseModel):

    fraud_probability: float

    fraud_prediction: int

    threshold: float

    model_version: str
