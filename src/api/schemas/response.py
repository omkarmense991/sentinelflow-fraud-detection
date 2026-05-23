from pydantic import BaseModel


class PredictionResponse(BaseModel):

    request_id: str

    prediction_timestamp: str

    fraud_probability: float

    fraud_prediction: int

    threshold: float

    model_version: str

    model_name: str

    sampling_strategy: str
