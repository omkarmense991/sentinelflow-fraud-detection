
# src/api/routes/prediction.py
from fastapi import APIRouter

from src.api.schemas.request import TransactionRequest

from src.api.schemas.response import PredictionResponse

from src.services.inference_service import predict_transaction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):

    result = predict_transaction(request.dict())

    return result
