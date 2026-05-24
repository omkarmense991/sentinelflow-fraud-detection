from fastapi import APIRouter
from fastapi import HTTPException

from src.api.schemas.request import TransactionRequest

from src.api.schemas.response import PredictionResponse

from src.api.schemas.batch_request import BatchTransactionRequest

from src.api.schemas.batch_response import BatchPredictionResponse

from src.services.inference_service import predict_transaction, batch_predict

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):

    try:

        result = predict_transaction(request.model_dump())

        return result

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def batch_prediction(request: BatchTransactionRequest):

    try:

        transactions = [
            transaction.model_dump() for transaction in request.transactions
        ]

        return batch_predict(transactions)

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
