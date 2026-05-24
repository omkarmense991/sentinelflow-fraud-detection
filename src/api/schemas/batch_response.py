from typing import List

from pydantic import BaseModel

from src.api.schemas.response import PredictionResponse


class BatchPredictionResponse(BaseModel):

    predictions: List[PredictionResponse]
