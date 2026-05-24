from typing import List

from pydantic import BaseModel

from src.api.schemas.request import TransactionRequest


class BatchTransactionRequest(BaseModel):

    transactions: List[TransactionRequest]
