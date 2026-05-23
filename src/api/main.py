# src/api/main.py
from fastapi import FastAPI

from src.api.routes.prediction import router as prediction_router

app = FastAPI(title="SentinelFlow Fraud Detection API")


app.include_router(prediction_router)
