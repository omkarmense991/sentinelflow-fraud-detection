# src/api/main.py
from fastapi import FastAPI

from src.api.routes.prediction import router as prediction_router
from src.api.routes.health import router as health_router

app = FastAPI(title="SentinelFlow Fraud Detection API")


@app.get("/")
def root():
    return {"message": "SentinelFlow Fraud Detection API"}


app.include_router(prediction_router)
app.include_router(health_router)
