# src/services/inference_service.py
import pandas as pd

from src.services.model_loader import pipeline

THRESHOLD = 0.3


def predict_transaction(data):

    input_df = pd.DataFrame([data])

    probability = pipeline.predict_proba(input_df)[0][1]

    prediction = int(probability >= THRESHOLD)

    return {
        "fraud_probability": probability,
        "fraud_prediction": prediction,
        "threshold": THRESHOLD,
        "model_version": "v1",
    }
