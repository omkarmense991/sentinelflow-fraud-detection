from datetime import datetime
import uuid

import pandas as pd

from src.services.model_loader import pipeline, metadata

from src.database.repositories.prediction_repository import save_prediction

THRESHOLD = metadata["threshold"]


def predict_transaction(data):

    input_df = pd.DataFrame([data])

    probability = pipeline.predict_proba(input_df)[0][1]

    prediction = int(probability >= THRESHOLD)

    result = {
        "request_id": str(uuid.uuid4()),
        "prediction_timestamp": datetime.utcnow().isoformat(),
        "fraud_probability": round(float(probability), 4),
        "fraud_prediction": prediction,
        "threshold": THRESHOLD,
        "model_version": metadata["experiment_version"],
        "model_name": metadata["model_name"],
        "sampling_strategy": metadata["sampling_strategy"],
    }

    save_prediction(result)

    return result
