from src.database.connection import SessionLocal

from src.database.models import PredictionLog


def save_prediction(prediction_data):

    db = SessionLocal()

    prediction = PredictionLog(**prediction_data)

    db.add(prediction)

    db.commit()

    db.close()
