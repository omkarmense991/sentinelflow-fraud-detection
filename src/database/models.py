from sqlalchemy import Column, Integer, Float, String

from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionLog(Base):

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    request_id = Column(String)

    prediction_timestamp = Column(String)

    fraud_probability = Column(Float)

    fraud_prediction = Column(Integer)

    threshold = Column(Float)

    model_version = Column(String)

    model_name = Column(String)

    sampling_strategy = Column(String)
