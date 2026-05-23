# src/services/model_loader.py
from pathlib import Path

import joblib

MODEL_PATH = Path("artifacts/champion/champion_model.pkl")


pipeline = joblib.load(MODEL_PATH)
