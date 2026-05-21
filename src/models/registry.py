# src/registry/py

from pathlib import Path

import joblib

MODEL_DIR = Path("artifacts/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_pipeline(pipeline, filename):

    model_path = MODEL_DIR / filename

    joblib.dump(pipeline, model_path)

    return model_path


def load_pipeline(filename):

    model_path = MODEL_DIR / filename

    pipeline = joblib.load(model_path)

    return pipeline
