# src/tabular_ml/models/registry.py

import joblib

from src.config.settings import MODELS_DIR


def save_pipeline(pipeline, filename, dataset_name):

    model_dir = MODELS_DIR / dataset_name

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / filename

    joblib.dump(pipeline, model_path)

    return model_path
