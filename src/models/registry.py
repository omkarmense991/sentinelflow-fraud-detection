import joblib

from src.config.settings import MODELS_DIR


def save_pipeline(pipeline, filename):

    model_path = MODELS_DIR / filename

    joblib.dump(pipeline, model_path)

    return model_path


def load_pipeline(filename):

    model_path = MODELS_DIR / filename

    pipeline = joblib.load(model_path)

    return pipeline
