# src/tabular_ml/models/pipeline.py
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


def build_pipeline(model, sampler=None):

    steps = []

    steps.append(("scaler", StandardScaler()))

    if sampler is not None:

        steps.append(("sampler", sampler))

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    return pipeline
