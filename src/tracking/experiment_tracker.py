# src/tracking/experiment_tracker.py

import pandas as pd
from pathlib import Path

METRICS_PATH = Path("artifacts/metrics/experiment_results.csv")


def save_experiment_result(result):

    df = pd.DataFrame([result])

    if METRICS_PATH.exists():

        existing_df = pd.read_csv(METRICS_PATH)

        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(METRICS_PATH, index=False)
