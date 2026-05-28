# src/tracking/experiment_tracker.py

import pandas as pd

from pathlib import Path

from src.config.settings import METRICS_DIR


def save_experiment_result(result, dataset_name):

    # =========================================
    # Dataset Metrics Directory
    # =========================================

    dataset_metrics_dir = METRICS_DIR / dataset_name

    dataset_metrics_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Metrics CSV Path
    # =========================================

    metrics_path = dataset_metrics_dir / "experiment_results.csv"

    # =========================================
    # Create DataFrame
    # =========================================

    df = pd.DataFrame([result])

    # =========================================
    # Append Existing Results
    # =========================================

    if metrics_path.exists():

        existing_df = pd.read_csv(metrics_path)

        df = pd.concat([existing_df, df], ignore_index=True)

    # =========================================
    # Save CSV
    # =========================================

    df.to_csv(metrics_path, index=False)
