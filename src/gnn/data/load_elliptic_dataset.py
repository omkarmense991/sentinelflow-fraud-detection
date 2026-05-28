from pathlib import Path

import pandas as pd

# =========================================
# Resolve Project Root
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =========================================
# Dataset Directory
# =========================================

DATA_DIR = PROJECT_ROOT / "data/raw/elliptic/" "elliptic_bitcoin_dataset"


def load_elliptic_dataset():

    features_path = DATA_DIR / "elliptic_txs_features.csv"

    classes_path = DATA_DIR / "elliptic_txs_classes.csv"

    edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

    features_df = pd.read_csv(features_path, header=None)

    classes_df = pd.read_csv(classes_path)

    edges_df = pd.read_csv(edges_path)

    return (features_df, classes_df, edges_df)
