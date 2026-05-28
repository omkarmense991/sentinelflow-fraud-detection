# src/tabular_ml/data/load_elliptic_tabular.py

from src.gnn.data.load_elliptic_dataset import load_elliptic_dataset


def load_elliptic_tabular_dataset():

    features_df, classes_df, _ = load_elliptic_dataset()

    # =========================================
    # Rename Transaction ID Column
    # =========================================

    features_df = features_df.rename(columns={0: "txId"})

    # =========================================
    # Merge Labels
    # =========================================

    df = features_df.merge(classes_df, on="txId")

    # =========================================
    # Remove Unknown Labels
    # =========================================

    df = df[df["class"] != "unknown"].copy()

    # =========================================
    # Label Encoding
    # illicit = 1
    # licit = 0
    # =========================================

    df["label"] = df["class"].map({"1": 1, "2": 0}).astype(int)

    return df
