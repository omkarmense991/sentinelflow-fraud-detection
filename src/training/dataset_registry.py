# src/training/dataset_registry.py

"""
Purpose:
Central registry mapping dataset names to their
per-family training callables.

IMPORTANT: All imports inside each callable are
intentionally lazy. Loading the full tabular stack
(sklearn, xgboost, joblib) alongside PyTorch + MPS
at module-import time exhausts memory and can
segfault on macOS when a large CSV is later read.
"""


# =========================================
# Per-Family Callables (Lazy Imports)
# =========================================

def _tabular_elliptic():

    from src.tabular_ml.training.tabular_trainer import run_training

    from src.tabular_ml.data.load_elliptic_tabular import (
        load_elliptic_tabular_dataset,
    )

    from src.tabular_ml.features.elliptical_preprocessing import split_elliptic_data

    run_training(
        dataset_name="elliptic",
        dataset_loader=load_elliptic_tabular_dataset,
        split_function=split_elliptic_data,
    )


def _gnn_elliptic():

    from src.gnn.training.gcn_trainer import train_gcn

    train_gcn()


# =========================================
# Dataset Registry
# =========================================
# Each entry maps a dataset name to
# {family_name: callable}.
#
# Add a new dataset or family here; nothing
# else in the orchestrator needs to change.
# =========================================

DATASET_REGISTRY = {
    "elliptic": {
        "tabular": _tabular_elliptic,
        "gnn": _gnn_elliptic,
    },
}
