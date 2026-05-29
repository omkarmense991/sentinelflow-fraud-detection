# src/gnn/models/registry.py

from pathlib import Path

import torch

from src.config.settings import MODELS_DIR

from src.gnn.models.gcn_model import GCN
from src.gnn.models.graphsage_model import GraphSAGE

MODEL_DIR = MODELS_DIR / "elliptic"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# Available GNN Architectures
# =========================================


def get_gnn_models():

    return {
        "gcn": GCN,
        "graphsage": GraphSAGE,
    }


# =========================================
# Model Persistence
# =========================================


def save_gnn_model(
    model,
    filename,
    dataset_name,
):

    model_dir = MODELS_DIR / dataset_name

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = model_dir / filename

    torch.save(
        model.state_dict(),
        save_path,
    )

    return save_path
