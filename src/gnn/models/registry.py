# src/gnn/models/registry.py

from pathlib import Path

import torch

from src.config.settings import MODELS_DIR

MODEL_DIR = MODELS_DIR / "elliptic"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_gnn_model(model, filename):

    save_path = MODEL_DIR / filename

    torch.save(model.state_dict(), save_path)

    return save_path
