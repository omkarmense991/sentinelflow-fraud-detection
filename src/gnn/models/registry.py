from pathlib import Path

import torch

MODEL_DIR = Path("artifacts/gnn_models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_gnn_model(model, filename):

    save_path = MODEL_DIR / filename

    torch.save(model.state_dict(), save_path)

    return save_path
