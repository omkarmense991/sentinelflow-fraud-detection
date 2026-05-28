# src/tabular_ml/models/champion.py

import shutil

from src.config.settings import CHAMPION_DIR


def save_champion_model(model_source_path, metadata_source_path, dataset_name):

    dataset_champion_dir = CHAMPION_DIR / dataset_name

    dataset_champion_dir.mkdir(parents=True, exist_ok=True)

    champion_model_path = dataset_champion_dir / "champion_model.pkl"

    champion_metadata_path = dataset_champion_dir / "champion_metadata.json"

    shutil.copy(model_source_path, champion_model_path)

    shutil.copy(metadata_source_path, champion_metadata_path)
