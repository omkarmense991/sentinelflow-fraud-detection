import shutil

from src.config.settings import CHAMPION_MODEL_PATH, CHAMPION_METADATA_PATH


def save_champion_model(model_source_path, metadata_source_path):

    shutil.copy(model_source_path, CHAMPION_MODEL_PATH)

    shutil.copy(metadata_source_path, CHAMPION_METADATA_PATH)
