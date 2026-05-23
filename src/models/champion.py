# src/models/champion.py

from pathlib import Path
import shutil


CHAMPION_DIR = Path(
    "artifacts/champion"
)

CHAMPION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_champion_model(
    model_source_path,
    metadata_source_path
):

    champion_model_path = (
        CHAMPION_DIR /
        "champion_model.pkl"
    )

    champion_metadata_path = (
        CHAMPION_DIR /
        "champion_metadata.json"
    )

    shutil.copy(
        model_source_path,
        champion_model_path
    )

    shutil.copy(
        metadata_source_path,
        champion_metadata_path
    )