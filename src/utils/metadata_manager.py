# src/utils/metadata_manager.py

import json

from src.config.settings import METADATA_DIR


def save_metadata(metadata, filename):

    metadata_path = METADATA_DIR / filename

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=4)

    return metadata_path
