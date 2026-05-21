# src/utils/metadata_manager.py

import json

from pathlib import Path

METADATA_DIR = Path("artifacts/metadata")

METADATA_DIR.mkdir(parents=True, exist_ok=True)


def save_metadata(metadata, filename):

    metadata_path = METADATA_DIR / filename

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=4)
