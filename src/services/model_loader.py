from pathlib import Path
import json

import joblib

from src.config.settings import CHAMPION_MODEL_PATH, CHAMPION_METADATA_PATH

pipeline = joblib.load(CHAMPION_MODEL_PATH)


with open(CHAMPION_METADATA_PATH, "r") as f:

    metadata = json.load(f)
