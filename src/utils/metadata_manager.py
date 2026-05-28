# src/utils/metadata_manager.py

import json
import math
import numpy as np
from pathlib import Path

from src.config.settings import METADATA_DIR


# =========================================
# Safe JSON Serializer
# =========================================

def json_serializer(obj):

    # -----------------------------------------
    # NumPy Integer
    # -----------------------------------------

    if isinstance(obj, np.integer):

        return int(obj)

    # -----------------------------------------
    # NumPy Float
    # -----------------------------------------

    if isinstance(obj, np.floating):

        if math.isnan(obj):

            return None

        return float(obj)

    # -----------------------------------------
    # NumPy Array
    # -----------------------------------------

    if isinstance(obj, np.ndarray):

        return obj.tolist()

    # -----------------------------------------
    # Path Objects
    # -----------------------------------------

    if isinstance(obj, Path):

        return str(obj)

    # -----------------------------------------
    # Python Float NaN
    # -----------------------------------------

    if isinstance(obj, float):

        if math.isnan(obj):

            return None

    # -----------------------------------------
    # Unsupported
    # -----------------------------------------

    raise TypeError(
        f"Object of type "
        f"{type(obj).__name__} "
        f"is not JSON serializable"
    )


# =========================================
# Sanitize Metadata (replace NaN with None)
# =========================================

def _sanitize(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# =========================================
# Save Metadata
# =========================================

def save_metadata(metadata, filename, dataset_name):

    # =========================================
    # Dataset Metadata Directory
    # =========================================

    dataset_metadata_dir = (
        METADATA_DIR / dataset_name
    )

    dataset_metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================
    # Metadata Path
    # =========================================

    metadata_path = (
        dataset_metadata_dir / filename
    )

    # =========================================
    # Save JSON
    # =========================================

    with open(metadata_path, "w") as f:

        json.dump(
            _sanitize(metadata),
            f,
            indent=4,
            default=json_serializer,
            allow_nan=False,
        )

    return metadata_path