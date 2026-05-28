# src/models/champion.py

"""
Purpose:
Global champion artifact handler.

Supports multiple champion types per dataset:
    - tabular   (.pkl via joblib)
    - gnn       (.pt   via torch)
    - future families

Final layout:
    artifacts/champion/{dataset_name}/
        {champion_type}_champion{.pkl|.pt}
        {champion_type}_champion_metadata.json
"""

import shutil
from pathlib import Path

from src.config.settings import CHAMPION_DIR

from src.utils.logger import logger


# =========================================
# Default Extensions Per Champion Family
# =========================================

CHAMPION_DEFAULT_EXTENSIONS = {
    "tabular": ".pkl",
    "gnn": ".pt",
}


# =========================================
# Internal Helpers
# =========================================

def _resolve_extension(champion_type, model_source_path):

    # -----------------------------------------
    # Prefer Source Suffix
    # -----------------------------------------

    source_suffix = Path(model_source_path).suffix

    if source_suffix:

        return source_suffix

    # -----------------------------------------
    # Fallback To Family Default
    # -----------------------------------------

    default_suffix = CHAMPION_DEFAULT_EXTENSIONS.get(
        champion_type
    )

    if default_suffix is None:

        raise ValueError(
            f"Unknown champion_type: "
            f"{champion_type}"
        )

    return default_suffix


# =========================================
# Save Champion
# =========================================

def save_champion_model(
    model_source_path,
    metadata_source_path,
    dataset_name,
    champion_type,
):

    """
    Copy the per-family best model and its metadata
    into the dataset champion directory under the
    canonical {champion_type}_champion* names.
    """

    # =========================================
    # Validate Sources
    # =========================================

    model_source_path = Path(model_source_path)

    metadata_source_path = Path(metadata_source_path)

    if not model_source_path.exists():

        raise FileNotFoundError(
            f"Champion model source not found: "
            f"{model_source_path}"
        )

    if not metadata_source_path.exists():

        raise FileNotFoundError(
            f"Champion metadata source not found: "
            f"{metadata_source_path}"
        )

    # =========================================
    # Champion Directory
    # =========================================

    dataset_champion_dir = CHAMPION_DIR / dataset_name

    dataset_champion_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Destination Paths
    # =========================================

    extension = _resolve_extension(
        champion_type, model_source_path
    )

    champion_model_path = (
        dataset_champion_dir
        / f"{champion_type}_champion{extension}"
    )

    champion_metadata_path = (
        dataset_champion_dir
        / f"{champion_type}_champion_metadata.json"
    )

    # =========================================
    # Copy Artifacts
    # =========================================

    shutil.copy(model_source_path, champion_model_path)

    shutil.copy(metadata_source_path, champion_metadata_path)

    logger.info(
        f"Saved {champion_type} champion "
        f"-> {champion_model_path}"
    )

    return champion_model_path, champion_metadata_path
