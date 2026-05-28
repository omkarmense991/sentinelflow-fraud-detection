# src/models/select_overall_champion.py

"""
Purpose:
Pick the overall dataset champion by comparing
per-family champion metadata (tabular vs gnn)
using PR-AUC as the primary metric.

Inputs (read from artifacts/champion/{dataset}/):
    tabular_champion_metadata.json   + tabular_champion.pkl
    gnn_champion_metadata.json       + gnn_champion.pt

Outputs (written to the same directory):
    {dataset}_champion{.pkl|.pt}
    {dataset}_champion_metadata.json
"""

import json
import shutil

from src.config.settings import CHAMPION_DIR

from src.utils.logger import logger


# =========================================
# Champion Families Considered
# =========================================

CANDIDATE_FAMILIES = ("tabular", "gnn")

SUPPORTED_EXTENSIONS = (".pkl", ".pt")


# =========================================
# Internal Helpers
# =========================================

def _extract_pr_auc(metadata):

    try:

        return float(metadata["selection_metrics"]["test_pr_auc"])

    except (KeyError, TypeError):

        raise KeyError("PR-AUC not found in champion metadata")


def _load_candidate(dataset_champion_dir, champion_type):

    """
    Load a per-family champion candidate.
    Returns None if either the model or metadata
    file is missing.
    """

    metadata_path = (
        dataset_champion_dir
        / f"{champion_type}_champion_metadata.json"
    )

    if not metadata_path.exists():

        return None

    # -----------------------------------------
    # Locate Model File (either .pkl or .pt)
    # -----------------------------------------

    model_path = None

    for ext in SUPPORTED_EXTENSIONS:

        candidate = (
            dataset_champion_dir
            / f"{champion_type}_champion{ext}"
        )

        if candidate.exists():

            model_path = candidate

            break

    if model_path is None:

        return None

    # -----------------------------------------
    # Load Metadata
    # -----------------------------------------

    with open(metadata_path, "r") as f:

        metadata = json.load(f)

    pr_auc = _extract_pr_auc(metadata)

    return {
        "champion_type": champion_type,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "pr_auc": pr_auc,
    }


def _clear_stale_overall_models(dataset_champion_dir, dataset_name, keep_path):

    """
    Remove any previous overall champion model with a
    different extension so we never leave stale files.
    """

    for ext in SUPPORTED_EXTENSIONS:

        stale = (
            dataset_champion_dir
            / f"{dataset_name}_champion{ext}"
        )

        if stale.exists() and stale != keep_path:

            stale.unlink()


# =========================================
# Public Entrypoint
# =========================================

def select_overall_champion(dataset_name):

    """
    Compare available per-family champions for the
    given dataset and write the overall winner to:
        artifacts/champion/{dataset_name}/
            {dataset_name}_champion{.pkl|.pt}
            {dataset_name}_champion_metadata.json
    """

    dataset_champion_dir = CHAMPION_DIR / dataset_name

    if not dataset_champion_dir.exists():

        raise FileNotFoundError(
            f"No champion directory for dataset: "
            f"{dataset_name}"
        )

    # =========================================
    # Collect Candidates
    # =========================================

    candidates = []

    for family in CANDIDATE_FAMILIES:

        candidate = _load_candidate(
            dataset_champion_dir, family
        )

        if candidate is None:

            logger.info(
                f"No {family} champion found for "
                f"{dataset_name} — skipping."
            )

            continue

        candidates.append(candidate)

        logger.info(
            f"{family} champion PR-AUC: "
            f"{candidate['pr_auc']:.4f}"
        )

    if not candidates:

        raise ValueError(
            f"No champion candidates available for "
            f"{dataset_name}"
        )

    # =========================================
    # Select Winner
    # =========================================

    winner = max(candidates, key=lambda c: c["pr_auc"])

    logger.info(
        f"Overall {dataset_name} champion: "
        f"{winner['champion_type']} "
        f"(PR-AUC={winner['pr_auc']:.4f})"
    )

    # =========================================
    # Persist Overall Champion
    # =========================================

    winner_extension = winner["model_path"].suffix

    overall_model_path = (
        dataset_champion_dir
        / f"{dataset_name}_champion{winner_extension}"
    )

    overall_metadata_path = (
        dataset_champion_dir
        / f"{dataset_name}_champion_metadata.json"
    )

    _clear_stale_overall_models(
        dataset_champion_dir,
        dataset_name,
        keep_path=overall_model_path,
    )

    shutil.copy(winner["model_path"], overall_model_path)

    shutil.copy(winner["metadata_path"], overall_metadata_path)

    logger.info(
        f"Wrote overall champion -> "
        f"{overall_model_path}"
    )

    return {
        "champion_type": winner["champion_type"],
        "model_path": overall_model_path,
        "metadata_path": overall_metadata_path,
        "pr_auc": winner["pr_auc"],
    }
