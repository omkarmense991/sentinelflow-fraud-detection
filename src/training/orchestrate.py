# src/training/orchestrate.py

"""
Purpose:
Unified training entrypoint for SentinelFlow.

Usage:
    # Train all families for a dataset, then elect overall champion
    python -m src.training.orchestrate --dataset elliptic

    # Train only one family (overall champion is still re-elected)
    python -m src.training.orchestrate --dataset elliptic --family tabular
    python -m src.training.orchestrate --dataset elliptic --family gnn
"""

import argparse
import subprocess
import sys

from src.training.dataset_registry import DATASET_REGISTRY

from src.models.select_overall_champion import select_overall_champion

from src.utils.logger import logger


# =========================================
# Supported Families
# =========================================

ALL_FAMILIES = ("tabular", "gnn")


# =========================================
# Subprocess Family Runner
# =========================================
# When training more than one family in a
# single invocation, each family is executed
# in a fresh Python subprocess.  This is the
# only reliable way to release memory between
# families on macOS: the tabular stack
# (sklearn + xgboost + joblib + pandas) keeps
# trained pipelines and large DataFrames
# resident, and the 658MB elliptic features
# CSV cannot be loaded on top of that plus
# PyTorch + MPS without swap-thrashing.

def _run_family_in_subprocess(dataset_name, family):

    cmd = [
        sys.executable,
        "-m",
        "src.training.orchestrate",
        "--dataset",
        dataset_name,
        "--family",
        family,
    ]

    logger.info(f"Spawning subprocess: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


# =========================================
# Argument Parser
# =========================================

def _build_parser():

    parser = argparse.ArgumentParser(
        prog="orchestrate",
        description="SentinelFlow unified training orchestrator",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to train on",
    )

    parser.add_argument(
        "--family",
        required=False,
        choices=list(ALL_FAMILIES),
        default=None,
        help="Model family to train (default: all families)",
    )

    return parser


# =========================================
# Core Orchestration
# =========================================

def orchestrate(dataset_name, family=None):

    # =========================================
    # Validate Dataset
    # =========================================

    if dataset_name not in DATASET_REGISTRY:

        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_entry = DATASET_REGISTRY[dataset_name]

    # =========================================
    # Resolve Which Families To Run
    # =========================================

    if family is not None:

        if family not in dataset_entry:

            raise ValueError(
                f"Family '{family}' not registered "
                f"for dataset '{dataset_name}'. "
                f"Available: {list(dataset_entry.keys())}"
            )

        families_to_run = [family]

    else:

        # Run in a stable order: tabular first, gnn second
        families_to_run = [
            f for f in ALL_FAMILIES if f in dataset_entry
        ]

    # =========================================
    # Training
    # =========================================

    logger.info("=" * 60)

    logger.info(
        f"Orchestrator | dataset={dataset_name} "
        f"| families={families_to_run}"
    )

    logger.info("=" * 60)

    # When the caller asked for a single family, run
    # it in-process; when running multiple families,
    # spawn a subprocess per family so the tabular
    # stack is fully torn down before PyTorch + the
    # 658MB elliptic features CSV are loaded.

    run_in_subprocess = family is None and len(families_to_run) > 1

    for fam in families_to_run:

        logger.info("\n" + "=" * 60)

        logger.info(f"Training family: {fam}")

        logger.info("=" * 60)

        if run_in_subprocess:

            _run_family_in_subprocess(dataset_name, fam)

        else:

            train_fn = dataset_entry[fam]

            train_fn()

        logger.info(f"Family {fam} training complete.")

    # =========================================
    # Overall Champion Election
    # =========================================
    # Both trainers already call select_overall_champion
    # internally, but when --family is used only one
    # trainer ran.  Re-running here ensures the overall
    # champion is always up-to-date regardless of which
    # families were trained in this session.

    logger.info("\n" + "=" * 60)

    logger.info("Electing overall dataset champion...")

    logger.info("=" * 60)

    try:

        result = select_overall_champion(dataset_name)

        logger.info(
            f"Overall champion: "
            f"{result['champion_type']} "
            f"(PR-AUC={result['pr_auc']:.4f})"
        )

        logger.info(
            f"Champion model -> "
            f"{result['model_path']}"
        )

    except (FileNotFoundError, ValueError) as exc:

        # Not all families have been trained yet — safe to skip.
        logger.warning(
            f"Overall champion election skipped: {exc}"
        )

    logger.info("\n" + "=" * 60)

    logger.info(f"Orchestration complete for: {dataset_name}")

    logger.info("=" * 60)


# =========================================
# CLI Entry
# =========================================

if __name__ == "__main__":

    parser = _build_parser()

    args = parser.parse_args()

    orchestrate(
        dataset_name=args.dataset,
        family=args.family,
    )
