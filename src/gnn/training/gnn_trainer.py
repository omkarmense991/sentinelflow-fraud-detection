# src/gnn/training/train_gnn.py

from datetime import datetime

import torch
import torch.nn.functional as F

import mlflow

from src.gnn.data.build_graph_dataset import build_graph_dataset

from src.gnn.models.registry import get_gnn_models, save_gnn_model

from src.utils.extract_probabilities_gnn import extract_probabilities

from src.config.gnn_config import GNN_CONFIG

from src.config.settings import (
    PLOTS_DIR,
    THRESHOLD_DIR,
    DEFAULT_THRESHOLD,
    THRESHOLD_CANDIDATES,
)

from src.evaluation.metrics import evaluate_predictions

from src.evaluation.thresholds import evaluate_thresholds

from src.evaluation.plots import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_metrics,
)

from src.tracking.experiment_tracker import save_experiment_result

from src.utils.metadata_manager import save_metadata

from src.models.champion import save_champion_model

from src.utils.logger import logger

# =========================================
# MLflow Experiment
# =========================================

from src.config.settings import EXPERIMENT_NAME

from pathlib import Path
import pandas as pd

from src.gnn.training.select_best_gnn_model import (
    select_best_gnn_model,
)

experiment_name = f"{EXPERIMENT_NAME}_elliptic"

mlflow.set_experiment(experiment_name)


def train_gnn_models():

    # =========================================
    # Device
    # =========================================

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    # =========================================
    # Dataset
    # =========================================

    logger.info("Loading graph dataset...")

    data = build_graph_dataset()

    data = data.to(device)

    logger.info(
        f"Graph Loaded | " f"Nodes: {data.num_nodes} | " f"Edges: {data.num_edges}"
    )

    # =========================================
    # Artifact Directories
    # =========================================

    dataset_name = "elliptic"

    dataset_plot_dir = PLOTS_DIR / dataset_name

    dataset_plot_dir.mkdir(parents=True, exist_ok=True)

    dataset_threshold_dir = THRESHOLD_DIR / dataset_name

    dataset_threshold_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # MLflow Run
    # =========================================

    gnn_models = get_gnn_models()

    for model_name, model_class in gnn_models.items():

        best_val_f1 = float("-inf")

        best_model_path = None

        logger.info(f"\nTraining GNN architecture: {model_name}")

        model = model_class(
            input_dim=data.num_node_features,
            hidden_dim=GNN_CONFIG["hidden_dim"],
            output_dim=2,
            dropout=GNN_CONFIG["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=GNN_CONFIG["learning_rate"],
            weight_decay=GNN_CONFIG["weight_decay"],
        )

        run_name = f"{model_name}_baseline"

        with mlflow.start_run(run_name=run_name):

            # =========================================
            # Training Loop
            # =========================================

            for epoch in range(1, GNN_CONFIG["epochs"] + 1):

                model.train()

                optimizer.zero_grad()

                logits = model(data.x, data.edge_index)

                loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

                loss.backward()

                optimizer.step()

                # =========================================
                # Validation
                # =========================================

                model.eval()

                with torch.no_grad():

                    val_labels, val_probabilities = extract_probabilities(
                        logits, data.y, data.val_mask
                    )

                    val_metrics = evaluate_predictions(
                        y_true=val_labels,
                        y_prob=val_probabilities,
                        threshold=DEFAULT_THRESHOLD,
                        verbose=False,
                    )

                val_f1 = val_metrics["f1_score"]

                # =========================================
                # Save Best Model
                # =========================================

                if val_f1 > best_val_f1:

                    best_val_f1 = val_f1

                    model_filename = f"{model_name}_model.pt"

                    best_model_path = save_gnn_model(
                        model,
                        model_filename,
                        dataset_name,
                    )

                # =========================================
                # Epoch Logging
                # =========================================

                if epoch % 10 == 0:

                    logger.info(
                        f"Epoch: {epoch}"
                        f" | Loss: {loss:.4f}"
                        f" | Val F1: "
                        f"{val_f1:.4f}"
                    )

            # =========================================
            # Reload Best Validation Checkpoint
            # =========================================

            logger.info("Reloading best validation checkpoint for final evaluation...")

            best_model = model_class(
                input_dim=data.num_node_features,
                hidden_dim=GNN_CONFIG["hidden_dim"],
                output_dim=2,
                dropout=GNN_CONFIG["dropout"],
            ).to(device)

            best_model.load_state_dict(torch.load(best_model_path, map_location=device))

            best_model.eval()

            # =========================================
            # Final Evaluation
            # =========================================

            logger.info("Running final test evaluation...")

            with torch.no_grad():

                final_logits = best_model(data.x, data.edge_index)

                test_labels, test_probabilities = extract_probabilities(
                    final_logits, data.y, data.test_mask
                )

                test_metrics = evaluate_predictions(
                    y_true=test_labels,
                    y_prob=test_probabilities,
                    threshold=DEFAULT_THRESHOLD,
                    verbose=True,
                )

            logger.info("\nFinal Test Metrics")

            logger.info(test_metrics)

            # =========================================
            # Threshold Analysis
            # =========================================

            threshold_results = evaluate_thresholds(
                y_true=test_labels,
                y_prob=test_probabilities,
                thresholds=THRESHOLD_CANDIDATES,
            )

            logger.info("\nThreshold Results")

            logger.info(threshold_results)

            # =========================================
            # Save Threshold CSV
            # =========================================

            threshold_path = (
                dataset_threshold_dir / f"{model_name}_threshold_analysis.csv"
            )

            threshold_results.to_csv(threshold_path, index=False)

            # =========================================
            # Threshold Plot
            # =========================================

            threshold_plot_path = (
                dataset_plot_dir / f"{model_name}_threshold_metrics.png"
            )

            plot_threshold_metrics(threshold_results, threshold_plot_path)

            # =========================================
            # PR Curve
            # =========================================

            pr_plot_path = dataset_plot_dir / f"{model_name}_pr_curve.png"

            plot_precision_recall_curve(test_labels, test_probabilities, pr_plot_path)

            # =========================================
            # ROC Curve
            # =========================================

            roc_plot_path = dataset_plot_dir / f"{model_name}_roc_curve.png"

            plot_roc_curve(test_labels, test_probabilities, roc_plot_path)

            # =========================================
            # Experiment Result CSV
            # =========================================

            experiment_result = {
                "timestamp": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "model": model_name,
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "threshold": DEFAULT_THRESHOLD,
                "best_val_f1": best_val_f1,
                "architecture": model_name,
            }

            save_experiment_result(experiment_result, dataset_name)

            # =========================================
            # Metadata
            # =========================================

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metrics": test_metrics,
                "selection_metrics": {
                    "validation_f1": best_val_f1,
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_f1": test_metrics["f1_score"],
                },
                "threshold": DEFAULT_THRESHOLD,
                "best_val_f1": best_val_f1,
                "epochs": GNN_CONFIG["epochs"],
                "hidden_dim": GNN_CONFIG["hidden_dim"],
                "dropout": GNN_CONFIG["dropout"],
                "learning_rate": GNN_CONFIG["learning_rate"],
                "architecture": model_name,
            }

            metadata_path = save_metadata(
                metadata,
                filename=f"{model_name}_metadata.json",
                dataset_name=dataset_name,
            )

            mlflow.log_artifact(metadata_path)

            # =========================================
            # MLflow Logging
            # =========================================

            mlflow.log_params(
                {
                    **GNN_CONFIG,
                    "architecture": model_name,
                }
            )

            mlflow.log_metrics(test_metrics)

            # =========================================
            # MLflow Artifacts
            # =========================================

            mlflow.log_artifact(best_model_path)

            mlflow.log_artifact(threshold_path)

            mlflow.log_artifact(threshold_plot_path)

            mlflow.log_artifact(pr_plot_path)

            mlflow.log_artifact(roc_plot_path)

            logger.info(f"\n{model_name} training complete.")

    logger.info("\nSelecting best GNN model...")

    results_path = Path("artifacts/metrics") / dataset_name / "experiment_results.csv"

    results_df = pd.read_csv(results_path)

    best_model = select_best_gnn_model(
        results_df,
        dataset_name,
    )

    best_model_name = best_model["model"]

    best_model_path = (
        Path("artifacts/models") / dataset_name / f"{best_model_name}_model.pt"
    )

    best_metadata_path = (
        Path("artifacts/metadata") / dataset_name / f"{best_model_name}_metadata.json"
    )

    save_champion_model(
        best_model_path,
        best_metadata_path,
        dataset_name,
        champion_type="gnn",
    )

    logger.info("\nBest GNN Model Selected:")
    logger.info(best_model)
