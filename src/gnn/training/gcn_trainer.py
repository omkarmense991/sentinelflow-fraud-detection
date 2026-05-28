# src/gnn/training/train_gcn.py

from datetime import datetime

import torch
import torch.nn.functional as F

import mlflow

from src.gnn.data.build_graph_dataset import build_graph_dataset

from src.gnn.models.gcn_model import GCN

from src.gnn.models.registry import save_gnn_model

from src.utils.extract_probabilities_gnn import extract_probabilities

from src.config.gcn_config import GCN_CONFIG

from src.config.settings import PLOTS_DIR, THRESHOLD_DIR

from src.evaluation.metrics import evaluate_predictions

from src.evaluation.thresholds import evaluate_thresholds

from src.evaluation.plots import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_metrics,
)

from src.tracking.experiment_tracker import save_experiment_result

from src.utils.metadata_manager import save_metadata

from src.utils.logger import logger

# =========================================
# MLflow Experiment
# =========================================

from src.config.settings import EXPERIMENT_NAME

experiment_name = f"{EXPERIMENT_NAME}_elliptic"

mlflow.set_experiment(experiment_name)


def train_gcn():

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
    # Model
    # =========================================

    model = GCN(
        input_dim=data.num_node_features,
        hidden_dim=GCN_CONFIG["hidden_dim"],
        output_dim=2,
        dropout=GCN_CONFIG["dropout"],
    ).to(device)

    # =========================================
    # Optimizer
    # =========================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GCN_CONFIG["learning_rate"],
        weight_decay=GCN_CONFIG["weight_decay"],
    )

    # =========================================
    # Artifact Directories
    # =========================================

    dataset_name = "elliptic"

    dataset_plot_dir = PLOTS_DIR / dataset_name

    dataset_plot_dir.mkdir(parents=True, exist_ok=True)

    dataset_threshold_dir = THRESHOLD_DIR / dataset_name

    dataset_threshold_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0

    # =========================================
    # MLflow Run
    # =========================================

    with mlflow.start_run(run_name="gcn_baseline"):

        # =========================================
        # Training Loop
        # =========================================

        for epoch in range(1, GCN_CONFIG["epochs"] + 1):

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
                    threshold=GCN_CONFIG["threshold"],
                    verbose=False,
                )

            val_f1 = val_metrics["f1_score"]

            # =========================================
            # Save Best Model
            # =========================================

            if val_f1 > best_val_f1:

                best_val_f1 = val_f1

                model_filename = "gcn_model.pt"

                model_path = save_gnn_model(model, model_filename)

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
        # Final Evaluation
        # =========================================

        logger.info("Running final test evaluation...")

        model.eval()

        with torch.no_grad():

            final_logits = model(data.x, data.edge_index)

            test_labels, test_probabilities = extract_probabilities(
                final_logits, data.y, data.test_mask
            )

            test_metrics = evaluate_predictions(
                y_true=test_labels,
                y_prob=test_probabilities,
                threshold=GCN_CONFIG["threshold"],
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
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        logger.info("\nThreshold Results")

        logger.info(threshold_results)

        # =========================================
        # Save Threshold CSV
        # =========================================

        threshold_path = dataset_threshold_dir / "gcn_threshold_analysis.csv"

        threshold_results.to_csv(threshold_path, index=False)

        # =========================================
        # Threshold Plot
        # =========================================

        threshold_plot_path = dataset_plot_dir / "gcn_threshold_metrics.png"

        plot_threshold_metrics(threshold_results, threshold_plot_path)

        # =========================================
        # PR Curve
        # =========================================

        pr_plot_path = dataset_plot_dir / "gcn_pr_curve.png"

        plot_precision_recall_curve(test_labels, test_probabilities, pr_plot_path)

        # =========================================
        # ROC Curve
        # =========================================

        roc_plot_path = dataset_plot_dir / "gcn_roc_curve.png"

        plot_roc_curve(test_labels, test_probabilities, roc_plot_path)

        # =========================================
        # Experiment Result CSV
        # =========================================

        experiment_result = {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "model": "gcn",
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "threshold": GCN_CONFIG["threshold"],
            "best_val_f1": best_val_f1,
        }

        save_experiment_result(experiment_result, dataset_name)

        # =========================================
        # Metadata
        # =========================================

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_name": "gcn",
            "dataset_name": dataset_name,
            "metrics": test_metrics,
            "threshold": GCN_CONFIG["threshold"],
            "best_val_f1": best_val_f1,
            "epochs": GCN_CONFIG["epochs"],
            "hidden_dim": GCN_CONFIG["hidden_dim"],
            "dropout": GCN_CONFIG["dropout"],
            "learning_rate": GCN_CONFIG["learning_rate"],
        }

        save_metadata(
            metadata, filename="gcn_model_metadata.json", dataset_name=dataset_name
        )

        # =========================================
        # MLflow Logging
        # =========================================

        mlflow.log_params(GCN_CONFIG)

        mlflow.log_metrics(test_metrics)

        # =========================================
        # MLflow Artifacts
        # =========================================

        mlflow.log_artifact(model_path)

        mlflow.log_artifact(threshold_path)

        mlflow.log_artifact(threshold_plot_path)

        mlflow.log_artifact(pr_plot_path)

        mlflow.log_artifact(roc_plot_path)

        logger.info("\nGCN training complete.")
