import torch

import torch.nn.functional as F

import mlflow

from src.gnn.data.build_graph_dataset import build_graph_dataset

from src.gnn.models.gcn_model import GCN

from src.gnn.config.gcn_config import GCN_CONFIG

from src.gnn.evaluation.evaluate_gnn import evaluate_gnn_model

from src.gnn.evaluation.thresholds import evaluate_thresholds

from src.gnn.models.registry import save_gnn_model

from src.gnn.tracking.mlflow_logger import log_gnn_metrics

mlflow.set_experiment("gnn_fraud_detection")


def train_gcn():

    # =========================================
    # Device
    # =========================================

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # =========================================
    # Load Graph Dataset
    # =========================================

    data = build_graph_dataset()

    data = data.to(device)

    # =========================================
    #  Model
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

    best_val_f1 = 0

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

                val_metrics = evaluate_gnn_model(
                    logits,
                    data.y,
                    data.val_mask,
                    threshold=GCN_CONFIG["threshold"],
                    verbose=False,
                )

            val_f1 = val_metrics["f1_score"]

            # =========================================
            # Save Best Model
            # =========================================

            if val_f1 > best_val_f1:

                best_val_f1 = val_f1

                save_gnn_model(model, "best_gcn_model.pt")

            if epoch % 10 == 0:

                print(
                    f"Epoch: {epoch}"
                    f" | Loss: {loss:.4f}"
                    f" | Val F1: "
                    f"{val_f1:.4f}"
                )

        # =========================================
        # Final Evaluation
        # =========================================

        model.eval()

        with torch.no_grad():

            final_logits = model(data.x, data.edge_index)

            test_metrics = evaluate_gnn_model(
                final_logits,
                data.y,
                data.test_mask,
                threshold=GCN_CONFIG["threshold"],
                verbose=True,
            )

        print("\nFinal Test Metrics")

        print(test_metrics)

        # =========================================
        # Threshold Analysis
        # =========================================

        threshold_results = evaluate_thresholds(
            final_logits,
            data.y,
            data.test_mask,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )

        print("\nThreshold Results")

        print(threshold_results)

        # =========================================
        # MLflow Logging
        # =========================================

        mlflow.log_params(GCN_CONFIG)

        log_gnn_metrics(test_metrics)


if __name__ == "__main__":

    train_gcn()
