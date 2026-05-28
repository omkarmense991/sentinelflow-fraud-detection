import torch

import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_gnn_model(logits, labels, mask, threshold=0.5, verbose=False):

    # =========================================
    # Mask Selection
    # =========================================

    logits = logits[mask]

    labels = labels[mask]

    # =========================================
    # Probabilities
    # =========================================

    probabilities = F.softmax(logits, dim=1)[:, 1]

    probabilities_np = probabilities.detach().cpu().numpy()

    labels_np = labels.detach().cpu().numpy()

    # =========================================
    # Threshold Predictions
    # =========================================

    predictions = (probabilities_np >= threshold).astype(int)

    # =========================================
    # Metrics
    # =========================================

    metrics = {
        "precision": precision_score(labels_np, predictions),
        "recall": recall_score(labels_np, predictions),
        "f1_score": f1_score(labels_np, predictions),
        "roc_auc": roc_auc_score(labels_np, probabilities_np),
        "pr_auc": average_precision_score(labels_np, probabilities_np),
    }

    # =========================================
    # Optional Verbose Output
    # =========================================

    if verbose:

        print("\nConfusion Matrix")

        print(confusion_matrix(labels_np, predictions))

        print("\nClassification Report")

        print(classification_report(labels_np, predictions))

    return metrics
