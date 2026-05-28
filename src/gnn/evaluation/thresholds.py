import pandas as pd

import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_thresholds(logits, labels, mask, thresholds):

    logits = logits[mask]

    labels = labels[mask]

    probabilities = F.softmax(logits, dim=1)[:, 1]

    probabilities = probabilities.detach().cpu().numpy()

    labels = labels.detach().cpu().numpy()

    results = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(labels, predictions),
                "recall": recall_score(labels, predictions),
                "f1_score": f1_score(labels, predictions),
            }
        )

    return pd.DataFrame(results)
