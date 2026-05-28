# src/evaluation/thresholds.py

import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.evaluation.metrics import apply_threshold


def evaluate_thresholds(y_true, y_prob, thresholds):

    results = []

    for threshold in thresholds:

        y_pred = apply_threshold(y_prob, threshold)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )

    return pd.DataFrame(results)
