from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

import numpy as np


def evaluate_model(y_true, y_pred, y_prob):

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_true, y_prob))

    print("\nPR-AUC Score:")
    print(average_precision_score(y_true, y_prob))


def apply_threshold(probabilities, threshold=0.5):

    return np.where(probabilities >= threshold, 1, 0)
