import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.config.settings import DEFAULT_THRESHOLD

from src.utils.logger import logger


def apply_threshold(probabilities, threshold=DEFAULT_THRESHOLD):

    return np.where(probabilities >= threshold, 1, 0)


def evaluate_predictions(y_true, y_prob, threshold=DEFAULT_THRESHOLD, verbose=True):

    predictions = apply_threshold(y_prob, threshold)

    metrics = {
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1_score": f1_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }

    if verbose:

        logger.info("\nConfusion Matrix:")

        logger.info(confusion_matrix(y_true, predictions))

        logger.info("\nClassification Report:")

        logger.info(classification_report(y_true, predictions))

        logger.info("\nMetrics:")

        for metric, value in metrics.items():

            logger.info(f"{metric}: " f"{value:.4f}")

    return metrics
