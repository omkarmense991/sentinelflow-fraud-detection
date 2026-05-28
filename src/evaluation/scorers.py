# src/evaluation/scorers.py

"""
Purpose:
Sklearn-compatible scorers that evaluate
precision, recall, and F1 at the deployment
threshold (DEFAULT_THRESHOLD) instead of the
implicit 0.5 used by sklearn's built-in
"precision" / "recall" / "f1" strings.

PR-AUC and ROC-AUC remain threshold-independent
and continue to use sklearn's probability-based
scorers.
"""

import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)

from src.config.settings import DEFAULT_THRESHOLD


# =========================================
# Internal Helper
# =========================================

def _fraud_probabilities(y_prob):

    # make_scorer(response_method="predict_proba")
    # hands us the full (N, 2) probability matrix
    # for binary classifiers. Extract the
    # positive-class (fraud) column.

    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:

        return y_prob[:, 1]

    return y_prob


# =========================================
# Thresholded Metrics
# =========================================

def precision_at_threshold(y_true, y_prob):

    probabilities = _fraud_probabilities(y_prob)

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    return precision_score(y_true, predictions, zero_division=0)


def recall_at_threshold(y_true, y_prob):

    probabilities = _fraud_probabilities(y_prob)

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    return recall_score(y_true, predictions, zero_division=0)


def f1_at_threshold(y_true, y_prob):

    probabilities = _fraud_probabilities(y_prob)

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    return f1_score(y_true, predictions, zero_division=0)


# =========================================
# Sklearn-Compatible Scorers
# =========================================

precision_scorer = make_scorer(
    precision_at_threshold,
    response_method="predict_proba",
)

recall_scorer = make_scorer(
    recall_at_threshold,
    response_method="predict_proba",
)

f1_scorer = make_scorer(
    f1_at_threshold,
    response_method="predict_proba",
)
