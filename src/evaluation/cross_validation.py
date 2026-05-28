# src/evalution/cross_validation.py

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.evaluation.scorers import (
    precision_scorer,
    recall_scorer,
    f1_scorer,
)


def run_cross_validation(pipeline, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "precision": precision_scorer,
        "recall": recall_scorer,
        "f1": f1_scorer,
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return results
