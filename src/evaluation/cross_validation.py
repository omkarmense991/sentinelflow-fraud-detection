from sklearn.model_selection import StratifiedKFold, cross_validate


def run_cross_validation(pipeline, X, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return results
