#src/evaluation/plots.py

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve

import pandas as pd


def plot_precision_recall_curve(y_true, y_prob, save_path):

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.savefig(save_path)

    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.savefig(save_path)

    plt.close()


def plot_feature_importance(model, feature_names, save_path):

    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(importance_df["feature"], importance_df["importance"])

    plt.gca().invert_yaxis()

    plt.title("Top Feature Importances")

    plt.savefig(save_path)

    plt.close()


def plot_threshold_metrics(threshold_df, save_path):

    plt.figure(figsize=(10, 6))

    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")

    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")

    plt.plot(threshold_df["threshold"], threshold_df["f1_score"], label="F1 Score")

    plt.xlabel("Threshold")

    plt.ylabel("Metric Score")

    plt.title("Threshold vs Metrics")

    plt.legend()

    plt.savefig(save_path)

    plt.close()
