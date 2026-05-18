from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(y_true, y_pred, y_prob):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_true, y_prob))
