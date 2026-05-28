#  src/tabular_ml/features/elliptic_preprocessing.py
from sklearn.model_selection import train_test_split


def split_elliptic_data(df):

    # =========================================
    # Remove Non-Feature Columns
    # =========================================

    X = df.drop(columns=["txId", "class", "label"])

    y = df["label"]

    # =========================================
    # Train/Test Split
    # =========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return (X_train, X_test, y_train, y_test)
