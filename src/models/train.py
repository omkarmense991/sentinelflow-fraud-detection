from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_dataset
from src.features.preprocessing import split_data, scale_features
from src.evaluation.evaluate import evaluate_model, apply_threshold

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train():

    print("Loading dataset...")
    df = load_dataset()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Scaling features...")
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print("Training Logistic Regression model...")

    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train_scaled, y_train)

    print("Generating predictions...")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    # lower threshold to increase fraud sensitivity
    y_pred = apply_threshold(y_prob, 0.3)

    print("Evaluating model...")

    evaluate_model(y_test, y_pred, y_prob)

    print("Saving model artifacts...")

    joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    print("Training complete.")


if __name__ == "__main__":
    train()
