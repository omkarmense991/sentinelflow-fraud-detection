from pathlib import Path
from datetime import datetime
import time

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.data.load_data import load_dataset

from src.features.preprocessing import (
    split_data,
    scale_features,
)

from src.features.sampling import (
    apply_random_oversampling,
    apply_random_undersampling,
    apply_smote,
)

from src.evaluation.evaluate import (
    evaluate_model,
    apply_threshold,
)

from src.evaluation.plots import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_feature_importance,
)

from src.utils.experimnent_tracker import (
    save_experiment_result,
)

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = Path("artifacts/metrics/experiment_results.csv")

THRESHOLD = 0.3


def train():

    # Clean old metrics file
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()

    print("Loading dataset...")
    df = load_dataset()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    sampling_methods = {
        "none": None,
        "undersampling": apply_random_undersampling,
        "oversampling": apply_random_oversampling,
        "smote": apply_smote,
    }

    for sampling_name, sampling_function in sampling_methods.items():

        print("\n" + "=" * 60)
        print(f"Sampling strategy: {sampling_name}")
        print("=" * 60)

        # -----------------------------------------
        # Apply sampling ONLY on training data
        # -----------------------------------------

        if sampling_function is not None:

            X_resampled, y_resampled = sampling_function(X_train, y_train)

        else:

            X_resampled, y_resampled = (X_train, y_train)

        # -----------------------------------------
        # Scale AFTER sampling
        # -----------------------------------------

        scaler, X_train_scaled, X_test_scaled = scale_features(X_resampled, X_test)

        for model_name, model in models.items():

            print("\n" + "-" * 60)
            print(f"Training model: {model_name}")
            print("-" * 60)

            start_time = time.time()

            model.fit(X_train_scaled, y_resampled)

            training_time = time.time() - start_time

            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            y_pred = apply_threshold(y_prob, threshold=THRESHOLD)

            # -----------------------------------------
            # Save PR Curve
            # -----------------------------------------

            pr_plot_path = PLOTS_DIR / f"{model_name}_{sampling_name}_pr_curve.png"

            plot_precision_recall_curve(y_test, y_prob, pr_plot_path)

            # -----------------------------------------
            # Save ROC Curve
            # -----------------------------------------

            roc_plot_path = PLOTS_DIR / f"{model_name}_{sampling_name}_roc_curve.png"

            plot_roc_curve(y_test, y_prob, roc_plot_path)

            # -----------------------------------------
            # Feature Importance
            # -----------------------------------------

            if model_name in ["random_forest", "xgboost"]:

                feature_plot_path = (
                    PLOTS_DIR / f"{model_name}_{sampling_name}_feature_importance.png"
                )

                plot_feature_importance(model, X_train.columns, feature_plot_path)

            # -----------------------------------------
            # Evaluation
            # -----------------------------------------

            metrics = evaluate_model(y_test, y_pred, y_prob)

            print("\nMetrics Summary:")
            print(metrics)

            # -----------------------------------------
            # Save Experiment Metadata
            # -----------------------------------------

            experiment_result = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "sampling": sampling_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "threshold": THRESHOLD,
                "train_samples": len(X_resampled),
                "fraud_samples": int(sum(y_resampled)),
                "training_time_sec": round(training_time, 4),
            }

            save_experiment_result(experiment_result)

            # -----------------------------------------
            # Save Model
            # -----------------------------------------

            model_filename = f"{model_name}_{sampling_name}.pkl"

            scaler_filename = f"{model_name}_{sampling_name}_scaler.pkl"

            joblib.dump(model, MODEL_DIR / model_filename)

            joblib.dump(scaler, MODEL_DIR / scaler_filename)

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
