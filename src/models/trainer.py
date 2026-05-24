# src/models/trainer.py

"""
Purpose:
Central orchestration engine for
fraud detection model training,
evaluation, experiment tracking,
and artifact persistence.
"""

from pathlib import Path
from datetime import datetime
import time

import mlflow
import mlflow.sklearn

from src.data.load_data import load_dataset

from src.features.preprocessing import split_data

from src.features.sampling import get_sampling_methods

from src.config.model_config import get_models

from src.models.pipeline import build_pipeline

from src.models.registry import save_pipeline

from src.evaluation.evaluate import evaluate_model, apply_threshold

from src.evaluation.cross_validation import run_cross_validation

from src.evaluation.plots import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_feature_importance,
    plot_threshold_metrics,
)

from src.tracking.experiment_tracker import save_experiment_result

from src.utils.metadata_manager import save_metadata

from src.evaluation.thresholds import evaluate_thresholds

import pandas as pd

from src.models.model_selector import select_best_model

from src.models.champion import save_champion_model

from src.config.settings import (
    PLOTS_DIR,
    THRESHOLD_DIR,
    DEFAULT_THRESHOLD,
    THRESHOLD_CANDIDATES,
    EXPERIMENT_NAME,
)

from src.utils.logger import logger

mlflow.set_experiment(EXPERIMENT_NAME)


def run_training():

    logger.info("Loading dataset...")

    df = load_dataset()

    logger.info("Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(df)

    fraud_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = get_models(fraud_ratio)

    sampling_methods = get_sampling_methods()

    thresholds = THRESHOLD_CANDIDATES

    for sampling_name, sampler in sampling_methods.items():

        logger.info("\n" + "=" * 60)

        logger.info(f"Sampling strategy: {sampling_name}")

        logger.info("=" * 60)

        for model_name, model in models.items():

            run_name = f"{model_name}_{sampling_name}"

            with mlflow.start_run(run_name=run_name):

                logger.info("\n" + "-" * 60)

                logger.info(f"Training model: {model_name}")

                logger.info("-" * 60)

                pipeline = build_pipeline(model=model, sampler=sampler)

                logger.info("Running cross-validation...")

                cv_results = run_cross_validation(pipeline, X_train, y_train)

                cv_metrics = {
                    # -----------------------------------------
                    # Mean Metrics
                    # -----------------------------------------
                    "cv_precision_mean": cv_results["test_precision"].mean(),
                    "cv_recall_mean": cv_results["test_recall"].mean(),
                    "cv_f1_mean": cv_results["test_f1"].mean(),
                    "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
                    "cv_pr_auc_mean": cv_results["test_pr_auc"].mean(),
                    # -----------------------------------------
                    # Standard Deviations
                    # -----------------------------------------
                    "cv_precision_std": cv_results["test_precision"].std(),
                    "cv_recall_std": cv_results["test_recall"].std(),
                    "cv_f1_std": cv_results["test_f1"].std(),
                    "cv_roc_auc_std": cv_results["test_roc_auc"].std(),
                    "cv_pr_auc_std": cv_results["test_pr_auc"].std(),
                }

                start_time = time.time()

                pipeline.fit(X_train, y_train)

                training_time = time.time() - start_time

                y_prob = pipeline.predict_proba(X_test)[:, 1]

                threshold_results = evaluate_thresholds(y_test, y_prob, thresholds)

                y_pred = apply_threshold(y_prob, threshold=DEFAULT_THRESHOLD)

                metrics = evaluate_model(y_test, y_pred, y_prob)

                pr_plot_path = PLOTS_DIR / f"{run_name}_pr_curve.png"

                plot_precision_recall_curve(y_test, y_prob, pr_plot_path)

                roc_plot_path = PLOTS_DIR / f"{run_name}_roc_curve.png"

                plot_roc_curve(y_test, y_prob, roc_plot_path)

                trained_model = pipeline.named_steps["model"]

                if model_name in ["random_forest", "xgboost"]:

                    feature_plot_path = PLOTS_DIR / f"{run_name}_feature_importance.png"

                    plot_feature_importance(
                        trained_model, X_train.columns, feature_plot_path
                    )

                    mlflow.log_artifact(feature_plot_path)

                experiment_result = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "sampling": sampling_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "threshold": DEFAULT_THRESHOLD,
                    "training_time_sec": round(training_time, 4),
                    **cv_metrics,
                }

                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "sampling_strategy": sampling_name,
                    "threshold": DEFAULT_THRESHOLD,
                    "training_time_sec": round(training_time, 4),
                    "metrics": metrics,
                    "cross_validation_metrics": cv_metrics,
                    "feature_names": list(X_train.columns),
                    "model_parameters": trained_model.get_params(),
                    "dataset": {
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "fraud_ratio": float(fraud_ratio),
                    },
                    "pipeline_steps": list(pipeline.named_steps.keys()),
                    "experiment_version": "v1",
                }

                metadata_filename = f"{model_name}_{sampling_name}_metadata.json"

                save_metadata(metadata, metadata_filename)

                save_experiment_result(experiment_result)

                metadata_path = f"artifacts/metadata/{metadata_filename}"

                mlflow.log_artifact(metadata_path)

                threshold_path = THRESHOLD_DIR / f"{run_name}_thresholds.csv"

                threshold_results.to_csv(threshold_path, index=False)
                mlflow.log_artifact(threshold_path)

                threshold_plot_path = PLOTS_DIR / f"{run_name}_threshold_metrics.png"

                plot_threshold_metrics(threshold_results, threshold_plot_path)

                mlflow.log_artifact(threshold_plot_path)

                mlflow.log_params(
                    {
                        "model": model_name,
                        "sampling": sampling_name,
                        "threshold": DEFAULT_THRESHOLD,
                    }
                )

                mlflow.log_metrics(
                    {**metrics, **cv_metrics, "training_time_sec": training_time}
                )

                mlflow.log_artifact(pr_plot_path)

                mlflow.log_artifact(roc_plot_path)

                model_filename = f"{run_name}.pkl"

                save_pipeline(pipeline, model_filename)

                mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # -----------------------------------------
    # Best Model Selection
    # -----------------------------------------

    logger.info("\nSelecting best model...")

    results_df = pd.read_csv("artifacts/metrics/experiment_results.csv")

    best_model = select_best_model(results_df, min_precision=0.80)

    best_run_name = f"{best_model['model']}_" f"{best_model['sampling']}"

    best_model_path = Path("artifacts/models") / f"{best_run_name}.pkl"

    best_metadata_path = Path("artifacts/metadata") / f"{best_run_name}_metadata.json"

    save_champion_model(best_model_path, best_metadata_path)

    logger.info("\nBest Model Selected:")

    logger.info(best_model)
    logger.info("\nTraining complete.")
