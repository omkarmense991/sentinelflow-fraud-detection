# src/config/model_config.py

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def get_models(fraud_ratio):

    return {
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
            scale_pos_weight=fraud_ratio,
        ),
    }
