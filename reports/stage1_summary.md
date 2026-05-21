# SentinelFlow — Production-Style Fraud Detection ML System - Stage 1

## Project Overview

SentinelFlow is a production-style machine learning fraud detection system designed for financial transaction fraud analysis and realtime fraud scoring workflows.

The primary goal of this project was not only to train fraud classification models, but to design and implement a scalable and reproducible ML engineering platform that simulates real-world fraud detection systems used in fintech and banking environments.

The project evolved incrementally from baseline machine learning experimentation into a modular ML experimentation and model management platform with:

* reusable ML pipelines,
* imbalance-aware training workflows,
* experiment tracking,
* threshold analysis,
* cross-validation,
* model persistence,
* metadata management,
* and deployment-ready ML artifacts.

The long-term roadmap of the project includes:

* production inference APIs,
* realtime fraud prediction services,
* Dockerized deployment,
* PostgreSQL integration,
* and Graph Neural Network (GNN)-based relational fraud detection.

---

# Problem Understanding

## Financial Fraud Detection Challenges

Financial fraud detection is a highly challenging machine learning problem because fraudulent transactions represent an extremely small percentage of total transactions.

The dataset used in this project contains severe class imbalance where fraudulent transactions represent less than 0.2% of all records.

This creates several important ML engineering challenges:

* traditional accuracy metrics become misleading,
* fraud patterns are rare and difficult to generalize,
* false negatives cause direct financial loss,
* and false positives negatively impact customer experience.

A fraud detection system must therefore balance:

* fraud detection sensitivity,
* customer friction,
* operational investigation cost,
* and business risk.

---

## False Positives vs False Negatives

Fraud systems must carefully manage two types of prediction errors:

| Error Type     | Business Impact                                     |
| -------------- | --------------------------------------------------- |
| False Positive | Legitimate transaction incorrectly flagged as fraud |
| False Negative | Fraudulent transaction incorrectly allowed          |

False negatives are extremely expensive because they represent missed fraud and direct financial loss.

However, excessive false positives can also damage customer trust and overload fraud investigation teams.

As a result, fraud detection systems optimize operational tradeoffs rather than raw classification accuracy.

---

# Dataset Overview

The project uses the publicly available credit card fraud detection dataset containing anonymized transaction features.

Dataset characteristics:

| Property                | Value                              |
| ----------------------- | ---------------------------------- |
| Total Transactions      | 284,807                            |
| Fraudulent Transactions | 492                                |
| Fraud Ratio             | ~0.172%                            |
| Features                | PCA-transformed numerical features |
| Target Column           | `Class`                            |

The dataset is heavily imbalanced, making it highly suitable for studying real-world fraud detection workflows.

---

# System Architecture

The project was designed using modular ML engineering principles to simulate a production-grade ML training system.

Project structure:

```plaintext
src/
├── config/
├── data/
├── evaluation/
├── features/
├── models/
├── registry/
├── tracking/
└── utils/
```

## Architectural Goals

Key architectural objectives included:

* separation of concerns,
* reusable preprocessing pipelines,
* reproducible experimentation,
* centralized experiment tracking,
* deployment-ready artifact persistence,
* and maintainable ML orchestration.

The system architecture evolved incrementally as project complexity increased.

---

# Training Pipeline Architecture

The training workflow follows a modular ML pipeline architecture:

```plaintext
Raw Dataset
    ↓
Train/Test Split
    ↓
Cross Validation
    ↓
Pipeline:
    - Feature Scaling
    - Sampling Strategy
    - Model Training
    ↓
Probability Prediction
    ↓
Threshold-Based Classification
    ↓
Evaluation
    ↓
MLflow Experiment Tracking
    ↓
Artifact Persistence
```

---

# ML Pipeline Design

## Leakage Prevention

Preventing data leakage was a major design priority throughout the project.

Several important decisions were implemented to ensure leakage-safe training:

* scaling was performed inside reusable pipelines,
* sampling strategies were applied only on training folds,
* and cross-validation used stratified splitting to preserve fraud ratios across folds.

This ensured that the model never received information from validation or test data during training.

---

## Pipeline-Based Architecture

The project uses reusable `imblearn` pipelines that integrate:

* feature scaling,
* imbalance handling,
* and model training.

This approach provides several advantages:

* reproducible preprocessing,
* cleaner orchestration,
* leakage prevention,
* deployment consistency,
* and simplified inference workflows.

Persisting full pipelines also simplifies deployment because preprocessing and model inference remain tightly coupled.

---

# Models Evaluated

Three primary models were evaluated during experimentation.

| Model               | Purpose                           |
| ------------------- | --------------------------------- |
| Logistic Regression | Baseline probabilistic classifier |
| Random Forest       | Nonlinear ensemble model          |
| XGBoost             | High-performance boosting model   |

---

## Logistic Regression

Logistic Regression was used as an interpretable baseline model.

Advantages:

* simple,
* fast,
* probabilistic,
* and easy to interpret.

The model demonstrated strong fraud recall but generated large numbers of false positives due to aggressive fraud sensitivity.

---

## Random Forest

Random Forest was introduced to model nonlinear fraud patterns and feature interactions.

Advantages:

* nonlinear decision boundaries,
* strong robustness,
* feature importance support,
* and improved operational precision.

Random Forest produced strong precision-recall balance and significantly reduced false positive rates.

---

## XGBoost

XGBoost was evaluated because boosting-based models are highly effective for structured/tabular fraud datasets.

Advantages:

* strong ranking performance,
* effective imbalance handling,
* nonlinear modeling capability,
* and excellent PR-AUC performance.

XGBoost consistently produced the strongest overall fraud detection performance across multiple experiments.

---

# Imbalance Handling Strategies

Severe class imbalance was one of the most important challenges addressed in the project.

Several imbalance handling strategies were implemented and compared.

| Strategy        | Purpose                                       |
| --------------- | --------------------------------------------- |
| Undersampling   | Reduce majority class dominance               |
| Oversampling    | Duplicate minority fraud samples              |
| SMOTE           | Generate synthetic fraud samples              |
| Class Weighting | Penalize fraud misclassification more heavily |

---

## Undersampling

Undersampling significantly increased fraud recall by balancing the dataset.

However, it also introduced very large numbers of false positives because the model became overly sensitive to fraud patterns.

This created operationally unrealistic behavior.

---

## Oversampling

Oversampling duplicated minority fraud samples to improve fraud representation during training.

This improved recall while preserving majority class information, but introduced risk of overfitting because fraud examples were repeatedly duplicated.

---

## SMOTE

SMOTE generated synthetic fraud samples using interpolation between minority examples.

Advantages:

* improved minority representation,
* smoother decision boundaries,
* and better recall.

However, synthetic fraud generation may introduce unrealistic fraud patterns that do not perfectly represent real-world financial behavior.

---

## Class Weighting

Class weighting allowed imbalance handling directly within the optimization objective without modifying dataset distribution.

This is commonly used in real-world fraud detection systems because it preserves original data characteristics while emphasizing minority fraud classification.

---

# Evaluation Metrics

Traditional accuracy metrics were intentionally deprioritized because fraud detection is an extreme imbalance problem.

Instead, the project focused on metrics that better represent operational fraud detection quality.

| Metric    | Importance                  |
| --------- | --------------------------- |
| Precision | Fraud alert quality         |
| Recall    | Fraud detection sensitivity |
| F1 Score  | Precision-recall balance    |
| ROC-AUC   | Ranking capability          |
| PR-AUC    | Positive-class performance  |

---

## Why Accuracy Is Misleading

Because fraudulent transactions are extremely rare, a model can achieve very high accuracy while completely failing to detect fraud.

For example:

* predicting every transaction as legitimate would still achieve extremely high accuracy.

Therefore, fraud systems prioritize:

* recall,
* precision,
* PR-AUC,
* and operational error tradeoffs.

---

## PR-AUC Importance

PR-AUC was prioritized because it focuses specifically on positive-class performance under severe class imbalance.

This makes it significantly more informative than raw accuracy for fraud detection systems.

PR-AUC became one of the primary metrics used during final model comparison and selection.

---

# Threshold Analysis

Fraud detection systems rarely use default classification thresholds.

Threshold tuning was implemented to analyze operational tradeoffs between:

* fraud recall,
* false positives,
* customer friction,
* and investigation workload.

The system evaluated multiple thresholds and generated threshold-performance analysis tables and visualizations.

Lower thresholds generally:

* increased fraud recall,
* but dramatically increased false positives.

Higher thresholds:

* improved precision,
* but increased risk of missed fraud.

This demonstrated that fraud systems optimize business tradeoffs rather than maximizing generic ML metrics.

---

# Cross Validation

The project integrated stratified cross-validation to improve evaluation reliability.

`StratifiedKFold` was used to preserve fraud ratios across validation folds.

Cross-validation was important because:

* fraud datasets are highly imbalanced,
* and single train/test splits may produce unstable results.

Both mean and standard deviation metrics were logged to analyze model stability and robustness.

---

# Experiment Tracking with MLflow

MLflow was integrated to provide centralized experiment management and reproducibility.

Tracked information included:

* model parameters,
* sampling strategies,
* evaluation metrics,
* training duration,
* PR curves,
* ROC curves,
* feature importance plots,
* trained pipelines,
* and metadata artifacts.

This significantly improved:

* experiment reproducibility,
* model comparison,
* artifact management,
* and ML engineering workflow maturity.

---

# Metadata Management

The project implemented structured metadata persistence for each experiment.

Saved metadata included:

* model configuration,
* sampling strategy,
* feature names,
* threshold values,
* cross-validation statistics,
* evaluation metrics,
* and training timestamps.

This simulates primitive model governance workflows commonly required in financial ML systems.

---

# Model Registry

A lightweight model registry abstraction was implemented to centralize:

* pipeline saving,
* pipeline loading,
* and artifact management.

Persisting full pipelines rather than isolated models simplified deployment readiness because preprocessing and inference remain tightly coupled.

---

# Final Model Selection

Final model selection was based on:

* PR-AUC,
* precision-recall tradeoffs,
* operational false positive behavior,
* and cross-validation stability.

XGBoost provided the strongest overall PR-AUC performance and demonstrated highly effective fraud ranking capability.

Random Forest also produced strong operational performance with excellent precision characteristics and low false positive rates.

The final production model choice depends on operational priorities such as:

* fraud sensitivity,
* acceptable customer friction,
* and fraud investigation capacity.

---

# Key Engineering Learnings

This project provided significant practical exposure to real-world ML engineering concepts including:

* leakage prevention,
* pipeline-based preprocessing,
* experiment reproducibility,
* imbalance handling,
* threshold optimization,
* cross-validation,
* MLflow experiment tracking,
* model persistence,
* artifact management,
* and deployment-oriented ML architecture.

One of the most important learnings was that fraud detection systems optimize operational business tradeoffs rather than generic ML accuracy metrics.

---

# Future Work

## Stage 2 — Production Inference Platform

The next stage of the project will extend the system into a realtime fraud prediction service using:

* FastAPI,
* PostgreSQL,
* Docker Compose,
* inference APIs,
* model lifecycle management,
* and prediction logging infrastructure.

This stage will focus heavily on scalable backend engineering and production inference workflows.

---

## Stage 3 — Graph Neural Network Fraud Detection

The long-term roadmap includes extending the system into graph-based fraud detection using:

* PyTorch Geometric,
* Graph Neural Networks (GNNs),
* relational fraud modeling,
* and entity graph analysis.

Future graph structures may include:

* users,
* devices,
* merchants,
* accounts,
* and transaction relationships.

This stage will explore relational fraud reasoning beyond traditional tabular ML approaches.

---

# Conclusion

SentinelFlow evolved from a baseline fraud classification project into a production-style ML experimentation platform focused on financial fraud detection engineering.

The project emphasized:

* reproducibility,
* modular architecture,
* operational evaluation,
* and scalable ML system design.

The final system demonstrates practical ML engineering workflows that align closely with real-world fintech and fraud detection environments.
