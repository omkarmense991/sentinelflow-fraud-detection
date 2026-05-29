# SentinelFlow

A production-style fraud detection platform that combines traditional machine learning, graph neural networks, experiment tracking, champion/challenger model management, and realtime inference services.

The project was built to simulate how fraud detection systems are designed, trained, evaluated, deployed, and governed in real-world financial environments.

---

# Features

## Traditional Machine Learning

* Logistic Regression
* Random Forest
* XGBoost
* Class imbalance handling
* Cross-validation
* Threshold optimization
* Feature importance analysis

## Graph Neural Networks

* Graph Convolutional Networks (GCN)
* GraphSAGE
* Transaction graph processing
* Relational fraud detection
* Graph-based experimentation

## Model Governance

* Champion / Challenger architecture
* Automated model selection
* Dataset-level champion election
* Cross-family comparison (Tabular vs GNN)

## Experiment Tracking

* MLflow integration
* Metrics tracking
* Artifact versioning
* Model metadata management

## Production Services

* FastAPI inference service
* Batch prediction endpoint
* Prediction logging
* PostgreSQL integration
* Docker support

---

# Architecture

```text
                     ┌─────────────────┐
                     │    Datasets     │
                     └────────┬────────┘
                              │
                ┌─────────────▼─────────────┐
                │ Data Processing Layer     │
                └─────────────┬─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼

 ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
 │ LogisticReg │      │  XGBoost    │      │ RandomForest│
 └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
        │                    │                    │
        └────────────┬───────┴────────────┬───────┘
                     │
                     ▼

            Tabular Champion

                     │

        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼

   ┌───────────┐           ┌───────────┐
   │    GCN    │           │ GraphSAGE │
   └─────┬─────┘           └─────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼

              GNN Champion

                     │
                     ▼

         Overall Dataset Champion

                     │
                     ▼

            FastAPI Inference
```

---

# Datasets

## Credit Card Fraud Detection Dataset

Used for tabular fraud detection experiments.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Characteristics:

* Highly imbalanced dataset
* PCA-transformed features
* Binary fraud classification

---

## Elliptic Bitcoin Transaction Dataset

Used for graph-based fraud detection.

Dataset:
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Characteristics:

* Bitcoin transaction graph
* ~46,000 nodes
* ~36,000 edges
* 166 node features
* Illicit transaction detection

---

# Models

## Tabular Models

### Logistic Regression

Baseline linear model used for comparison.

### Random Forest

Tree-based ensemble model.

### XGBoost

Gradient boosted decision trees.

Best performing tabular model.

---

## Graph Models

### Graph Convolutional Network (GCN)

Learns node representations through graph convolution operations.

### GraphSAGE

Neighborhood aggregation based graph neural network.

Best performing graph model.

---

# Evaluation Metrics

The following metrics are tracked:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

Additional analysis:

* Threshold tuning
* Confusion matrix
* Precision-Recall curves
* ROC curves

---

# Champion Selection Strategy

## Tabular Champion

The best tabular model is selected based on:

1. Validation metrics
2. Minimum precision requirements
3. Test PR-AUC

---

## GNN Champion

The best graph model is selected based on:

1. Validation F1 score
2. Test PR-AUC

---

## Overall Champion

The system compares:

* Tabular Champion
* GNN Champion

The model with the highest PR-AUC becomes the overall dataset champion.

---

# Experiment Tracking

MLflow is used for:

* Parameter tracking
* Metric tracking
* Artifact storage
* Model comparison
* Reproducibility

Tracked artifacts:

* Model checkpoints
* ROC curves
* Precision-Recall curves
* Threshold analysis
* Metadata files

---

# API

Stage 2 introduced a FastAPI-based inference service.

Endpoints:

```http
GET /api/v1/health
```

```http
POST /api/v1/predict
```

```http
POST /api/v1/predict/batch
```

Features:

* Realtime predictions
* Batch predictions
* Prediction logging
* Model metadata tracking

---

# Technologies

* Python
* Scikit-learn
* XGBoost
* PyTorch
* PyTorch Geometric
* FastAPI
* PostgreSQL
* MLflow
* Docker
* Pandas
* NumPy

---

# Project Stages

## Stage 1 — Traditional Machine Learning

Implemented:

* Data ingestion
* Feature engineering
* Sampling strategies
* Model training
* Cross-validation
* Experiment tracking

Outcome:

* Multiple fraud detection models
* Automated evaluation pipeline

---

## Stage 2 — Production Inference Platform

Implemented:

* FastAPI service
* Model registry
* Champion promotion
* Prediction logging
* PostgreSQL integration
* Docker support

Outcome:

* Production-style serving layer

---

## Stage 3 — Graph Neural Networks

Implemented:

* Elliptic graph dataset support
* PyTorch Geometric integration
* GCN
* GraphSAGE
* Graph model registry
* GNN champion selection
* Cross-family model comparison

Outcome:

* GraphSAGE became the strongest GNN architecture
* XGBoost remained overall dataset champion

---

# Results

Best Tabular Model

* XGBoost
* PR-AUC ≈ 0.986

Best Graph Model

* GraphSAGE
* PR-AUC ≈ 0.929

Overall Champion

* XGBoost

---

# Future Work

Potential enhancements:

* Graph Attention Networks (GAT)
* Temporal Graph Networks
* Ensemble models
* Explainability dashboards
* Online graph inference
* Drift monitoring
* Automated retraining pipelines

---

# Author

Omkar Mense

Machine Learning Engineering Project
SentinelFlow Fraud Detection Platform
