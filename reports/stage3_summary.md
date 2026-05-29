# Stage 3 Summary – Graph-Based Fraud Detection

## Overview

Stage 3 extended SentinelFlow beyond traditional tabular machine learning by introducing Graph Neural Networks (GNNs) for fraud detection on transaction networks.

While Stage 1 focused on data engineering and experimentation infrastructure, and Stage 2 focused on production-ready tabular machine learning models, Stage 3 explored graph-based learning techniques capable of leveraging relationships between transactions.

The Elliptic Bitcoin Transaction Dataset was introduced as a graph-structured fraud detection benchmark where transactions are represented as nodes and transaction flows are represented as edges.

---

## Objectives

The primary objectives of Stage 3 were:

* Integrate graph-based datasets into the training framework.
* Build an end-to-end GNN training pipeline.
* Implement Graph Convolutional Networks (GCN).
* Implement GraphSAGE.
* Compare multiple graph architectures.
* Extend the champion/challenger framework to support GNN models.
* Enable cross-family model selection between tabular and graph models.
* Preserve experiment tracking, artifact management, and reproducibility.

---

## Dataset

### Elliptic Bitcoin Transaction Dataset

The Elliptic dataset represents Bitcoin transactions as a graph.

#### Graph Structure

* Nodes represent Bitcoin transactions.
* Edges represent transaction flows between transactions.
* Node features contain transaction-level attributes.
* Labels indicate:

  * Licit transactions
  * Illicit transactions
  * Unknown transactions

#### Dataset Statistics

* ~46,000 transaction nodes
* ~36,000 graph edges
* 166 node features per transaction
* Binary fraud classification task

---

## Graph Data Pipeline

A graph dataset construction pipeline was developed to transform the raw Elliptic dataset into a PyTorch Geometric Data object.

The pipeline performs:

* Feature loading
* Label processing
* Edge construction
* Train/Validation/Test mask generation
* Graph serialization into PyTorch Geometric format

Output:

```python
Data(
    x=node_features,
    edge_index=graph_edges,
    y=labels,
    train_mask=...,
    val_mask=...,
    test_mask=...
)
```

---

## GCN Implementation

A Graph Convolutional Network (GCN) architecture was implemented using PyTorch Geometric.

### Architecture

Layer 1:

* GCNConv
* ReLU
* Dropout

Layer 2:

* GCNConv

### Configuration

* Hidden Dimension: 64
* Learning Rate: 0.01
* Weight Decay: 5e-4
* Dropout: 0.3
* Epochs: 200

---

## GraphSAGE Implementation

A GraphSAGE architecture was implemented as a second graph learning approach.

### Architecture

Layer 1:

* SAGEConv
* ReLU
* Dropout

Layer 2:

* SAGEConv

### Benefits

Compared with GCN, GraphSAGE performs neighborhood aggregation using learnable functions and is generally more scalable for large graphs.

---

## Training Pipeline

A reusable GNN training framework was developed.

The framework supports:

* Multiple graph architectures
* Automatic model registration
* Validation-based checkpointing
* Experiment logging
* Artifact persistence
* Threshold analysis
* Champion promotion

Each architecture is trained independently using the same configuration and evaluation pipeline.

---

## Validation Strategy

Models are selected using Validation F1 Score.

During training:

1. Model trains on train nodes.
2. Validation probabilities are generated.
3. Validation F1 is computed.
4. Best checkpoint is saved.

This prevents selecting the final epoch if validation performance degrades.

---

## Evaluation Metrics

The following metrics are computed:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

Threshold-independent metrics:

* ROC-AUC
* PR-AUC

Threshold-dependent metrics:

* Precision
* Recall
* F1

A unified fraud threshold was used across both tabular and graph models.

---

## Threshold Analysis

Threshold analysis was introduced to evaluate performance under different fraud decision thresholds.

Evaluated thresholds:

```text
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

For each threshold:

* Precision
* Recall
* F1 Score
* False Positives
* False Negatives

are recorded and visualized.

---

## Experiment Tracking

MLflow integration was extended to support graph models.

Tracked artifacts include:

* Model parameters
* Metrics
* Threshold analysis
* ROC curves
* Precision-Recall curves
* Model checkpoints

This enables reproducible experimentation across graph architectures.

---

## Champion/Challenger Framework

Stage 3 introduced a separate GNN champion lane.

### GNN Champion Selection

Among all graph architectures:

```text
GCN
GraphSAGE
```

the model with the highest PR-AUC becomes the GNN Champion.

### Overall Champion Selection

The system then compares:

* Tabular Champion
* GNN Champion

The model with the highest PR-AUC becomes the overall dataset champion.

This creates a unified model selection process across fundamentally different model families.

---

## Experimental Results

### GCN

Performance achieved:

* Precision ≈ 0.82
* Recall ≈ 0.82
* F1 ≈ 0.82
* ROC-AUC ≈ 0.97
* PR-AUC ≈ 0.86

### GraphSAGE

Performance achieved:

* Precision ≈ 0.87
* Recall ≈ 0.85
* F1 ≈ 0.86
* ROC-AUC ≈ 0.98
* PR-AUC ≈ 0.93

GraphSAGE consistently outperformed GCN across most evaluation metrics.

### Comparison Against Tabular Models

Best Tabular Model:

* XGBoost
* PR-AUC ≈ 0.986

Best GNN Model:

* GraphSAGE
* PR-AUC ≈ 0.929

Result:

The tabular XGBoost model remained the overall dataset champion.

---

## Key Achievements

Stage 3 successfully introduced:

* Graph Neural Networks
* PyTorch Geometric integration
* Graph dataset processing
* GCN implementation
* GraphSAGE implementation
* GNN experiment tracking
* GNN model registry
* GNN champion selection
* Cross-family model comparison
* Unified threshold management

---

## Future Work

Potential future enhancements include:

* GAT (Graph Attention Networks)
* Heterogeneous Graph Neural Networks
* Temporal Graph Networks
* Inductive GraphSAGE inference
* Graph-based online serving
* Large-scale distributed graph training
* Ensemble learning between tabular and graph models

---

## Conclusion

Stage 3 transformed SentinelFlow from a traditional fraud detection platform into a hybrid machine learning framework capable of training and evaluating both tabular and graph-based fraud detection models.

The addition of Graph Neural Networks significantly expanded the platform's capabilities while preserving reproducibility, experiment tracking, and champion/challenger model governance.
