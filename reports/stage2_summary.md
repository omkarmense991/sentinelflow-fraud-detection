# SentinelFlow Fraud Detection — Stage 2 Summary

## Overview

Stage 2 focused on transforming the fraud detection machine learning pipeline from an offline experimentation platform into a production-style ML inference and serving system.

The goal of this stage was to operationalize the trained fraud detection model using scalable backend engineering principles and production deployment workflows.

This stage introduced:

* realtime inference APIs,
* model lifecycle management,
* PostgreSQL prediction persistence,
* Dockerized infrastructure,
* cloud deployment on AWS EC2,
* structured backend architecture,
* and production ML serving concepts.

The result is a deployable machine learning platform capable of serving fraud predictions in realtime while maintaining auditability and observability.

---

# Stage 2 Objectives

The primary objectives of Stage 2 were:

* Convert the trained fraud detection model into a production inference service.
* Build scalable REST APIs using FastAPI.
* Serve predictions using persisted sklearn pipelines.
* Introduce model loading and champion model concepts.
* Add PostgreSQL persistence for prediction logging.
* Containerize the application using Docker and Docker Compose.
* Deploy the platform on AWS EC2.
* Apply production backend engineering and MLOps fundamentals.

---

# Architectural Evolution

## Transition From Offline ML To Online Inference

Stage 1 primarily focused on:

* training workflows,
* experimentation,
* evaluation,
* model benchmarking,
* and artifact persistence.

Stage 2 shifted the focus toward:

* realtime prediction systems,
* deployment infrastructure,
* inference lifecycle management,
* and operational ML engineering.

This introduced a major architectural transition:

```text
Offline ML Experimentation
        ↓
Production ML Inference Platform
```

---

# Final Stage 2 System Architecture

```text
Client
   ↓
FastAPI REST API
   ↓
Inference Service
   ↓
Champion Pipeline
   ↓
Prediction Logging
   ↓
PostgreSQL
```

Infrastructure was orchestrated using Docker Compose and deployed to AWS EC2.

---

# FastAPI Inference Layer

## Realtime Prediction API

A production-style FastAPI application was implemented to expose fraud detection inference endpoints.

### Endpoints Implemented

| Endpoint                | Purpose                             |
| ----------------------- | ----------------------------------- |
| `/api/v1/predict`       | Single transaction fraud prediction |
| `/api/v1/predict/batch` | Batch fraud prediction              |
| `/api/v1/health`        | Health check endpoint               |
| `/`                     | Root API status endpoint            |

---

# API Versioning

API versioning was introduced using:

```text
/api/v1/
```

This improves:

* maintainability,
* backward compatibility,
* and long-term API evolution.

---

# Request & Response Validation

Pydantic schemas were used to validate:

* incoming transaction payloads,
* batch prediction requests,
* and structured prediction responses.

This prevents malformed inputs and ensures strong API contracts.

Implemented schemas included:

* `TransactionRequest`
* `PredictionResponse`
* `BatchTransactionRequest`
* `BatchPredictionResponse`

---

# Champion Model Architecture

Stage 2 introduced a champion model lifecycle.

Instead of serving arbitrary experiment artifacts, the system promotes a single approved model for deployment.

## Champion Artifacts

```text
artifacts/champion/
├── champion_model.pkl
└── champion_metadata.json
```

The champion model is selected automatically using evaluation criteria defined during Stage 1.

This introduced:

* primitive model registry concepts,
* deployment governance,
* and reproducible serving workflows.

---

# Model Loading System

The deployed pipeline is loaded once during application startup.

## Key Design Principle

```text
Load Once → Serve Many Requests
```

The model is NOT reloaded per request.

This significantly improves:

* latency,
* memory efficiency,
* and production scalability.

The inference layer dynamically loads:

* threshold,
* model version,
* sampling strategy,
* and metadata

from the deployed champion metadata file.

---

# Inference Service Layer

A dedicated service layer was introduced to separate:

* API routes,
* business logic,
* and persistence logic.

## Responsibilities

The inference service:

* converts request payloads into tabular DataFrames,
* performs pipeline inference,
* applies fraud thresholds,
* generates metadata-rich prediction responses,
* and persists predictions to PostgreSQL.

Prediction responses include:

* request IDs,
* timestamps,
* fraud probabilities,
* model versions,
* and sampling metadata.

---

# Batch Prediction Support

Enterprise-style batch inference support was added.

## Endpoint

```text
POST /api/v1/predict/batch
```

This endpoint processes multiple transactions in a single request.

Batch inference is highly relevant in:

* fraud analysis,
* banking systems,
* overnight scoring jobs,
* and enterprise ML systems.

---

# PostgreSQL Prediction Persistence

Stage 2 introduced persistent prediction logging using PostgreSQL.

## Why Persistence Matters

Real fraud systems require:

* auditability,
* compliance,
* prediction history,
* monitoring,
* and forensic investigation.

Prediction logs are therefore persisted after every inference request.

---

# Database Architecture

## SQLAlchemy ORM

SQLAlchemy was used as the ORM layer.

The persistence layer includes:

```text
src/database/
├── connection.py
├── models.py
├── init_db.py
└── repositories/
```

---

# Prediction Log Table

A `prediction_logs` table was implemented containing:

* request IDs,
* prediction timestamps,
* fraud probabilities,
* threshold values,
* model versions,
* model names,
* and sampling strategies.

This creates a persistent audit trail for inference activity.

---

# Repository Pattern

Database operations were abstracted into repository modules.

This separates:

* business logic,
* persistence logic,
* and database implementation details.

This is a common production backend engineering pattern.

---

# Structured Logging

Structured application logging was introduced using Python logging.

This replaced raw print statements and improved:

* observability,
* debugging,
* and production monitoring.

Request logs and inference activity are now visible through container logs.

---

# Middleware Integration

FastAPI middleware was added to measure request processing time.

This introduced:

* latency tracking,
* response timing,
* and API observability concepts.

Custom response headers now expose processing time metrics.

---

# Exception Handling

Production-style exception handling was introduced for inference APIs.

This prevents:

* unhandled crashes,
* malformed responses,
* and unstable inference behavior.

The API now returns structured HTTP error responses.

---

# Dockerization

The entire ML inference platform was containerized.

## Docker Components

### FastAPI Container

Contains:

* inference APIs,
* model loading,
* business logic,
* and service orchestration.

### PostgreSQL Container

Contains:

* persistent prediction storage,
* audit trail data,
* and database infrastructure.

---

# Docker Compose Infrastructure

Docker Compose was used to orchestrate:

* FastAPI,
* PostgreSQL,
* networking,
* environment variables,
* and persistent volumes.

This introduced:

* infrastructure-as-code,
* multi-container orchestration,
* and reproducible deployments.

---

# Environment-Based Configuration

Stage 2 introduced centralized environment-driven configuration.

A `.env` file was used to manage:

* database credentials,
* thresholds,
* deployment configuration,
* and infrastructure settings.

Configuration values are loaded through:

```text
src/config/settings.py
```

This improves:

* deployment portability,
* security,
* and maintainability.

---

# AWS EC2 Deployment

The complete ML platform was deployed to AWS EC2.

## Deployment Stack

```text
AWS EC2
   ↓
Docker Compose
   ↓
FastAPI Container
   ↓
PostgreSQL Container
```

---

# EC2 Deployment Workflow

Deployment included:

* launching an Ubuntu EC2 instance,
* configuring security groups,
* installing Docker and Docker Compose,
* cloning the project repository,
* configuring environment variables,
* and launching containers.

---

# Public API Serving

After deployment, the system successfully served:

* realtime fraud predictions,
* batch predictions,
* Swagger/OpenAPI documentation,
* and health endpoints

through the public EC2 instance.

Swagger UI was accessible externally using:

```text
http://<EC2-IP>:8000/docs
```

---

# Production Engineering Concepts Learned

Stage 2 introduced several critical production ML engineering concepts.

| Concept             | Description                 |
| ------------------- | --------------------------- |
| Inference Lifecycle | Model serving architecture  |
| Champion Models     | Deployment governance       |
| API Engineering     | FastAPI service design      |
| Batch Inference     | Enterprise ML workflows     |
| Persistence         | Prediction audit trails     |
| PostgreSQL          | Production storage          |
| Docker              | Containerized deployment    |
| Docker Compose      | Multi-service orchestration |
| AWS EC2             | Cloud infrastructure        |
| Structured Logging  | Observability               |
| Middleware          | API instrumentation         |
| Environment Config  | Deployment portability      |
| Lifespan Management | Startup/shutdown lifecycle  |
| Repository Pattern  | Clean backend architecture  |

---

# Final Stage 2 Outcome

At the conclusion of Stage 2, the project evolved into a deployable production-style ML platform capable of:

* training fraud detection models,
* serving realtime predictions,
* logging inference activity,
* operating in containerized infrastructure,
* and running on cloud infrastructure.

The platform now demonstrates:

* ML engineering,
* backend engineering,
* MLOps fundamentals,
* deployment architecture,
* and production systems thinking.

---

# Current Project Status

The project now includes:

| Capability                  | Status |
| --------------------------- | ------ |
| Fraud Detection ML Pipeline | ✅      |
| MLflow Experiment Tracking  | ✅      |
| Cross Validation            | ✅      |
| Threshold Engineering       | ✅      |
| Champion Model Selection    | ✅      |
| Realtime Inference API      | ✅      |
| Batch Prediction API        | ✅      |
| PostgreSQL Logging          | ✅      |
| Dockerized Infrastructure   | ✅      |
| Structured Logging          | ✅      |
| AWS EC2 Deployment          | ✅      |
| Production ML Architecture  | ✅      |

---

# Transition To Stage 3

With the completion of Stage 2, the project now transitions from:

```text
Tabular Fraud Detection
        ↓
Graph-Based Relational Fraud Detection
```

Stage 3 will focus on:

* Graph Neural Networks (GNNs),
* relational fraud detection,
* entity relationship modeling,
* transaction graph construction,
* and graph-based fraud reasoning using PyTorch Geometric.

This stage will significantly extend the project beyond traditional tabular ML systems and into modern graph-based fraud intelligence systems.
