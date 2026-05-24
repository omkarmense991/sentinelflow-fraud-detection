from pathlib import Path

from dotenv import load_dotenv

import os

load_dotenv()


# =========================================
# Project
# =========================================

PROJECT_NAME = "SentinelFlow Fraud Detection API"

EXPERIMENT_NAME = "fraud_detection_experiments"


# =========================================
# Database
# =========================================

POSTGRES_USER = os.getenv("POSTGRES_USER")

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

POSTGRES_DB = os.getenv("POSTGRES_DB")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")

POSTGRES_PORT = os.getenv("POSTGRES_PORT")


DATABASE_URL = (
    f"postgresql://"
    f"{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


# =========================================
# ML Settings
# =========================================

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.3))


# =========================================
# Directories
# =========================================

ARTIFACTS_DIR = Path("artifacts")

MODELS_DIR = ARTIFACTS_DIR / "models"

PLOTS_DIR = ARTIFACTS_DIR / "plots"

METADATA_DIR = ARTIFACTS_DIR / "metadata"

THRESHOLD_DIR = ARTIFACTS_DIR / "thresholds"

METRICS_DIR = ARTIFACTS_DIR / "metrics"

CHAMPION_DIR = ARTIFACTS_DIR / "champion"


# =========================================
# Champion
# =========================================

CHAMPION_MODEL_PATH = CHAMPION_DIR / "champion_model.pkl"

CHAMPION_METADATA_PATH = CHAMPION_DIR / "champion_metadata.json"


# =========================================
# Auto Create
# =========================================

DIRECTORIES = [
    ARTIFACTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    METADATA_DIR,
    THRESHOLD_DIR,
    METRICS_DIR,
    CHAMPION_DIR,
]


for directory in DIRECTORIES:

    directory.mkdir(parents=True, exist_ok=True)
