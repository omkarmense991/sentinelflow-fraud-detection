from pathlib import Path

# =========================================
# Project
# =========================================

PROJECT_NAME = "SentinelFlow Fraud Detection API"

EXPERIMENT_NAME = "fraud_detection_experiments"


# =========================================
# Root Directories
# =========================================

ARTIFACTS_DIR = Path("artifacts")


# =========================================
# Artifact Subdirectories
# =========================================

MODELS_DIR = ARTIFACTS_DIR / "models"

PLOTS_DIR = ARTIFACTS_DIR / "plots"

METADATA_DIR = ARTIFACTS_DIR / "metadata"

THRESHOLD_DIR = ARTIFACTS_DIR / "thresholds"

METRICS_DIR = ARTIFACTS_DIR / "metrics"

CHAMPION_DIR = ARTIFACTS_DIR / "champion"


# =========================================
# Champion Model Paths
# =========================================

CHAMPION_MODEL_PATH = CHAMPION_DIR / "champion_model.pkl"

CHAMPION_METADATA_PATH = CHAMPION_DIR / "champion_metadata.json"


# =========================================
# ML Settings
# =========================================

DEFAULT_THRESHOLD = 0.3

THRESHOLD_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# =========================================
# Auto Create Directories
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
