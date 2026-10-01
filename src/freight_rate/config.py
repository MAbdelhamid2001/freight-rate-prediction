"""Single source of truth for paths, column names and the validation design.

Everything that more than one module needs to agree on lives here, so the
training script, the prediction CLI, the API and the Streamlit app can never
drift apart on a column name or a file location.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/freight_rate/config.py -> repo root. When the package is installed
# non-editable (the Docker image), set FREIGHT_PROJECT_ROOT instead.
PROJECT_ROOT = Path(os.getenv("FREIGHT_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "freight_rate_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"

PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"

# Local SQLite store by default (it supports the model registry, a plain file
# store does not); point MLFLOW_TRACKING_URI at a server to share runs.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "freight-rate-prediction")
REGISTERED_MODEL_NAME = "freight-rate-model"

CACHE_DIR = PROJECT_ROOT / ".cache"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
ID_COL = "load_id"
TARGET = "posted_rate"
DATE_COL = "date"

EQUIPMENT_TYPES = ("Dry Van", "Reefer", "Flatbed")

RAW_FEATURES = (
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
)

# ---------------------------------------------------------------------------
# Data-quality rules (see eda.py / the report for where each number comes from)
# ---------------------------------------------------------------------------
# 47,500 lb is the most common maximum in the data and sits just under the
# usual 80,000 lb gross / ~45-48k lb legal payload for a 53' trailer.
MAX_WEIGHT_LB = 47_500.0

# A load whose posted rate is more than this factor away from an out-of-fold
# prediction is treated as a corrupted label for *training* only. EDA shows the
# corrupted rows sit at x2-x5 or x0.2-x0.4 of the expected rate, while 99% of
# clean rows are within +/-10%, so 1.6 separates the two populations cleanly.
LABEL_NOISE_RATIO = 1.6

# ---------------------------------------------------------------------------
# Validation design
# ---------------------------------------------------------------------------
# The labelled data covers Jan-Oct 2025 and the loads we must price are
# Nov-Dec 2025: a two-month forecast horizon that starts right after training
# ends. The folds reproduce that exactly: expanding window, 2-month test block.
CV_FOLDS = (
    # (train end exclusive, test start, test end exclusive)
    ("2025-07-01", "2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-09-01", "2025-11-01"),
)
# The last fold is also reported on its own as the "holdout", since it is the
# closest rehearsal of the real task (train Jan-Aug, predict the next 2 months).
HOLDOUT_FOLD = CV_FOLDS[-1]

# Model selection metric: mean MAPE over the time folds, on rows whose label is
# not corrupted. MAPE because a $40 miss means very different things on a $300
# and a $6,000 load; "clean" because the corrupted labels are unpredictable
# noise that would otherwise decide the ranking (see validation.evaluate).
SELECTION_METRIC = "cv_clean_mape"

RANDOM_STATE = 42
