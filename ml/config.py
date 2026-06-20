"""
Central configuration for the Fake News Detection project.
Configured for DistilBERT + WELFake.
"""

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"

DATASET_PATH = DATA_DIR / "WELFake_Dataset.csv"

MODEL_PATH = MODELS_DIR / "distilbert_model"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# WELFake labels in your dataset:
# 0 = Real
# 1 = Fake
LABEL_NAMES = ["Real", "Fake"]

for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
