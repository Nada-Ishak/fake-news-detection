"""
Central configuration for the Fake News Detection project.

All paths are resolved relative to this file so the project works
the same way no matter where it's cloned to on disk.
"""

from pathlib import Path

# ---- Project layout -------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent          # .../fake-news-detection/ml
PROJECT_ROOT = ML_DIR.parent                       # .../fake-news-detection

DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"

# ---- Dataset ---------------------------------------------------------------
# Put WELFake_Dataset.csv inside ml/data/ (see ml/data/README.md)
DATASET_PATH = DATA_DIR / "WELFake_Dataset.csv"

# ---- Saved artifacts --------------------------------------------------------
MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ---- Reports (plots) --------------------------------------------------------
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"

# ---- Training params --------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
RUN_CROSS_VALIDATION = True   # set False to skip the slow 5-fold CV step

# Labels used by the WELFake dataset: 0 = Fake, 1 = Real
LABEL_NAMES = ["Fake", "Real"]

for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
