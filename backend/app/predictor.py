"""
Loads the trained TF-IDF vectorizer + classifier once at startup and
exposes a single `predict()` function used by the FastAPI routes.

Reuses the exact same cleaning logic from ml/preprocessing.py that was
used during training, so inference matches training preprocessing 1:1.
"""

import sys
from pathlib import Path

import joblib

# Make the project root importable so we can reuse `ml.preprocessing`
# and `ml.config` instead of duplicating the cleaning logic here.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml import config  # noqa: E402
from ml.preprocessing import preprocess_for_inference  # noqa: E402

_model = None
_vectorizer = None


class ModelNotLoadedError(RuntimeError):
    pass


def load_artifacts() -> None:
    """Load the model + vectorizer into memory. Called once at app startup."""
    global _model, _vectorizer

    if not config.MODEL_PATH.exists() or not config.VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Expected:\n"
            f"  {config.MODEL_PATH}\n  {config.VECTORIZER_PATH}\n"
            f"Run training first: python -m ml.train"
        )

    _model = joblib.load(config.MODEL_PATH)
    _vectorizer = joblib.load(config.VECTORIZER_PATH)


def predict(title: str, text: str) -> dict:
    if _model is None or _vectorizer is None:
        raise ModelNotLoadedError("Model is not loaded yet. Call load_artifacts() first.")

    cleaned = preprocess_for_inference(title, text)
    features = _vectorizer.transform([cleaned])

    pred = int(_model.predict(features)[0])

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(features)[0]
        fake_probability, real_probability = float(proba[0]), float(proba[1])
    else:
        # Models without predict_proba (e.g. LinearSVC) -> use decision_function
        # squashed through a sigmoid as an approximate confidence score.
        import math
        score = float(_model.decision_function(features)[0])
        real_probability = 1 / (1 + math.exp(-score))
        fake_probability = 1 - real_probability

    label = "Real" if pred == 1 else "Fake"
    confidence = real_probability if pred == 1 else fake_probability

    return {
        "label": label,
        "is_fake": pred == 0,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
    }
