"""
Core prediction logic shared by FastAPI and Streamlit.
Loads the trained TF-IDF vectorizer and model once at import time.
"""

import os
import re
import joblib

# ------------------------------------------------------------------
# Load model & vectorizer once (at module import time)
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

_VECTORIZER = None
_MODEL = None


def _load_artifacts():
    """Lazy-load vectorizer and model."""
    global _VECTORIZER, _MODEL
    if _VECTORIZER is None or _MODEL is None:
        _VECTORIZER = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        _MODEL = joblib.load(os.path.join(MODEL_DIR, "best_fake_news_model.pkl"))
    return _VECTORIZER, _MODEL


# ------------------------------------------------------------------
# Text preprocessing (same as training)
# ------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic text cleaner for news articles."""
    if text is None:
        return ""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove punctuation (keep spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------
# Prediction interface
# ------------------------------------------------------------------

def predict_article(title: str, text: str) -> dict:
    """
    Classify a news article as Fake or True.

    Returns a dict with keys:
        - prediction   : 0 (True) or 1 (Fake)
        - label        : "True News" or "Fake News"
        - confidence   : float between 0 and 1 (or None if model has no predict_proba)
        - is_fake      : bool
    """
    vectorizer, model = _load_artifacts()

    raw = str(title) + " " + str(text)
    cleaned = clean_text(raw)
    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = float(prob[pred])

    label = "Fake News" if pred == 1 else "True News"

    return {
        "prediction": pred,
        "label": label,
        "confidence": confidence,
        "is_fake": bool(pred == 1),
    }
