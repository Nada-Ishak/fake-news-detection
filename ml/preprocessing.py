"""
Text cleaning utilities shared between the training pipeline (ml/train.py)
and the backend prediction service (backend/app/predictor.py).

Keeping this logic in ONE place is critical: the model must see exactly
the same preprocessing at inference time as it saw during training,
otherwise the TF-IDF vocabulary and predictions will be wrong.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data() -> None:
    """Download the NLTK corpora needed for cleaning, if not already present."""
    for package, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Reproduce the exact cleaning steps used in the original notebook:
    lowercase -> strip non-letters -> tokenize -> remove stopwords -> lemmatize.
    """
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in _STOP_WORDS]
    words = [_LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def build_content(title: str, text: str) -> str:
    """Combine title + body the same way the training data was built."""
    title = title or ""
    text = text or ""
    return f"{title} {text}"


def preprocess_for_inference(title: str, text: str) -> str:
    """End-to-end helper: raw title/text -> cleaned string ready for the vectorizer."""
    return clean_text(build_content(title, text))
