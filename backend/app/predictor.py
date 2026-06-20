from pathlib import Path
from typing import Dict

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Root project path
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "ml" / "models" / "distilbert_model"

# Globals
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():
    global tokenizer, model

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"DistilBERT model folder not found:\n{MODEL_DIR}\n"
            f"Make sure you trained/saved the model first."
        )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()


def predict(title: str, text: str) -> Dict:
    if model is None or tokenizer is None:
        raise RuntimeError("Model artifacts are not loaded. Call load_artifacts() first.")

    combined_text = f"{title} {text}".strip()

    inputs = tokenizer(
        combined_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # WELFake labels in your project:
    # 0 = Real
    # 1 = Fake
    real_prob = float(probs[0].item())
    fake_prob = float(probs[1].item())

    pred = int(torch.argmax(probs).item())

    label = "Fake" if pred == 1 else "Real"
    confidence = fake_prob if pred == 1 else real_prob

    return {
        "label": label,
        "is_fake": pred == 1,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
    }