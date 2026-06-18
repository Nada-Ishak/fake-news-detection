"""
FastAPI backend for Fake News Detection.

Run locally:
    uvicorn api:app --reload --port 8000

Run in production:
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_article

app = FastAPI(
    title="Fake News Detection API",
    description="Classify news articles as Fake or True using a trained TF-IDF + Linear SVM model.",
    version="1.0.0",
)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    title: str
    text: str


class PredictResponse(BaseModel):
    prediction: int          # 0 = True, 1 = Fake
    label: str               # "True News" or "Fake News"
    confidence: float | None
    is_fake: bool


class HealthResponse(BaseModel):
    status: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
def health_check():
    """Root health-check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict whether a news article is Fake or True.

    - **title**: Article headline
    - **text**:  Article body text

    Returns prediction label, confidence score, and boolean flag.
    """
    result = predict_article(req.title, req.text)
    return PredictResponse(**result)
