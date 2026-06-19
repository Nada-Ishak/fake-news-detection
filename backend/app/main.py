from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import predictor
from app.schemas import NewsInput, PredictionOutput

app = FastAPI(
    title="Fake News Detection API",
    description="Predicts whether a news article is Fake or Real using a TF-IDF + ML pipeline.",
    version="1.0.0",
)

# Allow the static frontend (served from a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's real origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    predictor.load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: NewsInput):
    try:
        result = predictor.predict(payload.title, payload.text)
    except predictor.ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result
