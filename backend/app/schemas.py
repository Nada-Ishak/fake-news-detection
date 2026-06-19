from pydantic import BaseModel, Field


class NewsInput(BaseModel):
    title: str = Field(default="", description="News headline (optional)")
    text: str = Field(..., min_length=1, description="News article body to analyze")


class PredictionOutput(BaseModel):
    label: str            # "Real" or "Fake"
    is_fake: bool
    confidence: float      # 0..1, model's confidence in the predicted label
    fake_probability: float
    real_probability: float
