# Fake News Detection

End‑to‑end fake news classifier using a fine‑tuned DistilBERT model (binary classification: Fake vs Real) trained on the WELFake dataset, served through a FastAPI backend, with a small static frontend ("Wire Desk") for submitting an article and seeing the verdict.

## Project structure

```
fake-news-detection/
├── ml/                         # Training pipeline (DistilBERT fine‑tuning)
│   ├── config.py               # Paths & hyper‑parameters
│   ├── preprocessing.py        # Helper to combine title + text
│   ├── train.py                # Load data → tokenize → fine‑tune → evaluate → save model
│   ├── data/                   # Place `WELFake_Dataset.csv` here (see data/README.md)
│   ├── models/                 # `distilbert_model/` (saved model + tokenizer)
│   └── requirements.txt        # Python dependencies
│
├── backend/                    # FastAPI inference API
│   └── app/
│       ├── main.py             # /predict, /health endpoints
│       ├── predictor.py        # Loads DistilBERT model & tokenizer, performs inference
│       └── schemas.py          # Request/response models
│
├── frontend/                   # Static HTML/CSS/JS client ("Wire Desk")
│   ├── index.html
│   ├── style.css
│   └── script.js
│
│
├── results/
│
└── README.md
```

## 1. Train the model

```bash
cd fake-news-detection
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt

# Download WELFake_Dataset.csv and place it at ml/data/WELFake_Dataset.csv
python -m ml.train
```

The script fine‑tunes DistilBERT, prints training metrics (accuracy, precision, recall, F1) and saves the artifacts to:

```
ml/models/distilbert_model/   # model files + tokenizer
```

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API reads the DistilBERT model from `ml/models/distilbert_model/`; the `predictor.py` module adds the project root to `sys.path` automatically.

### Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /predict` → body `{"title": "...", "text": "..."}` → returns:
  ```json
  {
    "label": "Fake|Real",
    "is_fake": true|false,
    "confidence": float,
    "fake_probability": float,
    "real_probability": float
  }
  ```

## 3. Run the frontend

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500. If the backend runs on a different host/port, update `API_BASE_URL` at the top of `frontend/script.js`.

## Notes

- `ml/preprocessing.py` now only provides `build_content` / `preprocess_for_inference` which concatenates the title and body; the DistilBERT tokenizer handles all token‑level preprocessing.
- Model files and the dataset are git‑ignored (see `.gitignore`) – they are regenerated locally, not committed.