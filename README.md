# Fake News Detection

End-to-end fake news classifier: TF-IDF + classical ML models trained on the
WELFake dataset, served through a FastAPI backend, with a small static
frontend ("Wire Desk") for submitting an article and seeing the verdict.

## Project structure

```
fake-news-detection/
├── ml/                        # training pipeline (was the original notebook)
│   ├── config.py              # paths & hyperparameters, single source of truth
│   ├── preprocessing.py       # text cleaning — shared with the backend
│   ├── train.py                # load -> clean -> vectorize -> train -> evaluate -> save
│   ├── evaluate.py             # confusion matrix / ROC plotting helpers
│   ├── data/                   # put WELFake_Dataset.csv here (see data/README.md)
│   ├── models/                 # fake_news_model.pkl + tfidf_vectorizer.pkl land here
│   ├── reports/                # confusion_matrix.png + roc_curve.png land here
│   ├── notebooks/               # original exploration notebook, kept for reference
│   └── requirements.txt
│
├── backend/                    # FastAPI inference API
│   ├── app/
│   │   ├── main.py             # /predict, /health
│   │   ├── predictor.py        # loads the saved model + reuses ml.preprocessing
│   │   └── schemas.py          # request/response models
│   └── requirements.txt
│
└── frontend/                   # static HTML/CSS/JS client ("Wire Desk")
    ├── index.html
    ├── style.css
    └── script.js
```

## 1. Train the model

```bash
cd fake-news-detection
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt

# download WELFake_Dataset.csv and place it at ml/data/WELFake_Dataset.csv
python -m ml.train
```

This prints a model comparison table, saves `confusion_matrix.png` /
`roc_curve.png` into `ml/reports/`, and writes the trained artifacts to:

```
ml/models/fake_news_model.pkl
ml/models/tfidf_vectorizer.pkl
```

> Cross-validation is on by default and can take a few minutes on the full
> 70k-row dataset. Set `RUN_CROSS_VALIDATION = False` in `ml/config.py` to
> skip it while iterating.

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API reads the model from `ml/models/`, so it must be run with the
project root on the path — running it from `backend/` as shown above works
because `predictor.py` adds the project root to `sys.path` automatically.

Endpoints:
- `GET /health` → `{"status": "ok"}`
- `POST /predict` → body `{"title": "...", "text": "..."}` → returns label,
  confidence, and fake/real probabilities.

Interactive docs: http://localhost:8000/docs

## 3. Run the frontend

The frontend is plain HTML/CSS/JS — no build step. Serve it with any static
server (opening `index.html` directly also works, but a server avoids CORS
quirks with some browsers):

```bash
cd frontend
python -m http.server 5500
```

Then open http://localhost:5500. If your backend isn't on
`http://localhost:8000`, update `API_BASE_URL` at the top of
`frontend/script.js`.

## Notes

- `ml/preprocessing.py` is imported by **both** training and the backend, so
  inference always sees the exact same cleaning as training did. Don't
  duplicate this logic anywhere else.
- Model files and the dataset are git-ignored (see `.gitignore`) — they're
  regenerated locally, not committed.
