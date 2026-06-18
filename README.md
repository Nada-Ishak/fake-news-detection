# Fake News Detection

A lightweight, production-ready fake news classifier built with **scikit-learn** and deployable via **FastAPI** and **Streamlit**.

---

## 🚀 What's Inside

| File | Description |
|------|-------------|
| `api.py` | FastAPI backend — REST API for predictions |
| `app.py` | Streamlit frontend — interactive web UI |
| `predict.py` | Core prediction engine (model loading + text cleaning + inference) |
| `models/` | Trained model artifacts (`.pkl` files) |
| `training/` | Training scripts, notebooks, and guides |
| `data/` | Raw datasets (`fake.csv`, `true.csv`) — **not committed to Git** |

---

## 🏗️ Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Streamlit   │─────▶│   FastAPI    │─────▶│  scikit-learn │
│   (app.py)   │      │  (api.py)    │      │ (predict.py) │
└──────────────┘      └──────────────┘      └──────────────┘
```

- **FastAPI** serves a `/predict` endpoint that accepts JSON `{title, text}` and returns `{prediction, label, confidence, is_fake}`.
- **Streamlit** provides a user-friendly web form to paste articles and see results instantly.
- Both backends share the same `predict.py` core, so predictions are identical.

---

## ⚡ Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI backend

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/docs` to test the API in your browser.

### 3. Run the Streamlit frontend (in a new terminal)

```bash
streamlit run app.py
```

Open `http://localhost:8501` to use the web UI.

---

## 🌐 Deploy to Production

### FastAPI (Render / Railway / VPS)

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or with Gunicorn for production:

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

### Streamlit (Streamlit Community Cloud)

1. Push this repo to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch, and set the main file to `app.py`
4. Deploy — done!

> **Note:** Streamlit Cloud reads `requirements.txt` automatically. The `.pkl` model files (~2.5 MB) are small enough to commit to Git. If you prefer Git LFS, configure it for `models/*.pkl`.

---

## 📊 API Usage Example

### Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Federal Reserve Raises Interest Rates",
    "text": "The Federal Reserve announced Wednesday that it will raise the benchmark interest rate by 0.25 percentage points, citing continued economic growth."
  }'
```

### Response

```json
{
  "prediction": 0,
  "label": "True News",
  "confidence": 0.9823,
  "is_fake": false
}
```

---

## 🧠 Model Details

- **Algorithm:** TF-IDF + Linear SVM (with Naive Bayes and Logistic Regression baselines)
- **Features:** 50,000 TF-IDF unigrams + bigrams
- **Dataset:** ~45,000 labeled news articles (50% fake / 50% true)
- **Expected Accuracy:** ~93–97% on held-out test set
- **Training time:** 2–5 minutes on CPU
- **Inference time:** < 10 ms per article

See `training/TRAINING_GUIDE.md` for the full training pipeline.

---

## 📁 Project Structure

```
.
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit frontend
├── predict.py              # Core prediction module
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores data/ and cache files
├── models/
│   ├── best_fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/                   # Raw CSVs (ignored by Git)
│   ├── fake.csv
│   └── true.csv
└── training/
    ├── fake_news_train.py   # Full training pipeline
    ├── TRAINING_GUIDE.md    # Phase-by-phase guide
    └── task.ipynb           # Original notebook
```

---

## ⚠️ Notes

- The `data/` folder is listed in `.gitignore` because the CSV files are >100 MB combined and exceed GitHub's file size limits. Keep them locally or use Git LFS if you need to version them.
- The model `.pkl` files are ~2.5 MB total and are safe to commit to Git for deployment purposes.

---

## 📜 License

MIT
