# Fake News Detection — Training Guide (scikit-learn)

> **Goal:** Build a high-accuracy fake-news classifier using **only scikit-learn**, running in **Google Colab**.
> **Dataset:** ~45,000 news articles (`fake.csv` + `true.csv`)
> **Approach:** TF-IDF + classical linear models (Naive Bayes, Logistic Regression, Linear SVM)

---

## How to Run in Google Colab

### Step 1: Upload Data
In Colab, click the **folder icon** on the left sidebar, then upload:
- `fake.csv`
- `true.csv`

Or mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Then adjust paths to your Drive folder
```

### Step 2: Paste the Code
Open `fake_news_train.py` and **copy all the code** into a single Colab cell (or multiple cells if you prefer), then run.

**Runtime:** CPU is enough. No GPU required.
**Estimated time:** 2–5 minutes total.

---

## Phase-by-Phase Breakdown

### Phase 1 — Imports & Setup
We import `pandas`, `numpy`, `sklearn` modules, and `joblib` for saving models. No external libraries needed beyond what Colab already has installed.

### Phase 2 — Data Loading & Cleaning
- Load both CSVs
- Label: `1 = Fake`, `0 = True`
- Merge into one dataframe
- Remove rows with empty `title` or `text`
- Remove duplicate articles (by `text` and by `title`)

**Why clean?** Duplicates and empty rows inflate accuracy artificially and waste training time.

### Phase 3 — Text Preprocessing
- Combine `title` + `text` into one field
- Convert to lowercase
- Remove URLs, emails, punctuation
- Collapse extra whitespace

**Why combine title + text?** The title alone often contains clickbait signals. The body gives factual context. Together they are the strongest signal.

**Why not stem/lemmatize?** For TF-IDF + linear models, simple cleaning is usually enough. Stemming can sometimes hurt performance by collapsing distinct words.

### Phase 4 — Train / Validation / Test Split
| Set | Purpose | Size |
|-----|---------|------|
| Train | Fit model & learn weights | 70% |
| Validation | Compare models & tune | 15% |
| Test | **Final evaluation only** | 15% |

- `stratify=label` keeps the 50/50 fake/true ratio in every split.
- `random_state=42` makes results reproducible.

**Why 3 sets?** If you pick the best model using the Test set, you are "cheating" and overestimating real-world accuracy. The Validation set is your model-selection arena.

### Phase 5 — TF-IDF Feature Extraction
We convert text into numerical vectors using **TF-IDF**:

```python
TfidfVectorizer(
    max_features=50000,   # top 50K words/phrases
    ngram_range=(1, 2),   # single words + two-word phrases
    stop_words='english', # remove "the", "and", "is"...
    min_df=2,             # ignore extremely rare words
    max_df=0.9            # ignore extremely common words
)
```

**Why TF-IDF?**
- Fast on CPU (no neural networks needed).
- Highly interpretable — you can see exactly which words drive the prediction.
- Proven strong baseline for news classification.

**Why bigrams (`ngram_range=(1,2)`)?** Phrases like "fake news" or "breaking report" carry stronger signal than the individual words alone.

**Important:** We `fit()` on Train only, then `transform()` on Val and Test. This prevents "data leakage" (the model learning from test vocabulary).

### Phase 6 — Model Training
We train 3 classical models in parallel:

| Model | Why It Fits This Project |
|-------|--------------------------|
| **Naive Bayes** | Trains instantly. Great baseline for TF-IDF word features. |
| **Logistic Regression** | Strong, fast, gives probability scores. Best interpretability. |
| **Linear SVM** | Often the best classical model for text. Very robust to high-dimensional data. |

**Why not Random Forest / XGBoost?** Tree models usually underperform on sparse TF-IDF data compared to linear models. They are better for dense, tabular features.

**Why not Deep Learning?** You asked for scikit-learn simplicity. For this dataset size (~45k), a well-tuned Linear SVM or Logistic Regression often reaches **95%+ accuracy**, which is competitive with small LSTMs and even close to BERT in some cases.

### Phase 7 — Evaluation
Metrics computed on the **Validation set**:
- **Accuracy** — overall correct percentage
- **Precision** — of articles we called "fake", how many were really fake?
- **Recall** — of all real fake articles, how many did we catch?
- **F1 Score** — balance of precision and recall. **Use this to pick the winner.**

**Why F1?** In fake-news detection, both false positives (calling real news fake) and false negatives (missing real fake news) are bad. F1 balances both.

### Phase 8 — Final Test-Set Evaluation
The winning model (highest Validation F1) is evaluated **once** on the held-out Test set. This number is your honest estimate of real-world accuracy.

After that, we save:
- `best_fake_news_model.pkl` — the trained model
- `tfidf_vectorizer.pkl` — the vocabulary and IDF weights

### Phase 9 — Predict on New Articles
A `predict_article(title, text)` function is included. You can paste any headline + body and get an instant prediction.

---

## Expected Results

Based on similar datasets with this exact pipeline, you should expect:

| Model | Expected Validation F1 | Expected Test Accuracy |
|-------|------------------------|------------------------|
| Naive Bayes | ~0.88 – 0.92 | ~88 – 92% |
| Logistic Regression | ~0.93 – 0.96 | ~93 – 96% |
| Linear SVM | ~0.93 – 0.97 | ~93 – 97% |

> **Note:** Your exact numbers depend on the specific dataset version and random split, but they should fall in this range.

---

## What If You Want Even Better Accuracy Later?

If you finish this pipeline and want to push accuracy further, here are the **next steps** (still in scikit-learn land):

1. **Hyperparameter tuning** — Use `GridSearchCV` or `RandomizedSearchCV` to find the best `C` for Logistic Regression / SVM, and the best `max_features` for TF-IDF.
2. **Add character n-grams** — Set `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))` to catch spelling patterns and misspellings common in fake news.
3. **Class-weight tuning** — If one class is slightly harder, adjust `class_weight='balanced'`.
4. **Ensemble** — Average the predictions of LR + SVM + NB for a small boost.
5. **Only then** — If you still need more, switch to `transformers` (DistilBERT) with GPU.

---

## Project Files in Your Workspace

| File | Description |
|------|-------------|
| `fake_news_train.py` | Full training pipeline (copy into Colab) |
| `fake.csv` | Fake news dataset |
| `true.csv` | True news dataset |
| `best_fake_news_model.pkl` | Saved after training (produced by the script) |
| `tfidf_vectorizer.pkl` | Saved vectorizer (produced by the script) |

---

## Quick Start Checklist

- [ ] Upload `fake.csv` and `true.csv` to Colab
- [ ] Copy the code from `fake_news_train.py` into a Colab cell
- [ ] Run the cell (2–5 minutes)
- [ ] Check which model wins on Validation F1
- [ ] Note the **Test Accuracy** — this is your real-world score
- [ ] Download the `.pkl` files if you want to use the model later
- [ ] Try `predict_article()` with your own headlines

---

Good luck with your project! 🚀
