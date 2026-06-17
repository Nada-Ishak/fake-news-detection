
'''
================================================================================
FAKE NEWS DETECTION - PHASE-BY-PHASE TRAINING PIPELINE (scikit-learn only)
================================================================================
Run this in Google Colab.
Upload fake.csv and true.csv first (or mount Google Drive).

Recommended Colab runtime: CPU (no GPU needed).
Estimated time: 2-5 minutes for the full pipeline.

Contents:
  Phase 1  → Imports & Setup
  Phase 2  → Data Loading & Cleaning
  Phase 3  → Text Preprocessing
  Phase 4  → Train / Validation / Test Split
  Phase 5  → TF-IDF Feature Extraction
  Phase 6  → Model Training (Naive Bayes, Logistic Regression, Linear SVM)
  Phase 7  → Evaluation & Model Comparison
  Phase 8  → Final Model Selection & Saving
  Phase 9  → Prediction on New Text
================================================================================
'''

# ================================================================================
# PHASE 1: IMPORTS & SETUP
# ================================================================================

import pickle

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# joblib is built-in on Colab; fall back to pickle if unavailable elsewhere
try:
    import joblib
except ImportError:
    import pickle as joblib
    def _dump(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    joblib.dump = _dump

# ================================================================================
# PHASE 2: DATA LOADING & CLEANING
# ================================================================================

'''
We load both CSVs, add a label column (1 = fake, 0 = true), merge them,
remove duplicates, and drop rows with empty title or text.
'''

import os
# Absolute path so the script works regardless of where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
fake_df = pd.read_csv(os.path.join(BASE_DIR, 'fake.csv'))
true_df = pd.read_csv(os.path.join(BASE_DIR, 'true.csv'))

print("--- Raw Data ---")
print(f"Fake news rows: {len(fake_df)}")
print(f"True news rows: {len(true_df)}")

# Add labels: 1 = FAKE, 0 = TRUE
fake_df['label'] = 1
true_df['label'] = 0

# Merge into one dataset
df = pd.concat([fake_df, true_df], ignore_index=True)

# Drop rows with empty title or text
df.dropna(subset=['title', 'text'], inplace=True)

# Remove duplicate articles (same text or title)
df.drop_duplicates(subset=['text'], keep='first', inplace=True)
df.drop_duplicates(subset=['title'], keep='first', inplace=True)

print(f"\nAfter cleaning: {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Percent fake: {df['label'].mean()*100:.1f}%")

# ================================================================================
# PHASE 3: TEXT PREPROCESSING
# ================================================================================

'''
We combine title + text into a single field for the model.
Cleaning steps:
  1. Lowercase everything
  2. Remove URLs, email addresses, and extra whitespace
  3. Remove punctuation (TF-IDF already handles this, but explicit cleaning helps)
  4. Strip extra spaces

Why combine title + text?
  - The title often contains strong signals (clickbait, emotional words).
  - The body gives context and factual detail.
  - Together they give the richest signal.
'''

def clean_text(text):
    """Basic text cleaner for news articles."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove punctuation (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Combine title and text into one column
df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean_text)

# Drop any rows that became empty after cleaning
df = df[df['content'].str.len() > 20].reset_index(drop=True)

print(f"\nAfter preprocessing: {len(df)} rows")
print("Sample cleaned text:")
print(df['content'].iloc[0][:300] + "...\n")

# ================================================================================
# PHASE 4: TRAIN / VALIDATION / TEST SPLIT
# ================================================================================

'''
We split into 3 sets:
  - Training   (70%)  → used to train the model
  - Validation (15%)  → used to tune/compare models
  - Test       (15%)  → used ONLY ONCE at the end for final evaluation

stratify=label ensures both sets keep the same 50/50 fake/true ratio.
random_state=42 makes the split reproducible every time.
'''

X = df['content']
y = df['label']

# First split: separate Test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: separate Validation set from the remaining 85% (15% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)
# 0.1765 * 0.85 ≈ 0.15, so we get 70/15/15

print(f"Train:      {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation: {len(X_val)}   ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test:       {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)")
print(f"Train fake ratio: {y_train.mean()*100:.1f}%")

# ================================================================================
# PHASE 5: TF-IDF FEATURE EXTRACTION
# ================================================================================

'''
TF-IDF (Term Frequency - Inverse Document Frequency) converts text into numbers.

How it works:
  - Term Frequency (TF)     = how often a word appears in one article
  - Inverse Document Freq   = down-weights words that appear in EVERY article
                                 (e.g., "the", "said", "Trump")
  - Result: each article becomes a vector of word-importance scores.

Parameters we use:
  max_features=50000  → keep only the 50,000 most informative words (speed + memory)
  ngram_range=(1,2)   → use single words AND two-word phrases ("fake news" matters!)
  stop_words='english'→ remove common English words (the, and, is...)
  min_df=2            → ignore words that appear in fewer than 2 articles (reduces noise)
  max_df=0.9          → ignore words that appear in >90% of articles (reduces noise)

Why TF-IDF for this project?
  - Fast and lightweight (works great on Colab CPU).
  - Highly interpretable (you can see which words drive predictions).
  - Proven strong baseline for news classification.
'''

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),      # unigrams + bigrams
    stop_words='english',
    min_df=2,
    max_df=0.9
)

# Fit on training data ONLY, then transform all sets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ================================================================================
# PHASE 6: MODEL TRAINING
# ================================================================================

'''
We train 3 classical text-classification models:

1. Multinomial Naive Bayes (NB)
   - Best for: fast baseline, works great with TF-IDF word counts.
   - Pros:  trains instantly, surprisingly effective for text.
   - Cons:  assumes words are independent (naive), can be slightly weaker than LR/SVM.

2. Logistic Regression (LR)
   - Best for: strong, interpretable linear classifier.
   - Pros:  fast, gives probability scores, weights show which words matter.
   - Cons:  linear only (but text is often well-separated linearly).

3. Linear SVM (LinearSVC)
   - Best for: maximum-margin separation; often the best classical model for text.
   - Pros:  very robust to high-dimensional data, excellent generalization.
   - Cons:  no probability output by default (we can calibrate if needed).

Why these 3?
  - They are the "gold standard" trio for text classification with TF-IDF.
  - One will almost always be the best for your dataset.
'''

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    'Linear SVM': LinearSVC(random_state=42)
}

trained_models = {}

for name, model in models.items():
    print(f"\n→ Training {name}...")
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    print(f"   Done.")

# ================================================================================
# PHASE 7: EVALUATION & MODEL COMPARISON
# ================================================================================

'''
Metrics we track:
  - Accuracy    = overall correct percentage
  - Precision   = of articles predicted fake, how many were really fake
  - Recall      = of all real fake articles, how many did we catch
  - F1 Score    = harmonic mean of precision & recall (best single metric)

Why F1?
  - In fake-news detection, both false positives (calling real news fake)
    and false negatives (missing actual fake news) matter.
  - F1 balances both concerns.
'''

def evaluate_model(name, model, X, y, dataset_name="Validation"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name} — {dataset_name} Set")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y, preds, target_names=['True News', 'Fake News']))
    print(f"  Confusion Matrix:")
    print(f"                 Pred True   Pred Fake")
    cm = confusion_matrix(y, preds)
    print(f"  Actual True       {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  Actual Fake       {cm[1,0]:5d}       {cm[1,1]:5d}")
    return {'name': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

results = []
for name, model in trained_models.items():
    results.append(evaluate_model(name, model, X_val_tfidf, y_val, "Validation"))

# ================================================================================
# PHASE 8: FINAL MODEL SELECTION & TEST-SET EVALUATION
# ================================================================================

'''
We pick the model with the highest Validation F1 score,
then evaluate it on the held-out Test set to get an honest estimate
of real-world performance.
'''

best = max(results, key=lambda x: x['f1'])
print(f"\n{'='*60}")
print(f"  BEST MODEL (by Validation F1): {best['name']}")
print(f"  Validation F1 = {best['f1']:.4f}")
print(f"{'='*60}")

best_model = trained_models[best['name']]

# Final evaluation on TEST SET (never seen during training or validation)
print(f"\n>>> FINAL TEST-SET EVALUATION for {best['name']} <<<")
final_metrics = evaluate_model(best['name'], best_model, X_test_tfidf, y_test, "Test")

# Save the best model and vectorizer for later use
joblib.dump(best_model, os.path.join(BASE_DIR, 'best_fake_news_model.pkl'))
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
print("\n✅ Saved: best_fake_news_model.pkl")
print("✅ Saved: tfidf_vectorizer.pkl")

# ================================================================================
# PHASE 9: PREDICTION ON NEW TEXT
# ================================================================================

'''
Simple function to classify any new article you paste in.
Usage:
  result = predict_article("Your headline here", "Your article body here")
  print(result)
'''

def predict_article(title, text, model=best_model, vectorizer=vectorizer):
    """Classify a new article as Fake or True."""
    raw = str(title) + ' ' + str(text)
    cleaned = clean_text(raw)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    # Get probability if model supports it (Logistic Regression & Naive Bayes do)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vec)[0]
        confidence = prob[pred]
    else:
        # LinearSVC doesn't give probabilities by default
        confidence = None

    label_name = "FAKE NEWS" if pred == 1 else "TRUE NEWS"
    if confidence is not None:
        return f"Prediction: {label_name} (confidence: {confidence*100:.1f}%)"
    return f"Prediction: {label_name}"


# -------------------- DEMO PREDICTIONS --------------------
print("\n" + "="*60)
print("  DEMO PREDICTIONS")
print("="*60)

# Example 1: A clearly fake-sounding headline
fake_title = "SHOCKING: Government Hiding Alien Base in Antarctica, Scientists Confirm"
fake_text = "Top secret documents leaked today reveal that the government has been covering up an alien base for decades. Sources say the president knows everything."
print(f"\n1. {predict_article(fake_title, fake_text)}")

# Example 2: A neutral, factual headline
true_title = "Federal Reserve Raises Interest Rates by 0.25 Percentage Points"
true_text = "The Federal Reserve announced Wednesday that it will raise the benchmark interest rate by 0.25 percentage points, citing continued economic growth and stable inflation."
print(f"2. {predict_article(true_title, true_text)}")

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
