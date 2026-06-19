"""
Training pipeline for the Fake News Detection model.

This is the same logic that was originally in the Jupyter notebook,
reorganized into a runnable script:

  1. Load the WELFake dataset
  2. Clean it (dedupe, drop unused columns, build "content")
  3. Preprocess text (lowercase, strip punctuation, remove stopwords, lemmatize)
  4. TF-IDF vectorize
  5. Train & compare several classifiers
  6. Cross-validate the strongest model
  7. Save evaluation plots
  8. Persist the best model + vectorizer to ml/models/

Run from the project root with:
    python -m ml.train
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

from ml import config
from ml.evaluate import plot_confusion_matrix, plot_roc_curve
from ml.preprocessing import build_content, clean_text


def load_data() -> pd.DataFrame:
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATASET_PATH}.\n"
            f"Download WELFake_Dataset.csv and place it in {config.DATA_DIR}/ "
            "(see ml/data/README.md)."
        )
    data = pd.read_csv(config.DATASET_PATH)
    print(f"Loaded dataset: {data.shape}")
    return data


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop_duplicates()
    data = data.drop(columns=["subject", "date", "Unnamed: 0"], errors="ignore")

    data["content"] = data.apply(lambda row: build_content(row.get("title"), row.get("text")), axis=1)

    print("Cleaning text (this can take a few minutes on the full dataset)...")
    data["clean_content"] = data["content"].apply(clean_text)

    # Drop rows that ended up empty after cleaning, and any missing labels
    data = data.dropna(subset=["label"])
    data = data[data["clean_content"].astype(str).str.strip().ne("")]
    return data


def vectorize(data: pd.DataFrame):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["clean_content"],
        data["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}")
    return vectorizer, X_train, X_test, y_train, y_test


def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVC": LinearSVC(),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_scores),
        })
        fitted_models[name] = (model, y_pred, y_scores)
        print(f"{name} done -> Accuracy: {results[-1]['Accuracy']:.4f}")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print("\nModel Comparison:\n", results_df)
    return results_df, fitted_models


def cross_validate_best(data: pd.DataFrame) -> None:
    """5-fold stratified CV for the Random Forest pipeline (text -> TF-IDF -> RF)."""
    cv_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES, ngram_range=config.TFIDF_NGRAM_RANGE)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    print("\nRunning 5-fold cross-validation (Random Forest)...")
    cv_scores = cross_val_score(
        cv_pipeline, data["clean_content"], data["label"],
        cv=cv, scoring="accuracy", n_jobs=1, error_score="raise",
    )
    print("Cross Validation Accuracy:", cv_scores)
    print(f"Mean Accuracy: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")


def main():
    data = load_data()
    data = clean_dataframe(data)
    vectorizer, X_train, X_test, y_train, y_test = vectorize(data)
    results_df, fitted_models = train_and_compare(X_train, X_test, y_train, y_test)

    best_model_name = results_df.iloc[0]["Model"]
    best_model, best_y_pred, best_y_scores = fitted_models[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print(classification_report(y_test, best_y_pred, target_names=config.LABEL_NAMES))

    plot_confusion_matrix(y_test, best_y_pred, best_model_name, config.CONFUSION_MATRIX_PATH)
    plot_roc_curve(y_test, best_y_scores, best_model_name, results_df.iloc[0]["ROC-AUC"], config.ROC_CURVE_PATH)
    print(f"Saved plots to {config.REPORTS_DIR}/")

    if config.RUN_CROSS_VALIDATION:
        cross_validate_best(data)

    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"\nSaved '{best_model_name}' -> {config.MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer -> {config.VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
