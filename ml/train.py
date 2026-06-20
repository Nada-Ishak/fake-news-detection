from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "ml" / "data" / "WELFake_Dataset.csv"
MODEL_DIR = BASE_DIR / "ml" / "models" / "distilbert_model"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[["title", "text", "label"]].copy()

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["text"] = (df["title"] + " " + df["text"]).str.strip()
    df = df[["text", "label"]]

    # Ensure labels are int
    df["label"] = df["label"].astype(int)

    print("Dataset shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts())

    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Main Training
def main():
    df = load_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1},
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\nFinal Evaluation:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print(f"\nModel and tokenizer saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
