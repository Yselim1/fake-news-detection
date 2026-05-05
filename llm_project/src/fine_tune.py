"""
Full fine-tuning of xlm-roberta-base on the combined Turkish fake news dataset.
Trains on 80% split, evaluates on 20%.  Saves best model checkpoint.

Usage:
    python fine_tune.py
    python fine_tune.py --epochs 5 --batch_size 16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_CKPT  = Path(__file__).parent.parent / "results" / "finetuned_model"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_MODEL = "xlm-roberta-base"
MAX_LEN    = 64


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def run_fine_tune(df_full, epochs: int = 4, batch_size: int = 16):
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full["label"]
    )
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))

    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    test_ds  = test_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    test_ds  = test_ds.rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODEL_CKPT),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        warmup_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    raw_preds = trainer.predict(test_ds)
    probs  = torch.softmax(torch.tensor(raw_preds.predictions), dim=-1)[:, 1].numpy()
    preds  = np.argmax(raw_preds.predictions, axis=-1)
    labels = test_df["label"].tolist()

    report = classification_report(labels, preds, target_names=["fake", "real"])
    print(f"\nFull fine-tune results:\n{report}")

    results = {
        "regime": "full_fine_tune",
        "metrics": {
            "accuracy":    accuracy_score(labels, preds),
            "f1_macro":    f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "auc_roc":     roc_auc_score(labels, probs),
        },
        "report": report,
    }

    out_path = RESULTS_DIR / "fine_tune_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")

    np.save(RESULTS_DIR / "fine_tune_probs.npy",  probs)
    np.save(RESULTS_DIR / "fine_tune_preds.npy",  preds)
    np.save(RESULTS_DIR / "fine_tune_labels.npy", np.array(labels))

    # Save test_df for error analysis
    test_df = test_df.copy()
    test_df["pred"]  = preds
    test_df["prob"]  = probs
    test_df.to_csv(RESULTS_DIR / "fine_tune_test_predictions.csv", index=False)

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df = load_all()
    run_fine_tune(df, epochs=args.epochs, batch_size=args.batch_size)
