"""
Few-shot regime: fine-tune xlm-roberta-base on a small labelled subset
(K examples per class, default K=32) then evaluate on the full test set.

This simulates the few-shot scenario where only a handful of labelled
examples are available, contrasting with the full fine-tuning baseline.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_CKPT = Path(__file__).parent.parent / "results" / "few_shot_model"

BASE_MODEL = "xlm-roberta-base"
K_SHOTS    = 32   # examples per class in training


def tokenize(batch, tokenizer, max_len=64):
    return tokenizer(batch["text"], truncation=True, max_length=max_len)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def run_few_shot(df_full, k: int = K_SHOTS):
    """
    Samples k examples per class as training set; evaluates on 20% held-out test.
    """
    _, test_df = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full["label"])

    # Build k-shot train set (balanced)
    few_shot_parts = [
        df_full[df_full["label"] == lbl].sample(k, random_state=42)
        for lbl in [0, 1]
    ]
    import pandas as pd
    train_df = pd.concat(few_shot_parts).sample(frac=1, random_state=42)

    print(f"Few-shot train size: {len(train_df)}  |  test size: {len(test_df)}")

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
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=False,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Predict on full test set
    raw_preds = trainer.predict(test_ds)
    probs  = torch.softmax(torch.tensor(raw_preds.predictions), dim=-1)[:, 1].numpy()
    preds  = np.argmax(raw_preds.predictions, axis=-1)
    labels = test_df["label"].tolist()

    report = classification_report(labels, preds, target_names=["fake", "real"])
    print(f"\nFew-shot ({k}-shot) results:\n{report}")

    results = {
        "regime": f"few_shot_{k}",
        "k_shots": k,
        "metrics": {
            "accuracy":    accuracy_score(labels, preds),
            "f1_macro":    f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "auc_roc":     roc_auc_score(labels, probs),
        },
        "report": report,
    }

    out_path = RESULTS_DIR / f"few_shot_{k}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")

    np.save(RESULTS_DIR / f"few_shot_{k}_probs.npy",  probs)
    np.save(RESULTS_DIR / f"few_shot_{k}_preds.npy",  preds)
    np.save(RESULTS_DIR / f"few_shot_{k}_labels.npy", np.array(labels))

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all
    df = load_all()
    run_few_shot(df, k=K_SHOTS)
