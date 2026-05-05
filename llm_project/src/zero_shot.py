"""
Zero-shot fake news classification using multilingual XLM-RoBERTa NLI model.

The model (joeddav/xlm-roberta-large-xnli) is prompted via NLI:
    premise  = article text
    hypothesis = "Bu haber gerçektir." (This news is real.)
Entailment score -> probability of being real (label=1).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HYPOTHESIS = "Bu haber gerçektir."   # "This news is real."
MODEL_ID   = "joeddav/xlm-roberta-large-xnli"


def run_zero_shot(df: pd.DataFrame, batch_size: int = 16) -> dict:
    """
    Runs zero-shot classification on df['text'].
    Returns dict with predictions, probabilities, and metrics.
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_ID} on {'GPU' if device == 0 else 'CPU'}...")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_ID,
        device=device,
    )

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    probs, preds = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out   = clf(batch, candidate_labels=["gerçek", "sahte"],
                    hypothesis_template="Bu haber {}.")
        for res in out:
            # 'gerçek' (real) entailment score -> prob of label=1
            real_score = res["scores"][res["labels"].index("gerçek")]
            probs.append(real_score)
            preds.append(1 if real_score >= 0.5 else 0)

        if (i // batch_size) % 5 == 0:
            print(f"  {i + len(batch)}/{len(texts)}")

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "auc_roc": roc_auc_score(labels, probs),
        "report": classification_report(labels, preds, target_names=["fake", "real"]),
    }
    print(f"\nZero-shot results:\n{metrics['report']}")

    results = {
        "regime": "zero_shot",
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
        "report": metrics["report"],
        "preds": preds,
        "probs": probs,
        "labels": labels,
    }

    out_path = RESULTS_DIR / "zero_shot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k not in ("preds", "probs", "labels")}, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")

    # Save arrays separately for plotting
    np.save(RESULTS_DIR / "zero_shot_probs.npy",  np.array(probs))
    np.save(RESULTS_DIR / "zero_shot_preds.npy",  np.array(preds))
    np.save(RESULTS_DIR / "zero_shot_labels.npy", np.array(labels))

    return results


if __name__ == "__main__":
    from data_loader import load_all
    from sklearn.model_selection import train_test_split

    df = load_all()
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    # Use a 500-sample subset for zero-shot (slow model)
    sample = test_df.sample(min(500, len(test_df)), random_state=42)
    run_zero_shot(sample)
