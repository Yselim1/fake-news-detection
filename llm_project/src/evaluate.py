"""
Loads saved prediction arrays from all three regimes and produces:
  - results/comparison_bar.png   — accuracy & F1 bar chart across regimes
  - results/confusion_matrices.png
  - results/roc_curves.png
  - results/error_analysis.txt  — 10 false positives + 10 false negatives

Usage:
    python evaluate.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

REGIMES = [
    ("Zero-shot",   "zero_shot"),
    ("Few-shot (32)", "few_shot_32"),
    ("Fine-tuned",  "fine_tune"),
]


def load_arrays(prefix: str):
    probs  = np.load(RESULTS_DIR / f"{prefix}_probs.npy")
    preds  = np.load(RESULTS_DIR / f"{prefix}_preds.npy")
    labels = np.load(RESULTS_DIR / f"{prefix}_labels.npy")
    return probs, preds, labels


def load_metrics(prefix: str) -> dict:
    candidates = [
        RESULTS_DIR / f"{prefix}_results.json",
        RESULTS_DIR / f"few_shot_32_results.json" if "few" in prefix else None,
    ]
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                return json.load(f).get("metrics", {})
    return {}


# ---------------------------------------------------------------------------
# 1. Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(all_metrics: dict):
    labels_list  = [n for n, _ in REGIMES]
    accuracies   = [all_metrics[k].get("accuracy", 0)    for _, k in REGIMES]
    f1_macros    = [all_metrics[k].get("f1_macro", 0)    for _, k in REGIMES]
    f1_weighted  = [all_metrics[k].get("f1_weighted", 0) for _, k in REGIMES]

    x  = np.arange(len(labels_list))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, accuracies,  w, label="Accuracy",    color="#4C72B0")
    ax.bar(x,     f1_macros,   w, label="F1 (macro)",  color="#DD8452")
    ax.bar(x + w, f1_weighted, w, label="F1 (weighted)", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("LLM Project — Performance across Prompting Regimes")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    out = RESULTS_DIR / "comparison_bar.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# 2. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(arrays: dict):
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(14, 4))
    for ax, (name, key) in zip(axes, REGIMES):
        if key not in arrays:
            continue
        _, preds, labels = arrays[key]
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fake", "Real"],
                    yticklabels=["Fake", "Real"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# 3. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(arrays: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, key), color in zip(REGIMES, colors):
        if key not in arrays:
            continue
        probs, _, labels = arrays[key]
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — LLM Project")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# 4. Error analysis (fine-tuned model)
# ---------------------------------------------------------------------------

def error_analysis():
    pred_csv = RESULTS_DIR / "fine_tune_test_predictions.csv"
    if not pred_csv.exists():
        print("No prediction CSV found; skipping error analysis.")
        return

    df = pd.read_csv(pred_csv)
    fp = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("prob", ascending=False).head(10)
    fn = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("prob", ascending=True).head(10)

    lines = ["=== FALSE POSITIVES (predicted Real, actually Fake) ===\n"]
    for _, row in fp.iterrows():
        lines.append(f"[prob={row['prob']:.3f}] [{row.get('source','?')}] {row['text'][:200]}\n")

    lines.append("\n=== FALSE NEGATIVES (predicted Fake, actually Real) ===\n")
    for _, row in fn.iterrows():
        lines.append(f"[prob={row['prob']:.3f}] [{row.get('source','?')}] {row['text'][:200]}\n")

    out = RESULTS_DIR / "error_analysis.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    arrays      = {}
    all_metrics = {}

    for name, key in REGIMES:
        try:
            arrays[key] = load_arrays(key)
            all_metrics[key] = load_metrics(key)
            print(f"Loaded arrays for {name}")
        except FileNotFoundError:
            print(f"[warn] No saved arrays for {name} — skipping")

    if not arrays:
        print("No result arrays found. Run zero_shot.py, few_shot.py, fine_tune.py first.")
        return

    if all_metrics:
        plot_comparison(all_metrics)
    plot_confusion_matrices(arrays)
    plot_roc_curves(arrays)
    error_analysis()

    print("\nAll evaluation plots saved to results/")


if __name__ == "__main__":
    main()
