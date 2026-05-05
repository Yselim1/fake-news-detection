"""
Stage 1 — Out-of-context image detection using CLIP cosine similarity.

Dataset: clip-benchmark/wds_flickr30k (from HuggingFace, ~600MB).
  True pairs  (label=1): original image + its own caption  -> pristine
  False pairs (label=0): image + caption from another image -> out-of-context

This mirrors the NewsCLIPpings construction methodology: real images reused
in the wrong context are the most common form of visual misinformation.

Strategy:
  - First run: load dataset, compute CLIP embeddings, cache to disk.
  - Subsequent runs: load cached embeddings instantly.
  - Compute cosine similarity; high = pristine, low = out-of-context.

Outputs (results/):
  stage1_img_embs.npy, stage1_txt_embs.npy, stage1_labels.npy  <- cache
  stage1_metrics.json
  stage1_score_distribution.png
  stage1_auc_roc.png
  stage1_precision_recall.png
"""

import json, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score,
)
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

RESULTS_DIR   = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_ID    = "clip-benchmark/wds_flickr30k"
CACHE_IMG     = RESULTS_DIR / "stage1_img_embs.npy"
CACHE_TXT     = RESULTS_DIR / "stage1_txt_embs.npy"
CACHE_LBL     = RESULTS_DIR / "stage1_labels.npy"
BATCH_SIZE    = 64
RANDOM_SEED   = 42


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip(device):
    print(f"Loading CLIP ({CLIP_MODEL_ID}) on {device}...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


@torch.no_grad()
def encode_batch(model, processor, images: list, texts: list, device):
    inputs = processor(
        images=images, text=texts,
        return_tensors="pt", padding=True,
        truncation=True, max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    i_emb = outputs.image_embeds
    t_emb = outputs.text_embeds
    i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    return i_emb.cpu().float().numpy(), t_emb.cpu().float().numpy()


def build_pairs(ds):
    """
    Returns (images, captions, labels) with 50/50 true/false pairs.
    True pair:  sample[i].image + sample[i].caption
    False pair: sample[i].image + sample[j].caption  (j != i, random)
    """
    n = len(ds)
    rng = random.Random(RANDOM_SEED)

    images, captions, labels = [], [], []

    # True pairs
    for sample in ds:
        images.append(sample["jpg"].convert("RGB"))
        txt = sample["txt"]
        # txt can be a list of captions or a single string
        captions.append(txt[0] if isinstance(txt, list) else txt)
        labels.append(1)

    # False pairs: same images, shuffled captions
    shuffled_idx = list(range(n))
    rng.shuffle(shuffled_idx)
    # Ensure no index maps to itself
    for i in range(n):
        if shuffled_idx[i] == i:
            swap = (i + 1) % n
            shuffled_idx[i], shuffled_idx[swap] = shuffled_idx[swap], shuffled_idx[i]

    true_captions = captions.copy()
    for i in range(n):
        images.append(ds[i]["jpg"].convert("RGB"))
        captions.append(true_captions[shuffled_idx[i]])
        labels.append(0)

    return images, captions, labels


def compute_and_cache_embeddings(ds, model, processor, device):
    print(f"Building pairs from {len(ds)} samples (-> {len(ds)*2} total pairs)...")
    images, captions, labels = build_pairs(ds)

    print(f"Computing CLIP embeddings for {len(images)} pairs (caching to disk)...")
    img_embs, txt_embs = [], []

    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Encoding"):
        batch_imgs    = images[i : i + BATCH_SIZE]
        batch_txts    = captions[i : i + BATCH_SIZE]
        try:
            i_emb, t_emb = encode_batch(model, processor, batch_imgs, batch_txts, device)
        except Exception as e:
            print(f"  [skip batch {i}] {e}")
            continue
        img_embs.append(i_emb)
        txt_embs.append(t_emb)

    img_embs = np.vstack(img_embs)
    txt_embs = np.vstack(txt_embs)
    labels   = np.array(labels[:len(img_embs)])   # trim if batches were skipped

    np.save(CACHE_IMG, img_embs)
    np.save(CACHE_TXT, txt_embs)
    np.save(CACHE_LBL, labels)
    print(f"Cached -> {RESULTS_DIR}/stage1_*.npy")
    return img_embs, txt_embs, labels


def load_cached():
    print("Loading cached CLIP embeddings...")
    img_embs = np.load(CACHE_IMG)
    txt_embs = np.load(CACHE_TXT)
    labels   = np.load(CACHE_LBL)
    print(f"  {len(labels)} pairs loaded from cache.")
    return img_embs, txt_embs, labels


def run_stage1(force_recompute: bool = False):
    if not force_recompute and CACHE_IMG.exists():
        img_embs, txt_embs, labels = load_cached()
    else:
        device = get_device()
        model, processor = load_clip(device)
        print(f"\nLoading {DATASET_ID} (test split)...")
        ds = load_dataset(DATASET_ID, split="test")
        print(f"  {len(ds)} images")
        img_embs, txt_embs, labels = compute_and_cache_embeddings(ds, model, processor, device)

    # Cosine similarity (embeddings already L2-normalised)
    scores = (img_embs * txt_embs).sum(axis=1)

    # Normalise scores to [0,1] for AUC (higher = more likely pristine)
    probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    threshold = float(np.median(scores))
    preds     = (scores >= threshold).astype(int)

    metrics = {
        "auc_roc":       float(roc_auc_score(labels, probs)),
        "avg_precision": float(average_precision_score(labels, probs)),
        "accuracy":      float(accuracy_score(labels, preds)),
        "f1_macro":      float(f1_score(labels, preds, average="macro")),
        "f1_weighted":   float(f1_score(labels, preds, average="weighted")),
        "threshold":     threshold,
        "n_pairs":       int(len(labels)),
        "dataset":       DATASET_ID,
        "method":        "CLIP cosine similarity (true vs. shuffled pairs)",
    }
    report = classification_report(labels, preds, target_names=["out-of-context", "pristine"])
    print(f"\nStage 1 results:\n{report}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}  |  AP: {metrics['avg_precision']:.4f}")

    np.save(RESULTS_DIR / "stage1_scores.npy", scores)
    np.save(RESULTS_DIR / "stage1_probs.npy",  probs)

    with open(RESULTS_DIR / "stage1_metrics.json", "w") as f:
        json.dump({**metrics, "report": report}, f, indent=2)

    _plot_score_distribution(scores, labels)
    _plot_roc(labels, probs)
    _plot_precision_recall(labels, probs)
    return metrics


def _plot_score_distribution(scores, labels):
    fig, ax = plt.subplots(figsize=(8, 4))
    for lbl, name, color in [(0, "Out-of-context (shuffled)", "#E05C5C"),
                              (1, "Pristine (matched)",       "#4C8CBF")]:
        sns.kdeplot(scores[labels == lbl], ax=ax, label=name,
                    color=color, fill=True, alpha=0.4)
    ax.set_xlabel("CLIP Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Stage 1 — CLIP Score Distribution (Flickr30k)")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "stage1_score_distribution.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


def _plot_roc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"CLIP ViT-B/32 (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 1 — ROC Curve (Out-of-context Detection)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "stage1_auc_roc.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


def _plot_precision_recall(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#DD8452", lw=2, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Stage 1 — Precision-Recall Curve")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "stage1_precision_recall.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Force recompute embeddings even if cache exists")
    args = parser.parse_args()
    run_stage1(force_recompute=args.recompute)
