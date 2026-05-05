"""
Stage 2 — Image manipulation detection using fine-tuned ResNet-18.

Dataset: CASIA v2 image forgery dataset.
  Download instructions: see README.md
  Expected layout after extraction:
    data/CASIA2/
      Au/        — authentic images (~7491)
      Tp/        — tampered images  (~5123)

Fine-tunes ResNet-18 (pretrained on ImageNet) with a binary classification head.

Outputs (saved to results/):
  stage2_model.pth
  stage2_metrics.json
  stage2_probs.npy, stage2_labels.npy
  stage2_auc_roc.png
  stage2_confusion_matrix.png
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

DATA_DIR    = Path(__file__).parent.parent / "data" / "CASIA2"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Training config
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-4
IMG_SIZE    = 224
NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CASIADataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, paths: list[Path], labels: list[int], transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @classmethod
    def from_casia_dir(cls, root: Path, transform=None):
        au_dir = root / "Au"
        tp_dir = root / "Tp"
        if not au_dir.exists() or not tp_dir.exists():
            raise FileNotFoundError(
                f"Expected {au_dir} and {tp_dir}.\n"
                "Please extract CASIA v2 dataset to data/CASIA2/ first.\n"
                "See README.md for download instructions."
            )
        paths, labels = [], []
        for p in sorted(au_dir.iterdir()):
            if p.suffix.lower() in cls.EXTS:
                paths.append(p); labels.append(0)   # 0 = authentic
        for p in sorted(tp_dir.iterdir()):
            if p.suffix.lower() in cls.EXTS:
                paths.append(p); labels.append(1)   # 1 = tampered
        print(f"Authentic: {labels.count(0)}  |  Tampered: {labels.count(1)}")
        return paths, labels


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(lbls.numpy())
    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths, labels = CASIADataset.from_casia_dir(DATA_DIR)
    train_p, test_p, train_l, test_l = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_tf, val_tf = get_transforms()
    train_ds = CASIADataset(train_p, train_l, train_tf)
    test_ds  = CASIADataset(test_p,  test_l,  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        probs, lbls = eval_epoch(model, test_loader, device)
        epoch_auc = roc_auc_score(lbls, probs)
        print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  AUC={epoch_auc:.4f}")
        scheduler.step()
        if epoch_auc > best_auc:
            best_auc   = epoch_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), RESULTS_DIR / "stage2_model.pth")

    probs, labels_arr = eval_epoch(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    report = classification_report(labels_arr, preds, target_names=["authentic", "tampered"])
    metrics = {
        "auc_roc":     roc_auc_score(labels_arr, probs),
        "accuracy":    accuracy_score(labels_arr, preds),
        "f1_macro":    f1_score(labels_arr, preds, average="macro"),
        "f1_weighted": f1_score(labels_arr, preds, average="weighted"),
    }
    print(f"\nStage 2 results:\n{report}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

    np.save(RESULTS_DIR / "stage2_probs.npy",  probs)
    np.save(RESULTS_DIR / "stage2_labels.npy", labels_arr)

    with open(RESULTS_DIR / "stage2_metrics.json", "w") as f:
        json.dump({**metrics, "report": report}, f, indent=2)

    _plot_confusion(labels_arr, preds)
    _plot_roc(labels_arr, probs)


def _plot_confusion(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Authentic", "Tampered"],
                yticklabels=["Authentic", "Tampered"], ax=ax)
    ax.set_title("Stage 2 — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    out = RESULTS_DIR / "stage2_confusion_matrix.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


def _plot_roc(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#55A868", label=f"ResNet-18 (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 2 — ROC Curve (Manipulation Detection)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "stage2_auc_roc.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    run_stage2()
