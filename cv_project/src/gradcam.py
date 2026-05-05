"""
Grad-CAM visualizations for Stage 2 (ResNet-18 manipulation detector).

Loads the saved model, selects N sample images per class, generates
Grad-CAM heatmaps, and saves overlaid PNG files to results/gradcam/.

Usage:
    python gradcam.py
    python gradcam.py --n_samples 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DATA_DIR    = Path(__file__).parent.parent / "data" / "CASIA2"
GT_DIR      = DATA_DIR / "CASIA 2 Groundtruth"   # ground truth masks (optional)
RESULTS_DIR = Path(__file__).parent.parent / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
VAL_TF   = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = RESULTS_DIR / "stage2_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No saved model at {ckpt}. Run stage2_cnn.py first.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)
    return model


def get_samples(n_per_class: int):
    au_imgs = sorted(p for p in (DATA_DIR / "Au").iterdir() if p.suffix.lower() in EXTS)[:n_per_class]
    tp_imgs = sorted(p for p in (DATA_DIR / "Tp").iterdir() if p.suffix.lower() in EXTS)[:n_per_class]
    return [(p, 0) for p in au_imgs] + [(p, 1) for p in tp_imgs]


def find_gt_mask(img_path: Path) -> np.ndarray | None:
    """Looks for a ground truth mask matching the tampered image filename."""
    if not GT_DIR.exists():
        return None
    # GT masks are often named similarly to Tp images, sometimes with _gt suffix
    for suffix in [img_path.suffix, ".png", ".bmp"]:
        candidates = [
            GT_DIR / img_path.name,
            GT_DIR / (img_path.stem + "_gt" + suffix),
            GT_DIR / (img_path.stem + suffix),
        ]
        for p in candidates:
            if p.exists():
                mask = np.array(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE)))
                return (mask > 127).astype(np.float32)  # binary mask
    return None


def run_gradcam(n_samples: int = 5):
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model         = load_model(device)
    target_layer  = [model.layer4[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layer)

    samples     = get_samples(n_samples)
    label_names = ["authentic", "tampered"]

    for img_path, true_label in samples:
        pil_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np  = np.array(pil_img).astype(np.float32) / 255.0
        tensor  = VAL_TF(pil_img).unsqueeze(0).to(device)

        grayscale_cam = cam_extractor(input_tensor=tensor)[0]
        overlay       = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        with torch.no_grad():
            pred_idx = model(tensor).argmax(dim=1).item()

        # Check for ground truth mask (tampered images only)
        gt_mask = find_gt_mask(img_path) if true_label == 1 else None
        n_cols  = 3 if gt_mask is not None else 2

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Original\nTrue: {label_names[true_label]}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM\nPred: {label_names[pred_idx]}")
        axes[1].axis("off")

        if gt_mask is not None:
            axes[2].imshow(gt_mask, cmap="Reds", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth\nForgery Mask")
            axes[2].axis("off")

        status = "correct" if pred_idx == true_label else "wrong"
        fname  = f"gradcam_{label_names[true_label]}_{img_path.stem}_{status}.png"
        out    = GRADCAM_DIR / fname
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of images per class to visualize")
    args = parser.parse_args()
    run_gradcam(args.n_samples)
