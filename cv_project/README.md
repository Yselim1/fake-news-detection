# Multimodal Misinformation Detection — CV Component

Two-stage visual credibility pipeline:
- **Stage 1**: Out-of-context detection (CLIP cosine similarity on NewsCLIPpings)
- **Stage 2**: Manipulation detection (fine-tuned ResNet-18 on CASIA v2)

## Setup

```bash
pip install -r requirements.txt
```

## Dataset — CASIA v2 (Stage 2) — ~180MB download

Download from: https://github.com/namtpham/casia2groundtruth  
or: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset

Extract so the structure is:
```
data/
  CASIA2/
    Au/    (authentic images)
    Tp/    (tampered images)
```

> **Note:** NewsCLIPpings (Stage 1) is loaded automatically from HuggingFace (`verite/newsclippings`).

## Step 1 — Run Stage 1 (CLIP out-of-context detector)

Downloads `verite/newsclippings` from HuggingFace (~1–2GB) automatically on first run.
CLIP embeddings are computed once and cached to `results/stage1_*.npy` — subsequent runs load instantly.

```bash
python src/stage1_clip.py
# Outputs: results/stage1_metrics.json
#          results/stage1_score_distribution.png
#          results/stage1_auc_roc.png
#          results/stage1_precision_recall.png

# Force recompute embeddings (e.g. after changing model):
python src/stage1_clip.py --recompute
```

## Step 2 — Run Stage 2 (manipulation detector)

Requires CASIA v2 extracted to `data/CASIA2/` first.

```bash
python src/stage2_cnn.py
# Outputs: results/stage2_model.pth
#          results/stage2_metrics.json
#          results/stage2_auc_roc.png
#          results/stage2_confusion_matrix.png
```

## Step 3 — Grad-CAM visualizations

```bash
python src/gradcam.py --n_samples 5
# Outputs: results/gradcam/*.png
```

## Results layout

```
results/
  stage1_metrics.json
  stage1_score_distribution.png
  stage1_auc_roc.png
  stage2_metrics.json
  stage2_auc_roc.png
  stage2_confusion_matrix.png
  gradcam/
    gradcam_authentic_*.png
    gradcam_tampered_*.png
```

## Models

| Stage | Model | Task |
|---|---|---|
| 1 | `openai/clip-vit-base-patch32` | Image-text cosine similarity |
| 2 | ResNet-18 (ImageNet pretrained) | Binary forgery classification |
