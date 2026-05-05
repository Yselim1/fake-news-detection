# Turkish Fake News Detection — LLM Component

Binary classification of Turkish news headlines (fake vs. real) using XLM-RoBERTa.
Three learning regimes are compared: zero-shot NLI, 32-shot fine-tuning, and full fine-tuning.
This is the text component of a larger multimodal misinformation detection system.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. A GPU is recommended for fine-tuning; zero-shot and few-shot run on CPU.

## Datasets

| Dataset | Source | Description |
|---|---|---|
| TR-FakeNews | HuggingFace: `isakulaksiz/turkish-fake-news-detection` | ~5,300 fact-checked Turkish headlines |
| MiDe22 | HuggingFace: `ogozcelik/MiDe22-dataset` | ~2,400 Turkish social media posts |
| Zaytung | Scraped (see Step 1) | Satirical Turkish news (The Onion equivalent) |

Real news sources: Hürriyet, Milliyet, CNN Türk, TRT. Ground-truth labels from Teyit.org.

## Running the pipeline

**Step 1 — Scrape Zaytung (optional, CSV included in `data/`)**
```bash
python src/zaytung_scraper.py --max_pages 20
# Output: data/zaytung_dataset.csv
```

**Step 2 — Build combined dataset**
```bash
python src/data_loader.py
# Output: data/combined_dataset.csv  (~8,078 samples, 80/20 split)
```

**Step 3 — Run experiments**
```bash
python src/zero_shot.py          # Zero-shot NLI (no training)
python src/few_shot.py           # 32-shot fine-tuning (64 examples total)
python src/fine_tune.py          # Full fine-tuning on all training data
```

**Step 4 — Evaluate and generate figures**
```bash
python src/evaluate.py
# Outputs: results/comparison_bar.png, confusion_matrices.png,
#          roc_curves.png, error_analysis.txt
```

## Demo (Colab)

`colab/predict_demo.py` runs 20 hand-crafted headlines (Turkish + English) through the
fine-tuned model. Requires the model saved to Google Drive at
`MyDrive/LLM-CV/results/finetuned_model`.

## Models

| Regime | Model | Notes |
|---|---|---|
| Zero-shot | `joeddav/xlm-roberta-large-xnli` | NLI entailment score as credibility signal |
| Few-shot | `xlm-roberta-base` | 64 labeled examples, 10 epochs |
| Fine-tuned | `xlm-roberta-base` | 6,462 examples, early stopping on F1-macro |

## Results

| Regime | Accuracy | F1-macro | AUC-ROC |
|---|---|---|---|
| Zero-shot | 65.8% | 57.5% | 0.696 |
| Few-shot (32/class) | 75.0% | 74.8% | 0.855 |
| Full fine-tuning | **93.1%** | **92.8%** | **0.969** |

Result figures are in `results/`. Pre-computed metrics are in `results/*_results.json`.
