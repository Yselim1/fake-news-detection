# Multimodal Fake News Detection

**Author:** Yavuz Selim Gezginci — 20210808028

A two-part project for detecting misinformation using both text and image signals.
The system is **pattern-based**: it learns stylistic and visual cues from labeled data
rather than fact-checking against external sources.

| Component | Approach | Key Result |
|---|---|---|
| [`llm_project/`](llm_project/) | XLM-RoBERTa fine-tuned on Turkish headlines | 93.1% acc · AUC 0.969 |
| [`cv_project/`](cv_project/) | CLIP (out-of-context) + ResNet-18 (manipulation) | AUC 0.999 · AUC 0.847 |

---

## Repository structure

```
fake-news-detection/
├── llm_project/          # Text credibility — Turkish fake news detection
│   ├── src/              # Zero-shot, few-shot, fine-tuning, evaluation scripts
│   ├── colab/            # Colab demo (predict_demo.py)
│   ├── results/          # Output figures and metrics JSON
│   └── requirements.txt
└── cv_project/           # Visual credibility — out-of-context + manipulation detection
    ├── src/              # Stage 1 (CLIP), Stage 2 (ResNet-18), Grad-CAM
    ├── results/          # Output figures, Grad-CAM samples, metrics JSON
    └── requirements.txt
```

---

## Quick start

Each component has its own dependencies and instructions. Follow the README in each folder:

- **LLM project →** [`llm_project/README.md`](llm_project/README.md)
- **CV project →** [`cv_project/README.md`](cv_project/README.md)

Both components require **Python 3.10+** and a GPU is recommended for Stage 2 training
and LLM fine-tuning. All models load from HuggingFace automatically on first run.

> **Datasets:** Raw datasets are not included in this repo (see individual READMEs for
> download links). Pre-computed result files and figures are included under `results/`.
