"""
Loads and merges TR-FakeNews and MiDe22 into a single DataFrame.
All sources are mapped to binary labels: 0 = fake/satirical, 1 = real.

TR-FakeNews columns: title, description, status (1=real, 0=fake)
MiDe22 columns:      tweet, label ('True'/'False'/'Other')
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def load_tr_fakenews() -> pd.DataFrame:
    """isakulaksiz/turkish-fake-news-detection — news headlines + summaries."""
    ds = load_dataset("isakulaksiz/turkish-fake-news-detection", split="train")
    df = ds.to_pandas()

    # Combine title + description for richer text
    df["text"] = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()
    df["label"] = df["status"].astype(int)   # 1=real, 0=fake
    df["source"] = "tr_fakenews"
    return df[["text", "label", "source"]].dropna()


def load_mide22() -> pd.DataFrame:
    """ogozcelik/turkish-fake-news-detection (MiDe22) — Turkish tweets."""
    ds = load_dataset("ogozcelik/turkish-fake-news-detection", split="train")
    df = ds.to_pandas()

    # Columns: tweet, label ('True'/'False'/'Other')
    df = df.rename(columns={"tweet": "text"})
    label_map = {"true": 1, "false": 0}
    df["label"] = df["label"].str.lower().map(label_map)
    df = df.dropna(subset=["label"])          # drops 'other' rows
    df["label"] = df["label"].astype(int)
    df["source"] = "mide22"
    return df[["text", "label", "source"]].dropna()


def load_zaytung(csv_path: str = None) -> pd.DataFrame:
    """Zaytung satirical headlines (0) + real RSS news (1). Loads from scraped CSV."""
    path = Path(csv_path) if csv_path else DATA_DIR / "zaytung_dataset.csv"
    if not path.exists():
        print(f"  Zaytung CSV not found at {path}. Run zaytung_scraper.py first.")
        return pd.DataFrame(columns=["text", "label", "source"])
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["source"] = "zaytung"
    return df


def load_all(include_zaytung: bool = True) -> pd.DataFrame:
    """Returns merged DataFrame with columns: text, label, source."""
    print("Loading TR-FakeNews...")
    tr = load_tr_fakenews()
    print(f"  TR-FakeNews: {len(tr)} samples  (fake={(tr.label==0).sum()}, real={(tr.label==1).sum()})")

    print("Loading MiDe22...")
    mi = load_mide22()
    print(f"  MiDe22:      {len(mi)} samples  (fake={(mi.label==0).sum()}, real={(mi.label==1).sum()})")

    parts = [tr, mi]

    if include_zaytung:
        print("Loading Zaytung...")
        za = load_zaytung()
        if len(za) > 0:
            print(f"  Zaytung:     {len(za)} samples  (fake={(za.label==0).sum()}, real={(za.label==1).sum()})")
            parts.append(za)

    df = pd.concat(parts, ignore_index=True)
    df = df[df["text"].str.strip().str.len() > 15].reset_index(drop=True)

    print(f"\nCombined: {len(df)} samples")
    print(df.groupby(["source", "label"]).size().to_string())
    return df


if __name__ == "__main__":
    df = load_all()
    out = DATA_DIR / "combined_dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved -> {out}")
