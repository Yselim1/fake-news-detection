"""
Scrapes Zaytung satirical headlines (label=0) by:
  1. Collecting all article links from the homepage
  2. Enumerating recent newsIDs to fetch more headlines via page titles

Real news (label=1) fetched via RSS using feedparser.

Output: data/zaytung_dataset.csv  (columns: text, label, url, source_name)

Usage:
    python zaytung_scraper.py
    python zaytung_scraper.py --max_id_range 2000
"""

import argparse
import time
import csv
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

BASE_URL  = "https://www.zaytung.com"
# Article types on current zaytung.com CMS
ARTICLE_TYPES = ["haberdetay.asp", "fotohaberdetay.asp", "sondakikadetay.asp"]

REAL_NEWS_RSS = [
    ("Hurriyet",  "https://www.hurriyet.com.tr/rss/anasayfa"),
    ("CNN Turk",  "https://www.cnnturk.com/feed/rss/news/turkey"),
    ("NTV",       "https://www.ntv.com.tr/gundem.rss"),
    ("Milliyet",  "https://www.milliyet.com.tr/rss/rssNew/gundemRss.xml"),
]


# ---------------------------------------------------------------------------
# Zaytung — collect article links from homepage
# ---------------------------------------------------------------------------

def collect_homepage_links() -> list[dict]:
    """Collect all article hrefs from the Zaytung homepage."""
    try:
        r = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  [homepage] failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    seen, articles = set(), []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(atype in href for atype in ARTICLE_TYPES):
            full_url = href if href.startswith("http") else f"{BASE_URL}/{href}"
            if full_url not in seen:
                seen.add(full_url)
                articles.append(full_url)

    print(f"  [homepage] found {len(articles)} article links")
    return articles


# ---------------------------------------------------------------------------
# Zaytung — fetch title from article detail page
# ---------------------------------------------------------------------------

def fetch_title(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title")
    if not title:
        return None

    text = title.get_text(strip=True)
    # Strip site prefixes (any "Zaytung XYZ - " pattern)
    import re
    text = re.sub(r'^Zaytung[^-]*-\s*', '', text).strip().rstrip(".")

    # Filter out non-satirical sections (astrology, blog, cinema)
    skip_keywords = ["astroloji", "sinema", "harry potter", "dizi", "film",
                     "kitap", "blog", "etkinlik", "rss"]
    if any(kw in text.lower() for kw in skip_keywords):
        return None
    if len(text) < 20:
        return None
    return text


# ---------------------------------------------------------------------------
# Zaytung — enumerate recent newsIDs
# ---------------------------------------------------------------------------

def enumerate_recent_ids(latest_id: int = 437500, id_range: int = 1000) -> list[str]:
    """Generate haberdetay URLs for a range of recent newsIDs."""
    urls = []
    for nid in range(latest_id, latest_id - id_range, -1):
        urls.append(f"{BASE_URL}/haberdetay.asp?newsid={nid}")
    return urls


# ---------------------------------------------------------------------------
# Real news via RSS (feedparser)
# ---------------------------------------------------------------------------

def fetch_real_news() -> list[dict]:
    all_items = []
    for name, rss_url in REAL_NEWS_RSS:
        try:
            feed = feedparser.parse(rss_url)
            count = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(strip=True)
                text = (title + " " + summary).strip() if summary else title
                if len(text) > 20:
                    all_items.append({
                        "text": text,
                        "label": 1,
                        "url": entry.get("link", ""),
                        "source_name": name,
                    })
                    count += 1
            print(f"  [{name}] {count} items")
        except Exception as e:
            print(f"  [{name}] RSS failed: {e}")
        time.sleep(0.3)
    return all_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_id_range: int = 1000):
    print("=== Collecting Zaytung articles ===")

    # Step 1: homepage links
    homepage_urls = collect_homepage_links()

    # Step 2: enumerate recent IDs
    print(f"  Enumerating {max_id_range} recent newsIDs...")
    id_urls = enumerate_recent_ids(id_range=max_id_range)

    all_urls = list(dict.fromkeys(homepage_urls + id_urls))  # deduplicate, keep order
    print(f"  Total URLs to fetch: {len(all_urls)}")

    fake_articles = []
    for i, url in enumerate(all_urls):
        title = fetch_title(url)
        if title:
            fake_articles.append({
                "text": title,
                "label": 0,
                "url": url,
                "source_name": "Zaytung",
            })
        if i % 100 == 0:
            print(f"  [{i}/{len(all_urls)}] collected {len(fake_articles)} satirical headlines so far")
        time.sleep(0.15)  # be polite

    print(f"\nZaytung total: {len(fake_articles)} headlines\n")

    print("=== Fetching real news via RSS ===")
    real_articles = fetch_real_news()
    print(f"Real news total: {len(real_articles)}\n")

    all_data = fake_articles + real_articles
    output = DATA_DIR / "zaytung_dataset.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "url", "source_name"])
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Saved {len(all_data)} samples -> {output}")
    print(f"  Satirical (Zaytung): {sum(1 for d in all_data if d['label'] == 0)}")
    print(f"  Real news (RSS):     {sum(1 for d in all_data if d['label'] == 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_id_range", type=int, default=1000,
                        help="How many recent newsIDs to enumerate (default: 1000)")
    args = parser.parse_args()
    main(max_id_range=args.max_id_range)
