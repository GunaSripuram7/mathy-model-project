# scripts/ingest_mathworld.py

import os, json, re, time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ─── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = "dataset"
IMG_DIR     = os.path.join(DATA_DIR, "images")
META_PATH   = os.path.join(DATA_DIR, "metadata.jsonl")

TOPIC_PAGES = [
    "https://mathworld.wolfram.com/topics/PlaneCurves.html",
    "https://mathworld.wolfram.com/topics/SpaceCurves.html",
    "https://mathworld.wolfram.com/topics/Fractals.html"
]
DOMAIN = "https://mathworld.wolfram.com"

DELAY = 0.5

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_soup(url):
    r = requests.get(url)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def get_existing_ids():
    if not os.path.exists(META_PATH):
        return set()
    with open(META_PATH) as f:
        return set(json.loads(line)["id"] for line in f if line.strip())

def write_metadata(meta):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

# ─── Scraper ──────────────────────────────────────────────────────────────────

def scrape_mathworld():
    os.makedirs(IMG_DIR, exist_ok=True)
    seen_ids = get_existing_ids()
    next_idx = len(seen_ids) + 1

    for topic_url in TOPIC_PAGES:
        print(f"→ Crawling topic: {topic_url}")
        soup = fetch_soup(topic_url)

        # ❗ Fix: look for real links on the page
        links = soup.select("a[href^='/']")

        for a in links:
            href = a.get("href", "")
            if not href.endswith(".html"): continue
            entry_url = urljoin(DOMAIN, href)
            title     = a.text.strip()
            if not title: continue

            entry_id  = f"{next_idx:04d}"
            next_idx += 1

            if entry_id in seen_ids:
                continue

            print(f"  • Fetching entry: {title}")
            try:
                entry_soup = fetch_soup(entry_url)
            except Exception as e:
                print("    ✖ failed to load:", e)
                continue

            # 1) Extract formula
            text = entry_soup.get_text(" ", strip=True)
            formula = ""
            m = re.search(r"x\s*\(t\)\s*=[^;]+;\s*y\s*\(t\)\s*=[^;]+", text)
            if m:
                formula = m.group(0)
            else:
                m2 = re.search(r"z\s*=\s*[^;]+", text)
                if m2:
                    formula = m2.group(0)

            # 2) Extract the main image
            img_tag = entry_soup.select_one("div.body img")
            if not img_tag or not img_tag.get("src"):
                print("    ⚠ no image found")
                continue

            img_url = urljoin(DOMAIN, img_tag["src"])
            ext = os.path.splitext(img_url)[1].split("?")[0]
            img_path = os.path.join(IMG_DIR, entry_id + ext)

            try:
                rimg = requests.get(img_url)
                rimg.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(rimg.content)
            except Exception as e:
                print("    ✖ failed to download image:", e)
                continue

            # 3) Description
            desc = ""
            ptag = entry_soup.select_one("div.body p")
            if ptag:
                desc = ptag.get_text(" ", strip=True)

            meta = {
                "id": entry_id,
                "source": "MathWorld",
                "title": title,
                "url": entry_url,
                "image": os.path.relpath(img_path, DATA_DIR),
                "formula": formula,
                "description": desc,
                "tags": ["math", "curve"]
            }
            write_metadata(meta)
            print(f"    ✅ Saved {entry_id}")

            time.sleep(DELAY)

    print("→ Scraping complete!")

if __name__ == "__main__":
    scrape_mathworld()
