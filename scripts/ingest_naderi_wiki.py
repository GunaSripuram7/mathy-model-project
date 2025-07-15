# scripts/ingest_naderi_wiki.py

import os, json, re, requests
from bs4 import BeautifulSoup

DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "metadata.jsonl")
BASE_URL = "https://mathworld.wolfram.com/ParametricCurves.html"
DOMAIN = "https://mathworld.wolfram.com"

os.makedirs(IMG_DIR, exist_ok=True)

def scrape_mathworld_parametrics():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = soup.find_all("div", class_="item")  # each curve entry
    idx = 1

    for item in items:
        img_tag = item.find("img")
        if not img_tag:
            continue

        thumb_url = img_tag["src"]
        img_url = DOMAIN + thumb_url

        title = img_tag.get("alt") or f"curve_{idx}"
        desc_tag = item.find("p")
        desc = desc_tag.get_text(" ", strip=True) if desc_tag else ""

        # Look for formula patterns nearby
        formula = ""
        formula_match = re.search(r"x\(t\)\s*=\s*[^;]+;\s*y\(t\)\s*=\s*[^;\)]+", desc)
        if formula_match:
            formula = formula_match.group(0)

        img_id = f"{idx:04d}"
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        try:
            r_img = requests.get(img_url)
            r_img.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r_img.content)
        except Exception as e:
            print(f"[WARN] Failed to download {img_url}:", e)
            continue

        meta = {
            "id": img_id,
            "source": "MathWorld_ParametricCurves",
            "title": title,
            "formula": formula,
            "description": desc,
            "tags": ["parametric", "mathworld"]
        }

        with open(META_PATH, "a") as mf:
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

        print(f"✅ Saved {img_id}: {title}")
        idx += 1

if __name__ == "__main__":
    scrape_mathworld_parametrics()
