# scripts/ingest_reddit.py

import os, json, requests
import praw

DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "metadata.jsonl")
os.makedirs(IMG_DIR, exist_ok=True)

# Fill these in with your Reddit app credentials
reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="MathArtHarvester/0.1"
)

def scrape_reddit(subreddit="mathart", limit=200):
    idx = len(open(META_PATH).read().splitlines()) + 1
    for post in reddit.subreddit(subreddit).top("all", limit=limit):
        if not post.url.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        # Extract code/formula from post.selftext
        formula = None
        for line in post.selftext.splitlines():
            if any(sym in line for sym in ("sin(", "cos(", "exp(", "fract", "{", "}")):
                formula = line.strip()
                break
        if not formula:
            continue  # skip if no formula found

        # Download image
        img_resp = requests.get(post.url)
        img_id = f"{idx:04d}"
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        with open(img_path, "wb") as f:
            f.write(img_resp.content)

        # Metadata
        meta = {
            "id": img_id,
            "source": "Reddit-mathart",
            "title": post.title,
            "formula": formula,
            "description": post.title,
            "tags": ["reddit", "procedural"]
        }
        with open(META_PATH, "a") as mf:
            mf.write(json.dumps(meta) + "\n")

        idx += 1

if __name__ == "__main__":
    scrape_reddit()
