import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

# Base directory for saving
IMAGE_DIR = Path("images")
META_FILE = Path("metadata.jsonl")
BASE_URL = "https://mathworld.wolfram.com/"

# Ensure image folder exists
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Topics to scrape
topic_urls = [
    "https://mathworld.wolfram.com/topics/PlaneCurves.html",
    "https://mathworld.wolfram.com/topics/SpaceCurves.html",
    "https://mathworld.wolfram.com/topics/Fractals.html",
]

def get_entry_links(topic_url):
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select("ul li a")
    return [urljoin(BASE_URL, link.get("href")) for link in links if link.get("href")]

def download_svg(page_url):
    try:
        page = requests.get(page_url)
        soup = BeautifulSoup(page.text, "html.parser")

        title = soup.find("title").text.replace(" -- from Wolfram MathWorld", "").strip()
        img_tag = soup.find("img", src=lambda s: s and "eps-svg" in s and s.endswith(".svg"))
        if not img_tag:
            print(f"  ⚠ No SVG image found for {title}")
            return

        img_url = urljoin(BASE_URL, img_tag['src'])
        img_name = img_url.split("/")[-1]
        img_path = IMAGE_DIR / img_name

        # Download image
        img_data = requests.get(img_url).content
        with open(img_path, "wb") as f:
            f.write(img_data)

        # Extract formula from the first paragraph (if any)
        description = soup.find("meta", attrs={"name": "description"})
        formula = description['content'] if description else ""

        # Save metadata
        metadata = {
            "title": title,
            "source": page_url,
            "image": str(img_path.relative_to(Path.cwd())),
            "description": formula.strip()
        }
        with open(META_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        print(f"  ✓ Saved: {title}")

    except Exception as e:
        print(f"  ❌ Error scraping {page_url}: {e}")

if __name__ == "__main__":
    for topic_url in topic_urls:
        print(f"→ Crawling topic: {topic_url}")
        entry_links = get_entry_links(topic_url)
        for link in entry_links:
            download_svg(link)
    print("→ Scraping complete!")
