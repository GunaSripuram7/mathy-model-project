# scripts/ingest_shadertoy.py

import os, json, requests

API_KEY = "YOUR_SHADERTOY_API_KEY"
SHADER_IDS = ["MlXXWR", "Xl3XXX", ...]  # list of top SDF shaders

DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "metadata.jsonl")
os.makedirs(IMG_DIR, exist_ok=True)

def scrape_shadertoy():
    idx = len(open(META_PATH).read().splitlines()) + 1
    headers = {"Authorization": API_KEY}
    for sid in SHADER_IDS:
        r = requests.get(f"https://www.shadertoy.com/api/v1/shaders/{sid}", headers=headers).json()
        code = r["Shader"]["renderpass"][0]["code"]
        # Construct preview URL
        img_url = f"https://www.shadertoy.com/media/shaders/{sid}.png"

        img_resp = requests.get(img_url)
        img_id = f"{idx:04d}"
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        with open(img_path, "wb") as f:
            f.write(img_resp.content)

        meta = {
            "id": img_id,
            "source": "ShaderToy",
            "title": r["Shader"]["info"]["name"],
            "formula": code,
            "description": r["Shader"]["info"].get("description",""),
            "tags": ["sdf","raymarch"]
        }
        with open(META_PATH, "a") as mf:
            mf.write(json.dumps(meta) + "\n")
        idx += 1

if __name__ == "__main__":
    scrape_shadertoy()
