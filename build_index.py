import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from embedder import ClipEmbedder  # your existing embedder.py

REF_DIR = Path("ref")
IMG_DIR = REF_DIR / "images"
CARDS_JSON = REF_DIR / "cards.json"
EMB_NPY = REF_DIR / "embeddings.npy"
IDS_JSON = REF_DIR / "ids.json"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def main():
    print("=== build_index.py ===")
    print("cwd:", Path.cwd())
    print("Expecting:")
    print(" -", CARDS_JSON)
    print(" -", IMG_DIR)

    if not CARDS_JSON.exists():
        raise FileNotFoundError(f"Missing {CARDS_JSON}")

    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Missing {IMG_DIR}")

    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)

    if not isinstance(cards, dict) or len(cards) == 0:
        raise RuntimeError("cards.json is empty or not a dict.")

    # Count image files
    img_files = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in VALID_EXTS]
    print(f"Found {len(img_files)} image files in ref/images")

    print("Loading embedder (OpenCLIP)...")
    embedder = ClipEmbedder()  # uses cuda automatically if available
    print("Embedder device:", getattr(embedder, "device", "unknown"))

    ids = []
    embs = []
    missing = 0

    for card_id, meta in tqdm(cards.items(), desc="Embedding"):
        img_name = meta.get("image")

        if not img_name:
            # fallback to card_id with png
            img_name = f"{card_id}.png"

        img_path = IMG_DIR / img_name
        if not img_path.exists():
            # try other extensions if needed
            base = img_path.with_suffix("")
            found = None
            for ext in VALID_EXTS:
                candidate = base.with_suffix(ext)
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                missing += 1
                continue
            img_path = found

        img = Image.open(img_path).convert("RGB")
        vec = embedder.embed_pil(img)  # returns normalized vector
        ids.append(card_id)
        embs.append(vec)

    print(f"Embedded {len(ids)} cards. Missing images for {missing} cards.")

    if len(embs) == 0:
        raise RuntimeError(
            "No embeddings created. This usually means the 'image' field in cards.json "
            "doesn't match filenames in ref/images, or you ran from the wrong folder."
        )

    embs = np.stack(embs, axis=0).astype(np.float32)
    np.save(EMB_NPY, embs)

    with open(IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print("Saved:", EMB_NPY, "shape=", embs.shape)
    print("Saved:", IDS_JSON, "count=", len(ids))
    print("Done.")

if __name__ == "__main__":
    main()
