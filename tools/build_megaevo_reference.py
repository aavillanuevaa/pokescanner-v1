# tools/build_megaevo_reference.py
# Complete copy-paste script:
# - Finds the Pokémon TCG API set by name ("Mega Evolution" by default)
# - Downloads all card "large" images into ref/images/
# - Generates ref/cards.json
# - Robust retries/backoff for slow API + timeouts
#
# Usage:
#   pip install requests tqdm
#   python tools/build_megaevo_reference.py
#
# Optional (recommended): set an API key to reduce rate limits:
#   PowerShell: setx POKEMONTCG_API_KEY "YOUR_KEY"
#
# After running, you'll have:
#   ref/cards.json
#   ref/images/MEG_001.png, ...

import os
import re
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://api.pokemontcg.io/v2"

# Output locations (relative to project root)
REF_DIR = Path("ref")
IMG_DIR = REF_DIR / "images"
CARDS_JSON_PATH = REF_DIR / "cards.json"

# Naming scheme
FILENAME_PREFIX = "MEG_"  # produces MEG_001.png, MEG_132A.png, etc.

# ---- HTTP session with retries ----
SESSION = requests.Session()
_retry = Retry(
    total=8,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)

def _headers() -> Dict[str, str]:
    """
    Pokémon TCG API supports optional API key via X-Api-Key header.
    Set env var POKEMONTCG_API_KEY to use it.
    """
    h = {"User-Agent": "poke-scanner/1.0"}
    api_key = os.getenv("POKEMONTCG_API_KEY", "").strip()
    if api_key:
        h["X-Api-Key"] = api_key
    return h

def _sleep_backoff(attempt: int) -> None:
    wait = min(30.0, (2 ** attempt)) + random.uniform(0, 0.6)
    time.sleep(wait)

def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Robust GET that retries on timeouts / 429 / 5xx.
    Uses separate connect and read timeouts.
    """
    timeout = (10, 120)  # (connect, read)

    for attempt in range(10):
        try:
            r = SESSION.get(url, params=params, headers=_headers(), timeout=timeout)

            if r.status_code == 429:
                _sleep_backoff(attempt)
                continue

            r.raise_for_status()
            return r.json()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            _sleep_backoff(attempt)
            continue
        except requests.exceptions.RequestException:
            # Could be transient; retry a few times
            _sleep_backoff(attempt)
            if attempt == 9:
                raise

    raise RuntimeError(f"Failed request after retries: {url}")

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")
    return s

def make_internal_id(card_number: str) -> str:
    """
    Convert API card number (might include letters for secret rares)
    into our file key.
    Examples:
      "1" -> MEG_001
      "12" -> MEG_012
      "132" -> MEG_132
      "132a" -> MEG_132A
      "S1" -> MEG_S1
    """
    num = str(card_number).strip()

    if num.isdigit():
        return f"{FILENAME_PREFIX}{int(num):03d}"

    num_clean = sanitize_filename(num).upper()

    m = re.match(r"^(\d+)(.*)$", num_clean)
    if m:
        d = int(m.group(1))
        tail = m.group(2)
        return f"{FILENAME_PREFIX}{d:03d}{tail}"

    return f"{FILENAME_PREFIX}{num_clean}"

def find_set_id_by_name(set_name: str) -> str:
    """
    Search sets by name and return the set id that matches exactly (case-insensitive).
    Falls back to first result if no exact match.
    """
    data = _get_json(f"{API_BASE}/sets", params={"q": f'name:"{set_name}"', "pageSize": 250})
    sets = data.get("data", [])
    if not sets:
        raise RuntimeError(f'No sets found for name "{set_name}".')

    for s in sets:
        if str(s.get("name", "")).strip().lower() == set_name.strip().lower():
            return s["id"]

    print("WARNING: No exact match found. Using first search result:")
    print(json.dumps(sets[0], indent=2))
    return sets[0]["id"]

def confirm_set(set_id: str) -> Dict[str, Any]:
    info = _get_json(f"{API_BASE}/sets/{set_id}").get("data", {})
    print(
        "SET CONFIRM:",
        info.get("id"),
        "|",
        info.get("name"),
        "| releaseDate:",
        info.get("releaseDate"),
        "| total:",
        info.get("total"),
    )
    return info

def fetch_all_cards_for_set(set_id: str) -> List[Dict[str, Any]]:
    """
    Pull all cards from a set using paging.
    """
    all_cards: List[Dict[str, Any]] = []
    page = 1
    page_size = 100  # smaller pages reduce timeouts

    while True:
        params = {
            "q": f"set.id:{set_id}",
            "page": page,
            "pageSize": page_size,
            "select": "id,name,number,rarity,supertype,subtypes,types,hp,attacks,images,set",
        }
        resp = _get_json(f"{API_BASE}/cards", params=params)
        chunk = resp.get("data", [])
        all_cards.extend(chunk)

        total_count = resp.get("totalCount", None)
        if total_count is None:
            if len(chunk) < page_size:
                break
        else:
            if len(all_cards) >= int(total_count):
                break

        page += 1

    return all_cards

def download_image(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    timeout = (10, 120)

    for attempt in range(10):
        try:
            r = SESSION.get(url, headers=_headers(), timeout=timeout)

            if r.status_code == 429:
                _sleep_backoff(attempt)
                continue

            r.raise_for_status()
            out_path.write_bytes(r.content)
            return

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            _sleep_backoff(attempt)
            continue
        except requests.exceptions.RequestException:
            _sleep_backoff(attempt)
            if attempt == 9:
                raise

def sort_key(card: Dict[str, Any]) -> Tuple[int, int, str]:
    n = str(card.get("number", "")).strip()

    if n.isdigit():
        return (0, int(n), "")

    m = re.match(r"^(\d+)(.*)$", n)
    if m:
        return (1, int(m.group(1)), m.group(2))

    return (2, 999999, n)

def build_cards_json_and_images(set_name: str = "Mega Evolution") -> None:
    REF_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Finding set id for "{set_name}"...')
    set_id = find_set_id_by_name(set_name)
    print(f"Using set id: {set_id}")

    # Print set details to ensure it's the right expansion
    set_info = confirm_set(set_id)

    print("Fetching all cards in set...")
    cards = fetch_all_cards_for_set(set_id)
    print(f"Found {len(cards)} cards.")

    cards = sorted(cards, key=sort_key)

    out: Dict[str, Any] = {}

    for c in tqdm(cards, desc="Downloading images + building cards.json"):
        number = str(c.get("number", "")).strip()
        internal_id = make_internal_id(number)

        images = c.get("images") or {}
        img_url = images.get("large") or images.get("small")
        if not img_url:
            continue

        filename = f"{internal_id}.png"
        out_path = IMG_DIR / filename

        # Download the clean reference scan
        download_image(img_url, out_path)

        # Clean attacks
        attacks = c.get("attacks") or []
        attacks_clean = []
        for a in attacks:
            attacks_clean.append({
                "name": a.get("name", ""),
                "cost": a.get("cost", []),
                "convertedEnergyCost": a.get("convertedEnergyCost", None),
                "damage": a.get("damage", ""),
                "text": a.get("text", ""),
            })

        out[internal_id] = {
            "api_id": c.get("id"),
            "name": c.get("name", ""),
            "number": number,
            "rarity": c.get("rarity", ""),
            "supertype": c.get("supertype", ""),
            "subtypes": c.get("subtypes", []),
            "types": c.get("types", []),
            "hp": c.get("hp", ""),
            "attacks": attacks_clean,
            "set": {
                "id": (c.get("set") or {}).get("id", set_id),
                "name": (c.get("set") or {}).get("name", set_name),
                "releaseDate": (c.get("set") or {}).get("releaseDate", set_info.get("releaseDate")),
                "total": (c.get("set") or {}).get("total", set_info.get("total")),
            },
            "image": filename,
            "image_url": img_url,
        }

    CARDS_JSON_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {CARDS_JSON_PATH}")
    print(f"Images in: {IMG_DIR}")
    print("Done.")

if __name__ == "__main__":
    build_cards_json_and_images("Mega Evolution")
