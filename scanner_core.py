# scanner_core.py
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cv_card import find_card_quad, warp_quad
from embedder import cosine_topk


# =========================
# Paths
# =========================
REF_DIR = Path("ref")
CARDS_JSON = REF_DIR / "cards.json"
EMB_NPY = REF_DIR / "embeddings.npy"
IDS_JSON = REF_DIR / "ids.json"
IMG_DIR = REF_DIR / "images"


# =========================
# Config (TUNED for low quality webcam + low lag)
# =========================
@dataclass(frozen=True)
class Config:
    # Stability (more forgiving)
    stable_frames_required: int = 6
    max_center_move: float = 20.0
    max_area_change: float = 0.25

    # Recognition throttling
    min_seconds_between_id: float = 0.25

    # Acceptance gates (balanced)
    min_accept_clip: float = 0.26
    min_accept_final: float = 0.32
    min_accept_roi: float = 0.12
    min_margin: float = 0.03

    # Blur rejection
    min_blur_var: float = 25.0

    # CLIP
    topk_candidates: int = 8
    w_clip: float = 0.65
    w_roi: float = 0.35

    # Warp size
    warp_w: int = 512
    warp_h: int = 712

    # ROI zones
    roi_specs = (
        (0.60, 0.78, 0.98, 0.98),
        (0.15, 0.62, 0.85, 0.78),
    )

    # Camera
    cam_index: int = 0
    cam_backend: int = cv2.CAP_DSHOW   # recommended on Windows



CFG = Config()


# =========================
# Camera
# =========================
def open_camera():
    cap = cv2.VideoCapture(CFG.cam_index, CFG.cam_backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Lower res = smoother
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    return cap



# =========================
# Preprocessing
# =========================
def sharpen_bgr(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def normalize_lighting_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# =========================
# Geometry helpers
# =========================
def scale_quad(quad, sx, sy):
    q = quad.astype(np.float32).copy()
    q[:, 0] *= sx
    q[:, 1] *= sy
    return q


def mirror_quad_x(quad, width):
    q = quad.copy()
    q[:, 0] = (width - 1) - q[:, 0]
    return q


def quad_center_and_area(quad):
    center = quad.mean(axis=0)
    area = cv2.contourArea(quad.astype(np.float32))
    return center, float(area)


def update_stability(quad, last_quad, stable, cfg: Config):
    if last_quad is None:
        return quad, 1

    c, a = quad_center_and_area(quad)
    c0, a0 = quad_center_and_area(last_quad)

    center_move = float(np.linalg.norm(c - c0))
    area_change = abs(a - a0) / max(a0, 1.0)

    if center_move < cfg.max_center_move and area_change < cfg.max_area_change:
        return quad, stable + 1
    return quad, 1


# =========================
# ROI verification (ORB)
# =========================
def crop_roi(img, spec):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = spec
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def roi_similarity_orb(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = cv2.resize(a, (300, 170), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (300, 170), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=500)
    kpa, desa = orb.detectAndCompute(a, None)
    kpb, desb = orb.detectAndCompute(b, None)
    if desa is None or desb is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desa, desb)
    if not matches:
        return 0.0

    good = [m for m in matches if m.distance < 55]
    return float(min(1.0, len(good) / 60.0))


def compute_roi_score(warped, ref_path, cfg: Config):
    if not ref_path.exists():
        return 0.0
    ref = cv2.imread(str(ref_path))
    if ref is None:
        return 0.0

    ref = cv2.resize(ref, (cfg.warp_w, cfg.warp_h), interpolation=cv2.INTER_AREA)
    ref = normalize_lighting_bgr(ref)

    scores = []
    for spec in cfg.roi_specs:
        q = crop_roi(warped, spec)
        r = crop_roi(ref, spec)
        if q is None or r is None or q.size == 0 or r.size == 0:
            continue
        scores.append(roi_similarity_orb(q, r))

    return float(np.mean(scores)) if scores else 0.0


# =========================
# Data + Recognition
# =========================
def load_reference():
    with open(CARDS_JSON, "r", encoding="utf-8") as f:
        cards = json.load(f)
    ref_embs = np.load(EMB_NPY).astype(np.float32)
    with open(IDS_JSON, "r", encoding="utf-8") as f:
        ref_ids = json.load(f)
    return cards, ref_embs, ref_ids


def rank_cards(q, ref_embs, ref_ids, cards, warped, cfg: Config):
    idxs, clip_scores = cosine_topk(q, ref_embs, cfg.topk_candidates)
    out = []
    for i, clip in zip(idxs, clip_scores):
        cid = ref_ids[int(i)]
        meta = cards.get(cid, {})
        ref_img = IMG_DIR / meta.get("image", f"{cid}.png")
        roi = compute_roi_score(warped, ref_img, cfg)
        final = cfg.w_clip * float(clip) + cfg.w_roi * float(roi)
        out.append((final, float(clip), float(roi), cid))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# =========================
# Frame processor (what the UI calls)
# =========================
def process_frame(frame_bgr, embedder, cards, ref_embs, ref_ids, state, cfg: Config):
    """
    Returns a dict with everything the UI needs to draw + display text.
    `state` is mutated (stability + throttling + last lines).
    """
    frame = sharpen_bgr(frame_bgr)

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    quad_small = find_card_quad(small)

    result = {
        "frame": frame,
        "has_quad": False,
        "quad": None,
        "warped": None,
        "blur_var": None,
        "attempted": False,
        "lines": state.get("lines", []),
        "stable": state.get("stable", 0),
        "stable_required": cfg.stable_frames_required,

        # NEW: best candidate info when attempted
        "best_id": None,
        "best_final": None,
        "best_clip": None,
        "best_roi": None,
        "best_margin": None,
        "accepted": False,
    }

    if quad_small is None:
        state["stable"] = 0
        result["stable"] = 0
        return result

    quad = scale_quad(quad_small, 2.0, 2.0)
    state["last_quad"], state["stable"] = update_stability(
        quad, state.get("last_quad"), state.get("stable", 0), cfg
    )

    warped = warp_quad(frame, quad, cfg.warp_w, cfg.warp_h)
    warped_n = normalize_lighting_bgr(warped)

    b = blur_score(warped)
    result.update({
        "has_quad": True,
        "quad": quad,
        "warped": warped_n,
        "blur_var": b,
        "stable": state["stable"],
    })

    now = time.time()
    can_try = (
        b > cfg.min_blur_var
        and state["stable"] >= cfg.stable_frames_required
        and (now - state.get("last_id", 0.0)) > cfg.min_seconds_between_id
    )

    if not can_try:
        return result

    state["last_id"] = now
    result["attempted"] = True

    q = embedder.embed_pil(bgr_to_pil(warped_n)).astype(np.float32)
    ranked = rank_cards(q, ref_embs, ref_ids, cards, warped_n, cfg)
    if not ranked:
        state["lines"] = ["No matches"]
        result["lines"] = state["lines"]
        return result

    best_final, best_clip, best_roi, best_id = ranked[0]
    second_final = ranked[1][0] if len(ranked) > 1 else -1.0
    margin = best_final - second_final

    ok_clip = best_clip >= cfg.min_accept_clip
    ok_final = best_final >= cfg.min_accept_final
    ok_roi = best_roi >= cfg.min_accept_roi
    ok_margin = margin >= cfg.min_margin

    if best_clip > 0.70 and best_roi > 0.45:
        ok_margin = True

    accepted = bool(ok_clip and ok_final and ok_roi and ok_margin)

    result.update({
        "best_id": best_id,
        "best_final": float(best_final),
        "best_clip": float(best_clip),
        "best_roi": float(best_roi),
        "best_margin": float(margin),
        "accepted": accepted,
    })

    if accepted:
        card = cards.get(best_id, {})
        state["lines"] = [
            f"{card.get('name','?')} #{card.get('number','')}",
            f"F={best_final:.3f} C={best_clip:.3f} R={best_roi:.3f} Δ={margin:.3f}",
        ]
    else:
        state["lines"] = [
            "Low confidence",
            f"F={best_final:.3f} C={best_clip:.3f} R={best_roi:.3f} Δ={margin:.3f}",
        ]

    result["lines"] = state["lines"]
    return result
