import time
import cv2
from scanner_core import CFG, mirror_quad_x
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"   # or "SILENT" if supported

WINDOW_NAME = "Pokemon Card Scanner (Scanner Mode)"
MIRROR_UI_DEFAULT = False

# SCANNER behavior
SCAN_SECONDS = 2.0

# Inset (top-right live zoom)
INSET_W = 220
INSET_H = 310
INSET_MARGIN = 12

# =========================
# LOCKED UI BUTTONS (CIRCLES)
# =========================
BTN_RADIUS = 26
BTN_MARGIN = 20
BTN_SPACING = 14

BTN_GREEN = (60, 200, 60)
BTN_GREEN_ACTIVE = (90, 255, 90)

BTN_RED = (60, 60, 200)
BTN_RED_ACTIVE = (90, 90, 255)

# =========================
# UI FONT (GLOBAL)
# =========================
UI_FONT = cv2.FONT_HERSHEY_DUPLEX


FONT_BIG = {
    "face": UI_FONT,
    "scale": 1.5,
    "thickness": 3,
}

FONT_MED = {
    "face": UI_FONT,
    "scale": 1.0,
    "thickness": 2,
}

FONT_SMALL = {
    "face": UI_FONT,
    "scale": 0.5,
    "thickness": 1,
}

def put_text(img, text, org, preset, color=(255, 255, 255)):
    cv2.putText(
        img,
        text,
        org,
        preset["face"],
        preset["scale"],
        color,
        preset["thickness"],
    )



# -------------------------
# Drawing + hit-test helpers
# -------------------------
def draw_circle_button(img, cx, cy, r, color):
    cv2.circle(img, (cx, cy), r, color, -1)


def point_in_circle(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2


def draw_hud(display, blur_var, min_blur_var, lines, always_show_blur_label=False):
    lines = lines or []
    blur_pos, lines_y = hud_bottom_left_layout(display, len(lines))

    # Blur line
    if always_show_blur_label:
        # Show label always; show number only if available
        if blur_var is None:
            blur_text = f"BlurVar -- (min {min_blur_var})"
        else:
            blur_text = f"BlurVar {blur_var:.0f} (min {min_blur_var})"

        put_text(
            display,
            blur_text,
            blur_pos,
            FONT_SMALL,
            (255, 255, 255),
        )
    else:
        # Old behavior: only show when we have a number
        if blur_var is not None:
            put_text(
                display,
                f"BlurVar {blur_var:.0f} (min {min_blur_var})",
                blur_pos,
                FONT_SMALL,
                (255, 255, 255),
            )

    # Data lines
    y = lines_y
    for line in lines:
        put_text(
            display,
            line,
            (blur_pos[0], y),
            FONT_SMALL,
            (255, 255, 255),
        )
        y += 26




def draw_quad(display, quad, mirror_ui):
    if quad is None:
        return
    q = mirror_quad_x(quad, display.shape[1]) if mirror_ui else quad
    cv2.polylines(display, [q.astype("int32")], True, (0, 255, 0), 2)


def draw_card_inset(display, card_bgr, title="CARD"):
    """
    Draws a zoomed-in card preview in the top-right corner.
    card_bgr should be a BGR image (like res["warped"]).
    """
    if card_bgr is None:
        return

    h, w = display.shape[:2]

    card_bgr = orient_for_inset(card_bgr, INSET_W, INSET_H)
    inset = cv2.resize(card_bgr, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)



    x1 = w - INSET_MARGIN
    x0 = x1 - INSET_W
    y0 = INSET_MARGIN 
    y1 = y0 + INSET_H

    if x0 < 0 or y1 > h:
        return

    pad = 6
    bx0, by0 = x0 - pad, y0 - pad
    bx1, by1 = x1 + pad, y1 + pad
    bx0 = max(0, bx0)
    by0 = max(0, by0)
    bx1 = min(w, bx1)
    by1 = min(h, by1)

    overlay = display.copy()
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
    display[:] = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)

    display[y0:y1, x0:x1] = inset
    cv2.rectangle(display, (x0, y0), (x1, y1), (255, 255, 255), 2)

 



# -------------------------
# Scanner decision helpers
# -------------------------
def choose_best_from_samples(samples):
    """
    samples: list of dicts with keys: id, final
    Choose card by highest average final score.
    """
    if not samples:
        return None

    sums = {}
    counts = {}
    for s in samples:
        cid = s["id"]
        sums[cid] = sums.get(cid, 0.0) + float(s["final"])
        counts[cid] = counts.get(cid, 0) + 1

    best_id = None
    best_avg = -1e9
    for cid in sums:
        avg = sums[cid] / max(1, counts[cid])
        if avg > best_avg:
            best_avg = avg
            best_id = cid

    return {"id": best_id, "avg_final": best_avg, "count": counts.get(best_id, 0)}


def is_ready_to_scan(res):
    return (
        res["has_quad"]
        and (res["blur_var"] is not None and res["blur_var"] > CFG.min_blur_var)
        and res["stable"] >= res["stable_required"]
    )


# -------------------------
# State init + reset helpers
# -------------------------
def init_core_state():
    return {
        "last_quad": None,
        "stable": 0,
        "last_id": 0.0,
        "lines": [],
    }


def init_scan_state():
    return {
        "locked": False,
        "locked_id": None,
        "locked_lines": [],
        "locked_warp": None,
        "scan_active": False,
        "scan_start": 0.0,
        "samples": [],
    }


def init_button_state():
    return {
        "green_active": False,
        "red_active": False,
        "mouse_x": 0,
        "mouse_y": 0,
        "clicked": False,
    }


def reset_scan(scan, buttons):
    scan["locked"] = False
    scan["locked_id"] = None
    scan["locked_lines"] = []
    scan["locked_warp"] = None
    scan["scan_active"] = False
    scan["scan_start"] = 0.0
    scan["samples"] = []

    buttons["red_active"] = False
    buttons["green_active"] = False


# -------------------------
# Window + IO helpers
# -------------------------
def install_mouse_handler(buttons):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            buttons["mouse_x"] = x
            buttons["mouse_y"] = y
            buttons["clicked"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)


def read_frame(cap):
    # Keep your "double grab" behavior
    cap.grab()
    cap.grab()
    ok, frame = cap.read()
    return ok, frame


def consume_click(buttons):
    """
    Consume click every frame to avoid queued clicks / "stuck" behavior.
    """
    click_x = buttons["mouse_x"]
    click_y = buttons["mouse_y"]
    clicked = buttons["clicked"]
    buttons["clicked"] = False
    return clicked, click_x, click_y


def apply_ui_mirror(frame_bgr, mirror_ui):
    return cv2.flip(frame_bgr, 1) if mirror_ui else frame_bgr.copy()


def orient_for_inset(card_bgr, target_w=INSET_W, target_h=INSET_H):
    """
    Choose 0° or 90° rotation that best matches the inset aspect ratio.
    Fixes sideways preview without touching scanner/warp logic.
    """
    if card_bgr is None:
        return None

    h, w = card_bgr.shape[:2]
    if h == 0 or w == 0:
        return card_bgr

    target_aspect = target_w / float(target_h)

    # aspect if unrotated
    aspect0 = w / float(h)

    # aspect if rotated 90 degrees (w/h swaps)
    aspect90 = h / float(w)

    if abs(aspect90 - target_aspect) < abs(aspect0 - target_aspect):
        # Rotate 90° clockwise (change to CCW if you prefer)
        return cv2.rotate(card_bgr, cv2.ROTATE_90_CLOCKWISE)

    return card_bgr


# -------------------------
# Locked-mode helpers
# -------------------------
def button_positions(display):
    h, w = display.shape[:2]
    red_cx = w - BTN_MARGIN - BTN_RADIUS
    red_cy = h - BTN_MARGIN - BTN_RADIUS
    green_cx = red_cx - (BTN_RADIUS * 2) - BTN_SPACING
    green_cy = red_cy
    return (green_cx, green_cy), (red_cx, red_cy)

def hud_bottom_left_layout(display, num_lines, line_height=26, margin=12):
    """
    Computes bottom-left aligned positions for HUD text.
    Returns:
        blur_pos: (x, y)
        lines_start_y: y
    """
    h, _ = display.shape[:2]

    total_height = line_height * (num_lines + 1)
    start_y = h - margin - total_height

    blur_pos = (margin, start_y)
    lines_start_y = start_y + line_height

    return blur_pos, lines_start_y


def handle_locked_click(scan, buttons, clicked, click_x, click_y, green_pos, red_pos):
    if not clicked:
        return

    mx, my = click_x, click_y
    green_cx, green_cy = green_pos
    red_cx, red_cy = red_pos

    if point_in_circle(mx, my, green_cx, green_cy, BTN_RADIUS):
        buttons["green_active"] = not buttons["green_active"]

    if point_in_circle(mx, my, red_cx, red_cy, BTN_RADIUS):
        reset_scan(scan, buttons)


def draw_locked_ui(display, scan, buttons, res, clicked, click_x, click_y):
    put_text(
        display,
        "LOCKED",
        (10, 30),
        FONT_MED,
        (0, 255, 0),
    )

    green_pos, red_pos = button_positions(display)

    handle_locked_click(scan, buttons, clicked, click_x, click_y, green_pos, red_pos)

    green_col = BTN_GREEN_ACTIVE if buttons["green_active"] else BTN_GREEN
    red_col = BTN_RED_ACTIVE if buttons["red_active"] else BTN_RED

    draw_circle_button(display, green_pos[0], green_pos[1], BTN_RADIUS, green_col)
    draw_circle_button(display, red_pos[0], red_pos[1], BTN_RADIUS, red_col)

    draw_hud(
        display=display,
        blur_var=res["blur_var"],
        min_blur_var=CFG.min_blur_var,
        lines=scan["locked_lines"],
        always_show_blur_label=True,   # ✅ keep label, number can disappear/reappear
    )

    


def pick_inset_for_display(scan, res, cards):
    """
    Decide which image goes in the inset + what the title is.
    """
    if scan["locked"]:
        inset_img = scan["locked_warp"] if scan["locked_warp"] is not None else res.get("warped")
        card = cards.get(scan["locked_id"], {})
        name = card.get("name", "Unknown Card")
        number = card.get("number", "")
        inset_title = f"{name} #{number}".strip()
        return inset_img, inset_title

    return res.get("warped"), "CARD"


# -------------------------
# Scanning-mode helpers
# -------------------------
def update_scan_activity(scan, ready_to_scan, now):
    if ready_to_scan and not scan["scan_active"]:
        scan["scan_active"] = True
        scan["scan_start"] = now
        scan["samples"] = []

    if not ready_to_scan:
        scan["scan_active"] = False
        scan["scan_start"] = 0.0
        scan["samples"] = []


def collect_sample_if_valid(scan, res):
    if res.get("attempted") and res.get("accepted") and res.get("best_id") is not None:
        scan["samples"].append({"id": res["best_id"], "final": res["best_final"]})


def draw_scanning_ui(display, now, scan, res):
    if not scan["scan_active"]:
        return

    elapsed = now - scan["scan_start"]
    remaining = max(0.0, SCAN_SECONDS - elapsed)

    put_text(
        display,
        f"Scanning... {remaining:.1f}s",
        (10, 30),
        FONT_MED,
        (0, 255, 255),
    )



def maybe_lock_from_samples(scan, res, cards, now):
    if not scan["scan_active"]:
        return

    elapsed = now - scan["scan_start"]
    if elapsed < SCAN_SECONDS:
        return

    winner = choose_best_from_samples(scan["samples"])
    if winner is not None:
        best_id = winner["id"]
        card = cards.get(best_id, {})

        scan["locked"] = True
        scan["locked_id"] = best_id
        scan["locked_lines"] = [
            f"{card.get('name','?')} #{card.get('number','')}",
            f"LOCK avgF={winner['avg_final']:.3f} (samples={winner['count']})",
        ]
        scan["locked_warp"] = res.get("warped")

    scan["scan_active"] = False
    scan["scan_start"] = 0.0
    scan["samples"] = []


def draw_attempting_text(display, res):
    if res.get("attempted"):
        put_text(
            display,
            "Attempting recognition...",
            (10, 115),
            FONT_SMALL,
            (0, 255, 0),
        )



# -------------------------
# Main loop
# -------------------------
def run_app_loop(cap, embedder, cards, ref_embs, ref_ids, process_frame_fn):
    mirror_ui = MIRROR_UI_DEFAULT

    state = init_core_state()
    scan = init_scan_state()
    buttons = init_button_state()

    install_mouse_handler(buttons)

    print("ESC quit | M mirror | Click RED to reset/unlock")

    while True:
        ok, frame = read_frame(cap)
        if not ok:
            break

        now = time.time()

        # Always process frame (even locked)
        res = process_frame_fn(frame, embedder, cards, ref_embs, ref_ids, state, CFG)

        # Display frame (mirror is UI-only)
        display = apply_ui_mirror(res["frame"], mirror_ui)

        # Consume click every frame
        clicked, click_x, click_y = consume_click(buttons)

        # Quad overlay
        if res["has_quad"]:
            draw_quad(display, res["quad"], mirror_ui)

        # Inset
        inset_img, inset_title = pick_inset_for_display(scan, res, cards)
        draw_card_inset(display, inset_img, inset_title)

        # =========================
        # LOCKED mode
        # =========================
        if scan["locked"]:
            draw_locked_ui(display, scan, buttons, res, clicked, click_x, click_y)

            cv2.imshow(WINDOW_NAME, display)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("m"):
                mirror_ui = not mirror_ui
            elif k == 27:
                break
            continue

        # =========================
        # Not locked: scanning logic
        # =========================
        ready_to_scan = is_ready_to_scan(res)

        update_scan_activity(scan, ready_to_scan, now)

        if scan["scan_active"]:
            collect_sample_if_valid(scan, res)
            draw_scanning_ui(display, now, scan, res)
            maybe_lock_from_samples(scan, res, cards, now)

        draw_attempting_text(display, res)

        draw_hud(
            display=display,
            blur_var=res["blur_var"],
            min_blur_var=CFG.min_blur_var,
            lines=res["lines"],
        )

        cv2.imshow(WINDOW_NAME, display)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("m"):
            mirror_ui = not mirror_ui
        elif k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
