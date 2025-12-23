# cv_card.py
import cv2
import numpy as np

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
    pts: (4,2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def find_card_quad(frame_bgr: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray | None:
    """
    Detect the largest convex 4-point contour (a card-like quad).

    Returns:
        quad: (4,2) float32 array of corner points in image coordinates, or None.
    """
    h, w = frame_bgr.shape[:2]
    img_area = float(h * w)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:15]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype("float32")
            return quad

    return None

def warp_quad(frame_bgr: np.ndarray, quad: np.ndarray, out_w: int = 512, out_h: int = 712) -> np.ndarray:
    """
    Perspective-warp the detected quad to a fixed-size rectangle.
    """
    rect = order_points(quad)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h))
    return warped
