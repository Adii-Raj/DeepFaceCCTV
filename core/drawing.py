"""
core/drawing.py
───────────────
All OpenCV drawing and frame annotation.

Single responsibility: take a frame and detection data, draw on it, return it.

No detection, no recognition, no database, no file I/O here.
If you want to change how boxes/labels look — this is the only file to touch.
"""

import cv2
import numpy as np

# ── Colours (BGR) ─────────────────────────────────────────────────────────────

BOX_KNOWN    = (0,   220,   0)     # green   — confirmed identity
BOX_UNSURE   = (0,   200, 255)     # amber   — unsure / borderline
BOX_UNKNOWN  = (0,    60, 220)     # red     — no match
BOX_SKIP     = (120, 120, 120)     # grey    — pose or quality skip
TEXT_COLOR   = (255, 255, 255)     # white

FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ── Label helpers ─────────────────────────────────────────────────────────────

def _status_color(status: str) -> tuple[int, int, int]:
    if status == "known":
        return BOX_KNOWN
    if status == "unsure":
        return BOX_UNSURE
    if status == "skip":
        return BOX_SKIP
    return BOX_UNKNOWN


def _status_label(display_name: str, score: float, status: str) -> str:
    if status == "known":
        return f"{display_name}  {score:.2f}"
    if status == "unsure":
        if display_name and display_name != "?":
            return f"?{display_name}  {score:.2f}"
        return f"?  {score:.2f}"
    if status == "skip":
        return "skip"
    return f"Unknown  {score:.2f}"


# ── Face label (box + name badge) ─────────────────────────────────────────────

def draw_face_label(
    frame:        np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    display_name: str,
    score:        float,
    status:       str,
) -> None:
    """
    Draw bounding box + name badge on frame in-place.

    Args:
        frame        : BGR frame to draw on (modified in-place)
        x1, y1       : top-left corner of face box
        x2, y2       : bottom-right corner of face box
        display_name : resolved identity name (or "?" / "Unknown")
        score        : cosine similarity score (0.0 – 1.0)
        status       : 'known' | 'unsure' | 'unknown' | 'skip'
    """
    color = _status_color(status)
    label = _status_label(display_name, score, status)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name badge (filled rectangle behind text)
    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
    badge_y1 = max(y1 - th - baseline - 6, 0)
    badge_y2 = max(y1, th + baseline + 6)
    cv2.rectangle(
        frame,
        (x1, badge_y1),
        (x1 + tw + 6, badge_y2),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + 3, badge_y2 - baseline - 2),
        FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA,
    )


# ── HUD overlay (top of frame) ────────────────────────────────────────────────

def draw_hud(
    frame:             np.ndarray,
    fps:               float,
    frame_idx:         int,
    recogniser_label:  str,
    vote_window:       int,
    max_yaw:           float,
    threshold_accept:  float,
    threshold_reject:  float,
    blur_threshold:    float,
    db_size:           int,
) -> None:
    """
    Draw two lines of diagnostic info at the top of the frame.

    Line 1: FPS, frame index, recogniser type, vote window, max yaw
    Line 2: thresholds and DB size
    """
    line1 = (
        f"FPS:{fps:.1f}  Frame:{frame_idx}  [{recogniser_label}]  "
        f"Vote:{vote_window}fr  MaxYaw:{max_yaw:.0f}°"
    )
    line2 = (
        f"Accept:{threshold_accept}  "
        f"Reject:{threshold_reject}  "
        f"Blur≥{blur_threshold}  "
        f"DB:{db_size}"
    )
    cv2.putText(frame, line1,
                (10, 28), FONT, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, line2,
                (10, 54), FONT, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


# ── Stream overlays ───────────────────────────────────────────────────────────

def draw_loading_overlay(frame: np.ndarray, msg: str = "Loading model…") -> None:
    """
    Semi-transparent dark overlay with a centred loading message.
    Drawn in-place.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),
                  (frame.shape[1], frame.shape[0]),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
    cx = (frame.shape[1] - tw) // 2
    cy = (frame.shape[0] + th) // 2
    cv2.putText(frame, msg, (cx, cy), FONT, 0.9, (200, 200, 200), 2, cv2.LINE_AA)


def draw_reconnect_overlay(frame: np.ndarray, elapsed_seconds: float) -> None:
    """
    Semi-transparent dark overlay with reconnecting message and timer.
    Drawn in-place.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),
                  (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    msg = f"RECONNECTING … {elapsed_seconds:.0f}s"
    (tw, th), _ = cv2.getTextSize(msg, FONT, 1.1, 2)
    cx = (frame.shape[1] - tw) // 2
    cy = (frame.shape[0] + th) // 2
    cv2.putText(frame, msg, (cx, cy), FONT, 1.1, (0, 180, 255), 2, cv2.LINE_AA)


def draw_headless_status(
    frame:       np.ndarray,
    frame_idx:   int,
    track_count: int,
) -> None:
    """
    Minimal overlay drawn when running in headless mode but a frame
    is being saved as a snapshot (e.g. for the Flask dashboard).
    """
    msg = f"Frame:{frame_idx}  Tracks:{track_count}"
    cv2.putText(frame, msg, (10, 28), FONT, 0.65,
                (200, 200, 200), 2, cv2.LINE_AA)