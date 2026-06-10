"""
core/quality.py
───────────────
Face quality filtering and vote resolution.

Single responsibility: decide whether a face crop is good enough to embed,
and resolve a rolling window of recognition votes into a stable display label.

Three pure concerns:
  1. Yaw estimation     — is the face too side-on to embed reliably?
  2. Quality gate       — is the crop large enough and sharp enough?
  3. Vote resolver      — given N recent votes, what label do we show?

No OpenCV drawing, no database, no detection here.
"""

from collections import Counter, deque

import cv2
import numpy as np

from core.tracker import POSE_SKIP, QUALITY_SKIP, SKIP_TOKENS, MIN_VOTES


# ── 1. Yaw estimator ──────────────────────────────────────────────────────────

def estimate_yaw(face_row: np.ndarray) -> float:
    """
    Estimate absolute yaw angle (degrees) from YuNet landmark positions.

    YuNet face_row landmark columns (0-indexed):
      4,5   = left eye x, y
      6,7   = right eye x, y
      8,9   = nose tip x, y
      10,11 = left mouth corner x, y
      12,13 = right mouth corner x, y

    Method:
      For a frontal face the nose sits roughly midway between the eyes
      (ratio ≈ 0.5). As the face yaws the nose shifts toward the near eye.
      We map the ratio deviation to degrees linearly:
        ratio 0.5 → 0° (frontal)
        ratio 0.0 or 1.0 → 45° (full side)

    Returns:
        Estimated yaw in degrees [0, 90+].
        Returns 999.0 for degenerate / invalid detections.
    """
    lx = face_row[4]   # left eye x
    rx = face_row[6]   # right eye x
    nx = face_row[8]   # nose tip x

    eye_dist = rx - lx
    if eye_dist < 4.0:
        return 999.0   # degenerate detection

    ratio = (nx - lx) / eye_dist      # frontal ≈ 0.5
    yaw   = abs(ratio - 0.5) * 90.0   # 0° frontal -> 45° at full side
    return yaw


# ── 2. Interocular distance gate ──────────────────────────────────────────────

def interocular_distance(face_row: np.ndarray) -> float:
    """
    Euclidean distance between left and right eye landmarks (pixels).

    Very small IED (< ~15px in display frame) indicates extreme pose or
    very far-away face — embedding will be noisy regardless of yaw angle.

    Returns:
        IED in pixels. Returns 0.0 for degenerate detections.
    """
    lx, ly = face_row[4], face_row[5]
    rx, ry = face_row[6], face_row[7]
    return float(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5)


# ── 3. Crop quality gate ──────────────────────────────────────────────────────

def face_quality_ok(
    crop:            np.ndarray,
    min_size:        int   = 25,
    blur_threshold:  float = 20.0,
) -> bool:
    """
    Returns True if the face crop passes size and sharpness checks.

    Args:
        crop           : BGR face crop
        min_size       : minimum crop width AND height in pixels
                         (default 25px — CCTV-optimised; lower for far cameras)
        blur_threshold : Laplacian variance threshold.
                         CCTV typically scores 10–40.
                         Lower to 10 for night / very dim cameras.

    Returns:
        True  → crop is usable for embedding
        False → crop is too small or too blurry; skip this frame
    """
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < min_size or h < min_size:
        return False
    grey        = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score  = cv2.Laplacian(grey, cv2.CV_64F).var()
    return blur_score >= blur_threshold


# ── 4. Vote resolver ──────────────────────────────────────────────────────────

def resolve_vote(vote_deque: deque) -> tuple[str, str]:
    """
    Resolve a rolling window of (name, status) votes into a stable label.

    Rules:
      - Skip tokens (POSE_SKIP, QUALITY_SKIP) are excluded from counting
        but they consume a slot so old good votes age out naturally.
      - If fewer than MIN_VOTES non-skip results exist, return ("?", "unsure")
        — not enough data yet to show a label.
      - Otherwise the name with the most votes wins.
        Ties broken by recency (the most recent occurrence wins).

    Args:
        vote_deque : deque of (name, status) tuples (maxlen = VOTE_WINDOW)

    Returns:
        (display_name, display_status)
    """
    real_votes = [
        (n, s) for n, s in vote_deque
        if n not in SKIP_TOKENS
    ]

    if len(real_votes) < MIN_VOTES:
        return "?", "unsure"

    name_counts = Counter(n for n, _ in real_votes)
    winner_name = name_counts.most_common(1)[0][0]

    # Status = most recent status for the winning name
    winner_status = next(
        s for n, s in reversed(list(vote_deque))
        if n == winner_name
    )
    return winner_name, winner_status


# ── 5. Combined quality check (used by pipeline) ──────────────────────────────

def passes_quality_gates(
    face_row:        np.ndarray,
    crop:            np.ndarray,
    max_yaw:         float = 45.0,
    min_ied:         float = 15.0,
    min_face_size:   int   = 25,
    blur_threshold:  float = 20.0,
) -> tuple[bool, str]:
    """
    Run all quality checks in order. Returns (passes, skip_reason).

    Checks (in order of cheapest to most expensive):
      1. Yaw angle (landmark math only — very cheap)
      2. Interocular distance (landmark math only — very cheap)
      3. Crop size + blur (requires crop extraction — slightly more expensive)

    Args:
        face_row       : YuNet face row (15 values)
        crop           : extracted BGR face crop
        max_yaw        : degrees beyond which pose is rejected
        min_ied        : minimum interocular distance in pixels
        min_face_size  : minimum crop dimension in pixels
        blur_threshold : Laplacian variance floor

    Returns:
        (True,  "")            → all checks passed, safe to embed
        (False, skip_reason)   → skip this frame, skip_reason for logging
    """
    yaw = estimate_yaw(face_row)
    if yaw > max_yaw:
        return False, POSE_SKIP

    ied = interocular_distance(face_row)
    if ied < min_ied:
        return False, QUALITY_SKIP

    if not face_quality_ok(crop, min_size=min_face_size,
                           blur_threshold=blur_threshold):
        return False, QUALITY_SKIP

    return True, ""