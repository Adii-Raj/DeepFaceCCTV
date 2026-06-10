"""
core/tracker.py
───────────────
Face tracking across video frames using IOU-based matching.

Single responsibility: given detected face rows for each frame,
maintain persistent Track objects across frames using bounding box IOU.

No detection, no recognition, no drawing — tracking only.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── Vote sentinels ────────────────────────────────────────────────────────────
# These are internal tokens that get pushed into the vote deque when a frame
# is skipped due to pose or quality. They are never shown as identity labels.

POSE_SKIP    = "__pose_skip__"
QUALITY_SKIP = "__quality_skip__"
SKIP_TOKENS  = {POSE_SKIP, QUALITY_SKIP}

# Minimum non-skip votes before a label is shown (avoids premature flips)
MIN_VOTES   = 1
VOTE_WINDOW = 5   # rolling deque maxlen


# ── Track dataclass ───────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    Represents a single tracked face across frames.

    Fields updated by tracker:
      id, box, face_row, age

    Fields updated by pipeline (after recognition):
      name, score, status, frames_no_recog, embed_pending,
      votes, display_name, display_status
    """
    id:              int
    box:             np.ndarray   # [x, y, w, h] in detection-scale coords
    face_row:        np.ndarray   # full YuNet row (15 values)

    # Recognition results
    name:            str   = "Unknown"
    score:           float = 0.0
    status:          str   = "unknown"   # 'known' | 'unsure' | 'unknown'

    # Recognition scheduling
    frames_no_recog: int   = 999         # counts frames since last embed
    embed_pending:   bool  = False       # True while embed job is in flight

    # Rolling vote window (filled by pipeline after each embed)
    votes: deque = field(
        default_factory=lambda: deque(maxlen=VOTE_WINDOW)
    )

    # Resolved display values (updated after each vote)
    display_name:   str = "?"
    display_status: str = "unsure"

    # Tracker internal
    age: int = 0   # frames since last matched detection (0 = matched this frame)


# ── IOU helper ────────────────────────────────────────────────────────────────

def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """IOU between two boxes in [x, y, w, h] format."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


# ── Face tracker ──────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Greedy IOU-based multi-face tracker.

    Each call to update() accepts the current frame's detected face rows
    and returns a list of Track objects (one per detection).

    Tracks that haven't matched a detection for more than max_age frames
    are automatically removed.

    Usage:
        tracker = FaceTracker()
        tracks = tracker.update(face_rows_np)  # call every frame
    """

    def __init__(self, max_age: int = 10, iou_thr: float = 0.35):
        """
        Args:
            max_age : frames a track survives without a matching detection
            iou_thr : minimum IOU to consider two boxes the same face
        """
        self._tracks: dict[int, Track] = {}
        self._next_id = 0
        self.max_age  = max_age
        self.iou_thr  = iou_thr

    def update(self, faces: np.ndarray) -> list[Track]:
        """
        Match new detections to existing tracks.

        Args:
            faces: np.ndarray shape (N, 15) from YuNetDetector.detect()
                   Pass empty (0, 15) array if no faces detected.

        Returns:
            list of Track objects, one per detection in `faces`.
            New detections get a fresh Track with a new id.
            Matched detections update the existing Track in-place.
        """
        # Age all tracks; prune stale ones
        for t in self._tracks.values():
            t.age += 1
        self._tracks = {
            k: v for k, v in self._tracks.items()
            if v.age <= self.max_age
        }

        if len(faces) == 0:
            return []

        n    = len(faces)
        tids = list(self._tracks.keys())

        # Build all (iou, det_idx, track_id) pairs above threshold
        pairs = []
        for di in range(n):
            for tid in tids:
                iou = _iou_xywh(faces[di, :4], self._tracks[tid].box)
                if iou >= self.iou_thr:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)   # highest IOU first

        result:  list       = [None] * n
        used_d:  set[int]   = set()
        used_t:  set[int]   = set()

        # Greedy matching: best IOU pairs get priority
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            t           = self._tracks[tid]
            t.box       = faces[di, :4].copy()
            t.face_row  = faces[di].copy()
            t.age       = 0
            t.frames_no_recog += 1
            used_d.add(di)
            used_t.add(tid)
            result[di]  = t

        # Unmatched detections -> new tracks
        for di in range(n):
            if result[di] is None:
                tid = self._next_id
                self._next_id += 1
                t = Track(
                    id       = tid,
                    box      = faces[di, :4].copy(),
                    face_row = faces[di].copy(),
                )
                self._tracks[tid] = t
                result[di] = t

        return result

    def reset(self):
        """Clear all tracks. Call when stream reconnects."""
        self._tracks.clear()

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)