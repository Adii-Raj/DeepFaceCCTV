"""
core/detector.py
────────────────
YuNet face detector wrapper.

Single responsibility: given a BGR frame, return face rows (N x 15 float32).

Face row columns (YuNet format):
  0-3   : x, y, w, h  (bounding box)
  4-5   : left eye x, y
  6-7   : right eye x, y
  8-9   : nose tip x, y
  10-11 : left mouth corner x, y
  12-13 : right mouth corner x, y
  14    : detection confidence score

To swap detector (e.g. YOLOv8) in the future:
  - Rewrite this file only
  - Keep detect() returning same (N x 15) format so pipeline.py is untouched
"""

import os
import socket
import urllib.request

import cv2
import numpy as np

# ── Default model info ────────────────────────────────────────────────────────

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
DOWNLOAD_TIMEOUT = 30


# ── Model download ────────────────────────────────────────────────────────────

def ensure_model(url: str, path: str) -> str:
    """Download model file if not already present. Returns path."""
    if os.path.exists(path):
        return path
    print(f"[detector] Downloading {os.path.basename(path)} ...")
    try:
        def _reporthook(count, block, total):
            if total > 0 and count % 50 == 0:
                pct = min(100, int(count * block * 100 / total))
                print(f"\r  {pct}%", end="", flush=True)

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)
        try:
            urllib.request.urlretrieve(url, path, _reporthook)
        finally:
            socket.setdefaulttimeout(old_timeout)
        print(f"\n[detector] Saved -> {path}")
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"[detector] Failed to download {url}: {e}") from e
    return path


# ── NMS / deduplication ───────────────────────────────────────────────────────

def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union for two boxes in [x, y, w, h] format."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def deduplicate_faces(faces: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """
    Non-maximum suppression over YuNet detections.
    Returns list of kept row indices sorted by confidence descending.
    """
    if len(faces) == 0:
        return []
    order = np.argsort(-faces[:, 14])
    kept, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(int(i))
        for j in order:
            if j != i and j not in suppressed:
                if _iou_xywh(faces[i, :4], faces[j, :4]) > iou_thr:
                    suppressed.add(int(j))
    return kept


# ── YuNet detector ────────────────────────────────────────────────────────────

class YuNetDetector:
    """
    Wraps OpenCV's FaceDetectorYN (YuNet).

    Usage:
        detector = YuNetDetector.from_path("models/face_detection_yunet_2023mar.onnx")
        faces = detector.detect(bgr_frame)   # np.ndarray shape (N, 15)
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.55,
        nms_threshold: float   = 0.30,
        top_k: int             = 5002,
    ):
        self._det = cv2.FaceDetectorYN.create(
            model_path, "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._current_hw = (0, 0)

    @classmethod
    def from_path(
        cls,
        model_path: str,
        score_threshold: float = 0.55,
    ) -> "YuNetDetector":
        """Load detector from a local path. Downloads if missing."""
        ensure_model(YUNET_MODEL_URL, model_path)
        return cls(model_path, score_threshold=score_threshold)

    def detect(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Detect faces in a BGR frame.

        Returns np.ndarray of shape (N, 15) float32.
        Returns empty (0, 15) array if no faces found.
        Automatically resizes input size on frame dimension change.
        """
        h, w = bgr_frame.shape[:2]
        if (h, w) != self._current_hw:
            self._det.setInputSize((w, h))
            self._current_hw = (h, w)
        _, faces = self._det.detect(bgr_frame)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def detect_and_deduplicate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Detect + NMS in one call.
        Returns deduplicated face rows as np.ndarray (M, 15).
        """
        raw = self.detect(bgr_frame)
        kept = deduplicate_faces(raw)
        return raw[kept] if kept else np.empty((0, 15), np.float32)