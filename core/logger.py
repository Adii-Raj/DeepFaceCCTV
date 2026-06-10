"""
core/logger.py
──────────────
Detection logging — CSV and face crop images.

Single responsibility: persist detection events to disk.

Two outputs:
  1. detections.csv  — one row per confirmed detection (with cooldown)
  2. crops/          — face crop images saved alongside CSV rows

No detection, no recognition, no drawing here.
If you want to change log format or add a database log — only this file changes.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── DetectionLogger ───────────────────────────────────────────────────────────

class DetectionLogger:
    """
    Logs confirmed face detections to CSV and saves face crop images.

    Cooldown prevents the same person being logged repeatedly every frame —
    by default a person is only logged once every 3 seconds.

    CSV columns:
        timestamp     : wall-clock time (YYYY-MM-DD HH:MM:SS)
        video_time_s  : position in video stream (seconds)
        name          : identity label
        confidence    : cosine similarity score
        status        : 'known' | 'unsure'
        crop_filename : filename of saved face crop image (or empty string)

    Usage:
        logger = DetectionLogger(csv_path="data/detections.csv",
                                 crops_dir="dashboard/static/crops")
        logger.log(name, score, video_time, status, crop_bgr)
        logger.close()
    """

    CSV_COLUMNS = [
        "timestamp",
        "video_time_s",
        "name",
        "confidence",
        "status",
        "crop_filename",
    ]

    def __init__(
        self,
        csv_path:   str = "data/detections.csv",
        crops_dir:  str = "dashboard/static/crops",
        cooldown:   float = 3.0,
        save_crops: bool  = True,
    ):
        """
        Args:
            csv_path   : path to output CSV file
            crops_dir  : directory where crop images are saved
            cooldown   : minimum seconds between log entries for the same person
            save_crops : whether to save face crop images to disk
        """
        self._csv_path   = csv_path
        self._crops_dir  = crops_dir
        self._cooldown   = cooldown
        self._save_crops = save_crops

        # Per-person last-logged timestamp
        self._last_seen: dict[str, float] = {}

        # Ensure directories exist
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if save_crops:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)

        # Open CSV — append mode so logs survive restarts
        file_exists = os.path.isfile(csv_path)
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(self.CSV_COLUMNS)
            self._file.flush()

        print(f"[logger] CSV -> {csv_path}")
        if save_crops:
            print(f"[logger] Crops -> {crops_dir}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(
        self,
        name:        str,
        score:       float,
        video_time:  float,
        status:      str,
        crop:        np.ndarray | None = None,
    ) -> bool:
        """
        Log a detection event if cooldown has passed.

        Args:
            name       : identity label (skip if "?" or "Unknown")
            score      : cosine similarity score
            video_time : stream position in seconds
            status     : 'known' | 'unsure' (skip 'unknown' / 'skip')
            crop       : optional BGR face crop to save as image

        Returns:
            True if the entry was logged, False if skipped (cooldown / filtered)
        """
        # Only log confirmed or unsure known identities
        if status not in ("known", "unsure"):
            return False
        if name in ("?", "Unknown", ""):
            return False

        now = time.time()
        if now - self._last_seen.get(name, 0.0) < self._cooldown:
            return False

        self._last_seen[name] = now

        # Save crop image
        crop_filename = ""
        if self._save_crops and crop is not None and crop.size > 0:
            crop_filename = self._save_crop(name, crop)

        # Write CSV row
        self._writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{video_time:.2f}",
            name,
            f"{score:.4f}",
            status,
            crop_filename,
        ])
        self._file.flush()
        return True

    def _save_crop(self, name: str, crop: np.ndarray) -> str:
        """
        Save crop image to crops_dir.
        Filename: <name>_<timestamp_ms>.jpg
        Returns just the filename (not full path) for the CSV.
        """
        try:
            ts       = int(time.time() * 1000)
            safe     = "".join(c if c.isalnum() or c in "_-" else "_"
                               for c in name)
            filename = f"{safe}_{ts}.jpg"
            path     = os.path.join(self._crops_dir, filename)
            cv2.imwrite(path, crop)
            return filename
        except Exception as e:
            print(f"[logger] Crop save failed for '{name}': {e}")
            return ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Flush and close the CSV file."""
        try:
            self._file.flush()
            self._file.close()
            print(f"[logger] Closed — {self._csv_path}")
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def recent_detections(self, n: int = 20) -> list[dict]:
        """
        Read the last N rows from the CSV and return as list of dicts.
        Used by the Flask dashboard for live preview without a DB query.
        Returns newest first.
        """
        try:
            with open(self._csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            return list(reversed(rows[-n:]))
        except Exception:
            return []