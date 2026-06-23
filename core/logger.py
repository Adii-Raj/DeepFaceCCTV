"""
core/logger.py
──────────────
Detection logging — CSV only

Single responsibility: persist detection events to disk.

one outputs:
  1. detections.csv  — one row per confirmed detection (with cooldown)

No detection, no recognition, no drawing here.
If you want to change log format or add a database log — only this file changes.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ── DetectionLogger ───────────────────────────────────────────────────────────

class DetectionLogger:
    """
    Logs confirmed face detections to CSV only

    Cooldown prevents the same person being logged repeatedly every frame —
    by default a person is only logged once every 3 seconds.

    CSV columns:
        timestamp     : wall-clock time (YYYY-MM-DD HH:MM:SS)
        video_time_s  : position in video stream (seconds)
        name          : identity label
        confidence    : cosine similarity score
        status        : 'known' | 'unsure'

    Usage:
        logger = DetectionLogger(csv_path="data/detections.csv")
        logger.log(name, score, video_time, status)
        logger.close()
    """

    CSV_COLUMNS = [
        "timestamp",
        "video_time_s",
        "name",
        "confidence",
        "status",
    ]

    def __init__(
        self,
        csv_path:   str = "data/detections.csv",
        cooldown:   float = 3.0,
    ):
        """
        Args:
            csv_path   : path to output CSV file
            cooldown   : minimum seconds between log entries for the same person
        """
        self._csv_path   = csv_path
        self._cooldown   = cooldown

        # Per-person last-logged timestamp
        self._last_seen: dict[str, float] = {}

        # Ensure directories exist
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)


        # Open CSV — append mode so logs survive restarts
        file_exists = os.path.isfile(csv_path)
        self._file = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(self.CSV_COLUMNS)
            self._file.flush()

        print(f"[logger] CSV -> {csv_path}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(
        self,
        name:        str,
        score:       float,
        video_time:  float,
        status:      str,
    ) -> bool:
        """
        Log a detection event if cooldown has passed.

        Args:
            name       : identity label (skip if "?" or "Unknown")
            score      : cosine similarity score
            video_time : stream position in seconds
            status     : 'known' | 'unsure' (skip 'unknown' / 'skip')

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

    
        # Write CSV row
        self._writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{video_time:.2f}",
            name,
            f"{score:.4f}",
            status,
        ])
        self._file.flush()
        return True
    
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