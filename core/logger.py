"""
core/logger.py
──────────────
Detection logging — SQLite only

Single responsibility: persist detection events to SQLite database.

one output:
  1. detections.db  — one row per confirmed detection (with cooldown)

No detection, no recognition, no drawing here.
If you want to change log format or add a database log — only this file changes.
"""

import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── DetectionLogger ───────────────────────────────────────────────────────────


class DetectionLogger:
    """
    Logs confirmed face detections to SQLite only

    Cooldown prevents the same person being logged repeatedly every frame —
    by default a person is only logged once every 3 seconds.

    SQLite columns:
        timestamp     : wall-clock time (YYYY-MM-DD HH:MM:SS)
        video_time_s  : position in video stream (seconds)
        name          : identity label
        confidence    : cosine similarity score
        status        : 'known' | 'unsure'

    Usage:
        logger = DetectionLogger(db_path="data/detections.db")
        logger.log(name, score, video_time, status)
        logger.close()
    """

    def __init__(
        self,
        db_path: str = "data/detections.db",
        cooldown: float = 3.0,
    ):
        """
        Args:
            db_path    : path to SQLite database file
            cooldown   : minimum seconds between log entries for the same person
        """
        self._db_path = db_path
        self._cooldown = cooldown

        # Per-person last-logged timestamp
        self._last_seen: dict[str, float] = {}

        # Ensure directories exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._init_table()

        print(f"[logger] SQLite -> {db_path}")

    # ── Database setup ────────────────────────────────────────────────────────

    def _init_table(self):
        """Create detections table if it doesn't exist."""
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                video_time_s REAL,
                name TEXT,
                confidence REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Indexes for fast queries
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)"
        )
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON detections(name)")
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON detections(created_at)"
        )
        self._conn.commit()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(
        self,
        name: str,
        score: float,
        video_time: float,
        status: str,
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

        # Write to SQLite
        self._cursor.execute(
            """
            INSERT INTO detections (timestamp, video_time_s, name, confidence, status)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(video_time, 2),
                name,
                round(score, 4),
                status,
            ),
        )
        self._conn.commit()
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Close the SQLite connection."""
        try:
            self._conn.close()
            print(f"[logger] Closed — {self._db_path}")
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def recent_detections(self, n: int = 20) -> list[dict]:
        """
        Read the last N rows from the database and return as list of dicts.
        Used by the Flask dashboard for live preview.
        Returns newest first.
        """
        try:
            self._cursor.execute(
                """
                SELECT timestamp, video_time_s, name, confidence, status
                FROM detections
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (n,),
            )
            rows = self._cursor.fetchall()

            columns = ["timestamp", "video_time_s", "name", "confidence", "status"]
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            return []

    def get_all_detections(self) -> list[dict]:
        """Return all detections as list of dicts."""
        try:
            self._cursor.execute("""
                SELECT timestamp, video_time_s, name, confidence, status
                FROM detections
                ORDER BY created_at DESC
            """)
            rows = self._cursor.fetchall()
            columns = ["timestamp", "video_time_s", "name", "confidence", "status"]
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            return []

    def get_detection_count(self) -> int:
        """Return total number of detection entries."""
        try:
            self._cursor.execute("SELECT COUNT(*) FROM detections")
            return self._cursor.fetchone()[0]
        except Exception:
            return 0

    def get_unique_names(self) -> list[str]:
        """Return list of unique detected names."""
        try:
            self._cursor.execute("SELECT DISTINCT name FROM detections ORDER BY name")
            return [row[0] for row in self._cursor.fetchall()]
        except Exception:
            return []
