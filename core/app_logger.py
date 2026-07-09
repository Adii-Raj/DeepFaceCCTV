"""
core/app_logger.py
──────────────────
Application logging — file rotation, structured format

Single responsibility: log application events (NOT detection events).
Detection events go to DetectionLogger (core/logger.py).

Usage:
    from core.app_logger import get_logger

    logger = get_logger("pipeline")
    logger.info("Frame processed")
    logger.error("Camera failed", exc_info=True)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 5 MB per file, keep 3 backups = max ~20 MB total
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


# ── Logger Factory ────────────────────────────────────────────────────────────


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get or create a logger with rotating file + console output.

    Args:
        name:   module name (e.g., "pipeline", "dashboard", "camera")
        level:  minimum level to capture (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Don't bubble to root logger

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    # 1. Rotating file handler (all levels)
    file_path = LOG_DIR / f"{name}.log"
    file_handler = RotatingFileHandler(
        file_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file

    # 2. Console handler (INFO and above only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 3. Error-only file (all ERROR+ across all modules)
    error_handler = RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=MAX_BYTES, backupCount=5, encoding="utf-8"
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


# ── Convenience: Root Error Catcher ───────────────────────────────────────────


def setup_global_exception_logging() -> None:
    """Catch uncaught exceptions and log them."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = get_logger("uncaught")
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
