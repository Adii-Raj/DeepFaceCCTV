"""
core/__init__.py
────────────────
Auto-initialize logging on import
"""

from core.app_logger import setup_global_exception_logging

setup_global_exception_logging()
