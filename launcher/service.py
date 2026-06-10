"""
launcher/service.py
-------------------
Subprocess management for the Survil pipeline.
Single responsibility: start, stop, monitor pipeline.py as a subprocess,
and read/write config.json.
No Tkinter, no UI logic.

Used by launcher/launcher.py exclusively.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG: dict = {
    "rtsp_url": "rtsp://192.168.1.10/stream1",
    "db_path": "data/face_db",
    "collection_name": "face_gallery",
    "model_dir": "models/buffalo_l",
    "detector_model": "models/face_detection_yunet_2023mar.onnx",
    "detections_csv": "data/detections.csv",
    "crops_dir": "dashboard/static/crops",
    "headless": True,
    "gallery_refresh_sec": 60,
    "score_threshold": 0.25,
    "nms_threshold": 0.3,
    "confidence_threshold": 0.7,
    "flask_port": 5000,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path = _CONFIG_PATH) -> dict:
    """
    Load config.json. Missing keys are filled from DEFAULT_CONFIG.
    Creates config.json with defaults if it doesn't exist.
    """
    if not path.exists():
        logger.info("config.json not found — creating with defaults.")
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG.copy()

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load config.json: %s — using defaults.", e)
        return DEFAULT_CONFIG.copy()

    # Fill in any missing keys from defaults
    merged = {**DEFAULT_CONFIG, **cfg}
    return merged


def save_config(cfg: dict, path: Path = _CONFIG_PATH) -> bool:
    """
    Write config dict to config.json.

    Returns
    -------
    bool : True on success, False on failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Config saved to %s", path)
        return True
    except OSError as e:
        logger.error("Failed to save config: %s", e)
        return False


def update_config_key(key: str, value, path: Path = _CONFIG_PATH) -> bool:
    """Update a single key in config.json."""
    cfg = load_config(path)
    cfg[key] = value
    return save_config(cfg, path)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

class PipelineService:
    def start_dashboard(self) -> bool:
        """Start dashboard/app.py as a subprocess."""
        if self._flask_proc is not None and self._flask_proc.poll() is None:
            logger.warning("Dashboard already running.")
            return False

        cmd = [sys.executable, str(Path("dashboard") / "app.py")]
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._flask_proc = subprocess.Popen(
                cmd,
                stdout=open(_log_path(), "a"),
                stderr=subprocess.STDOUT,
                **kwargs,
            )
            logger.info("Dashboard started — PID %s", self._flask_proc.pid)
            return True
        except Exception as e:
            logger.error("Failed to start dashboard: %s", e)
            return False
        
    def stop_dashboard(self, timeout: float = 5.0) -> bool:
        if self._flask_proc is None or self._flask_proc.poll() is not None:
            return True
        try:
            self._flask_proc.terminate()
            self._flask_proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._flask_proc.kill()
            self._flask_proc.wait()
        self._flask_proc = None
        return True
    
    def find_existing_process(self) -> Optional[int]:
        """Find existing pipeline process by checking for a stored PID file."""
        pid_file = Path("pipeline.pid")
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    pid = int(f.read().strip())
                # Verify process is still running
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}"],
                        capture_output=True, text=True
                    )
                    if str(pid) in result.stdout:
                        return pid
                else:
                    os.kill(pid, 0)  # Signal 0 checks if process exists
                    return pid
                # Process not found, clean up stale PID file
                pid_file.unlink()
            except Exception:
                pid_file.unlink() if pid_file.exists() else None
        return None

    def __init__(self, config_path: Path = _CONFIG_PATH):
        self._config_path = config_path
        self._proc: Optional[subprocess.Popen] = None
        self._flask_proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, headless: Optional[bool] = None) -> bool:
        if self.is_running():
            logger.warning("Pipeline already running (PID %s).", self.pid())
            return False

        cfg = load_config(self._config_path)

        use_headless = headless if headless is not None else cfg.get("headless", True)
        # Build CLI args – matching pipeline.py exactly
        cmd = [
            sys.executable,
            str(Path("core") / "pipeline.py"),
        ]

        # Decide whether source is RTSP or video file/webcam
        src = cfg["rtsp_url"].strip()
        if src.startswith("rtsp://"):
            cmd += ["--rtsp", src]
        else:
            # Assume it's a video file path or webcam index (e.g. "0")
            cmd += ["--video", src]

        cmd += [
            "--db-path", cfg["db_path"],
            "--collection-name", cfg["collection_name"],
            "--yunet-model", cfg["detector_model"],
            "--output-csv", cfg["detections_csv"],
            "--crops-dir", cfg["crops_dir"],
            "--threshold-accept", str(cfg["score_threshold"]),
            "--refresh-interval", str(cfg["gallery_refresh_sec"]),
        ]
        if use_headless:
            cmd.append("--headless")

        # Ensure directories exist
        Path(cfg["detections_csv"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["crops_dir"]).mkdir(parents=True, exist_ok=True)

        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self._proc = subprocess.Popen(
                cmd,
                stdout=open(_log_path(), "a"),
                stderr=subprocess.STDOUT,
                **kwargs,
            )
            logger.info("Pipeline started — PID %s", self._proc.pid)
            
            # Store PID for future launcher instances
            with open(Path("pipeline.pid"), "w") as f:
                f.write(str(self._proc.pid))
            return True
        except Exception as e:
            logger.error("Failed to start pipeline: %s", e)
            return False
        

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop pipeline AND dashboard."""
        self.stop_dashboard()  # ← ADD THIS
    
        if self._proc is None or self._proc.poll() is not None:
            return True
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        return True

    def is_running(self) -> bool:
        """Return True if the pipeline subprocess is alive."""
        # If we have a subprocess object, check its status
        if self._proc is not None and self._proc.poll() is None:
            return True
        
        # If _proc is None or terminated, check if there's a valid PID file with running process
        pid_file = Path("pipeline.pid")
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    stored_pid = int(f.read().strip())
                
                # Verify the process is still running
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {stored_pid}"],
                        capture_output=True, text=True
                    )
                    if str(stored_pid) in result.stdout:
                        return True
                else:
                    os.kill(stored_pid, 0)  # Signal 0 checks if process exists
                    return True
            except Exception:
                # If we can't verify, clean up the stale PID file
                pid_file.unlink(missing_ok=True)
        
        return False

    def pid(self) -> Optional[int]:
        """Return PID of running pipeline, or None."""
        # First check if we have a live subprocess object
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        
        # Check if there's a valid PID file with running process
        pid_file = Path("pipeline.pid")
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    stored_pid = int(f.read().strip())
                
                # Verify the process is still running
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {stored_pid}"],
                        capture_output=True, text=True
                    )
                    if str(stored_pid) in result.stdout:
                        return stored_pid
                else:
                    try:
                        os.kill(stored_pid, 0)  # Signal 0 checks if process exists
                        return stored_pid
                    except OSError:
                        pass
            except Exception:
                pass
            
            # Clean up stale PID file
            pid_file.unlink(missing_ok=True)
        
        return None

    def exit_code(self) -> Optional[int]:
        """Return exit code if process has ended, None if still running."""
        if self._proc is None:
            return None
        return self._proc.poll()

    def restart(self, headless: Optional[bool] = None) -> bool:
        """Stop then start the pipeline."""
        self.stop()
        time.sleep(0.5)
        return self.start(headless=headless)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_log_path() -> Path:
        return _log_path()

    @staticmethod
    def get_recent_log(lines: int = 50) -> str:
        """Return the last N lines of the pipeline log file."""
        path = _log_path()
        if not path.exists():
            return "(no log yet)"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            return "".join(all_lines[-lines:])
        except OSError:
            return "(could not read log)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_path() -> Path:
    """pipeline.log sits next to config.json (project root)."""
    return Path("pipeline.log")