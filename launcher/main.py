"""
launcher/main.py
--------------------
Tkinter UI for Survil — setup wizard / control panel.
Single responsibility: UI only.
All business logic is delegated to:
    launcher/service.py           — subprocess management, config read/write
    builder/buildDataSet.py       — dataset builder window

Flow
----
1. User fills in RTSP URL, reviews settings
2. Clicks "Start" → service.py spawns pipeline.py as subprocess
3. User can close this window — pipeline keeps running in background
4. "Stop" button stops the pipeline subprocess
5. "Build Dataset" button opens BuildDatasetWindow (modal)
6. Status bar polls is_running() every 2 seconds

Run from project root:
    python launcher/main.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

logger = logging.getLogger(__name__)

# Add project root to path so imports from core/, builder/, launcher/ work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Change cwd to project root so relative paths in config.json work
os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class LauncherWindow(tk.Tk):
    """
    Main Survil launcher window.

    Layout
    ------
    ┌──────────────────────────────────────┐
    │  Header: logo + version              │
    ├──────────────────────────────────────┤
    │  Status bar (running / stopped)      │
    ├──────────────────────────────────────┤
    │  Config section                      │
    │    RTSP URL                          │
    │    Headless checkbox                 │
    │    Gallery refresh interval          │
    ├──────────────────────────────────────┤
    │  Control buttons                     │
    │    [Start]  [Stop]  [Restart]        │
    │    [Build Dataset]  [Open Dashboard] │
    ├──────────────────────────────────────┤
    │  Log tail (last 8 lines)             │
    └──────────────────────────────────────┘
    """

    _POLL_INTERVAL_MS = 2000   # status poll every 2 seconds
    _LOG_REFRESH_MS = 3000     # log tail refresh every 3 seconds

    def __init__(self):
        super().__init__()
        self.title("Survil — CCTV Face Identification")
        self.resizable(True, False)
        self.minsize(520, 580)

        # Import service here (after sys.path is set)
        from launcher.service import PipelineService, load_config, save_config

        # ... header, title, geometry code here ...

        # Store config functions first
        self._load_config = load_config
        self._save_config = save_config

        # Load config
        self._cfg = load_config()

        # Create service
        self._svc = PipelineService()

        # Check if old pipeline is running
        existing_pid = self._svc.find_existing_process()

        # Build UI (needs self._cfg)
        self._build_ui()
        self._load_config_into_ui()
        self._start_polling()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        PAD = {"padx": 12, "pady": 6}

        # ── Header ──────────────────────────────────────────────────
        header = tk.Frame(self, bg="#1a1a2e")
        header.pack(fill="x")
        tk.Label(
            header,
            text="SURVIL",
            font=("Helvetica", 22, "bold"),
            fg="#e94560",
            bg="#1a1a2e",
        ).pack(side="left", padx=16, pady=10)
        tk.Label(
            header,
            text="CCTV Face Identification",
            font=("Helvetica", 10),
            fg="#a0a0c0",
            bg="#1a1a2e",
        ).pack(side="left", pady=10)

        # ── Status bar ──────────────────────────────────────────────
        status_frame = tk.Frame(self, bg="#0f3460", height=36)
        status_frame.pack(fill="x")
        status_frame.pack_propagate(False)

        self._status_dot = tk.Label(status_frame, text="●", font=("Helvetica", 14),
                                    fg="#e74c3c", bg="#0f3460")
        self._status_dot.pack(side="left", padx=(12, 4), pady=4)

        self._status_label = tk.Label(
            status_frame,
            text="Stopped",
            font=("Helvetica", 10, "bold"),
            fg="#ecf0f1",
            bg="#0f3460",
        )
        self._status_label.pack(side="left", pady=4)

        self._pid_label = tk.Label(
            status_frame, text="", font=("Helvetica", 9),
            fg="#95a5a6", bg="#0f3460"
        )
        self._pid_label.pack(side="right", padx=12, pady=4)

        # ── Config section ───────────────────────────────────────────
        cfg_frame = ttk.LabelFrame(self, text="Configuration")
        cfg_frame.pack(fill="x", **PAD)

        # RTSP URL
        ttk.Label(cfg_frame, text="RTSP URL:").grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        self._rtsp_var = tk.StringVar()
        ttk.Entry(cfg_frame, textvariable=self._rtsp_var, width=42).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=4
        )

        
        # DB path
        ttk.Label(cfg_frame, text="DB path:").grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )
        self._db_var = tk.StringVar()
        ttk.Entry(cfg_frame, textvariable=self._db_var, width=42).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=4
        )

        # Gallery refresh interval
        ttk.Label(cfg_frame, text="Gallery refresh (sec):").grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        self._refresh_var = tk.StringVar()
        ttk.Entry(cfg_frame, textvariable=self._refresh_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=4
        )

        # Headless checkbox
        self._headless_var = tk.BooleanVar()
        ttk.Checkbutton(
            cfg_frame,
            text="Headless mode (no preview window)",
            variable=self._headless_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=4)

        cfg_frame.columnconfigure(1, weight=1)

        ttk.Button(cfg_frame, text="Save config", command=self._save_cfg).grid(
            row=4, column=0, columnspan=3, pady=6
        )

        # ── Control buttons ──────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self, text="Pipeline Control")
        ctrl_frame.pack(fill="x", **PAD)

        btn_row1 = ttk.Frame(ctrl_frame)
        btn_row1.pack(pady=(8, 4))

        self._start_btn = ttk.Button(btn_row1, text="▶  Start", command=self._on_start, width=12)
        self._start_btn.pack(side="left", padx=6)

        self._stop_btn = ttk.Button(btn_row1, text="■  Stop", command=self._on_stop,
                                    width=12, state="disabled")
        self._stop_btn.pack(side="left", padx=6)

        self._restart_btn = ttk.Button(btn_row1, text="↺  Restart", command=self._on_restart,
                                       width=12, state="disabled")
        self._restart_btn.pack(side="left", padx=6)

        btn_row2 = ttk.Frame(ctrl_frame)
        btn_row2.pack(pady=(4, 8))

        ttk.Button(
            btn_row2, text="🗂  Build Dataset",
            command=self._on_build_dataset, width=18
        ).pack(side="left", padx=6)

        self._dashboard_btn = ttk.Button(
            btn_row2, text="🌐  Open Dashboard",
            command=self._on_open_dashboard, width=18
        )
        self._dashboard_btn.pack(side="left", padx=6)

        # ── Log tail ──────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Pipeline log (last 50 lines)")
        log_frame.pack(fill="both", expand=True, **PAD)

        self._log_text = tk.Text(
            log_frame, height=10, font=("Courier", 9),
            state="disabled", wrap="none", bg="#0d0d0d", fg="#a8ff78",
            insertbackground="white",
        )
        log_scroll_y = ttk.Scrollbar(log_frame, orient="vertical",
                                     command=self._log_text.yview)
        log_scroll_x = ttk.Scrollbar(log_frame, orient="horizontal",
                                     command=self._log_text.xview)
        self._log_text.configure(
            yscrollcommand=log_scroll_y.set,
            xscrollcommand=log_scroll_x.set,
        )
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(4, 0))
        log_scroll_y.grid(row=0, column=1, sticky="ns", pady=(4, 0))
        log_scroll_x.grid(row=1, column=0, sticky="ew", padx=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_config_into_ui(self):
        self._rtsp_var.set(self._cfg.get("rtsp_url", ""))
        self._db_var.set(self._cfg.get("db_path", "data/face_db"))
        self._refresh_var.set(str(self._cfg.get("gallery_refresh_sec", 60)))
        self._headless_var.set(bool(self._cfg.get("headless", True)))

    def _save_cfg(self):
        """Read UI fields -> update cfg dict -> write config.json."""
        try:
            refresh = int(self._refresh_var.get())
        except ValueError:
            messagebox.showwarning("Invalid value", "Gallery refresh must be a number.")
            return

        self._cfg["rtsp_url"] = self._rtsp_var.get().strip()
        self._cfg["db_path"] = self._db_var.get().strip()
        self._cfg["gallery_refresh_sec"] = refresh
        self._cfg["headless"] = self._headless_var.get()

        if self._save_config(self._cfg):
            messagebox.showinfo("Saved", "config.json updated.")
        else:
            messagebox.showerror("Error", "Failed to save config.json.")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self):
        self._save_cfg_silent()
        success = self._svc.start(headless=self._headless_var.get())
        self._svc.start_dashboard()  # ← add this
        if not success:
            messagebox.showerror(...)
        self._update_status()

    def _on_stop(self):
        self._svc.stop()
        self._svc.stop_dashboard() 
        self._update_status()

    def _on_restart(self):
        self._save_cfg_silent()
        self._svc.stop_dashboard() 
        self._svc.restart(headless=self._headless_var.get())
        self._update_status()

    def _on_build_dataset(self):
        """Open Build Dataset window as a modal child."""
        try:
            from builder.buildDataSet import open_build_dataset_window
            open_build_dataset_window(
                parent=self,
                db_path=self._cfg.get("db_path", "data/face_db"),
                model_dir=self._cfg.get("model_dir", "models/buffalo_l"),
            )
        except ImportError as e:
            messagebox.showerror("Import error", str(e))

    def _on_open_dashboard(self):
        port = self._cfg.get("flask_port", 5002)
        self._svc.start_dashboard()  # ensures Flask is up
        import time; time.sleep(1)   # brief wait for Flask to bind
        webbrowser.open(f"http://localhost:{port}")

    def _on_close(self):
        self._svc.stop_dashboard() 
        """Closing the launcher does NOT stop the pipeline subprocess."""
        if self._svc.is_running():
            if messagebox.askyesno(
                "Pipeline still running",
                "The detection pipeline is running in the background.\n"
                "It will continue after you close this window.\n\n"
                "Close the launcher?"
            ):
                self.destroy()
        else:
            self.destroy()

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _start_polling(self):
        self._update_status()
        self._refresh_log()

    def _update_status(self):
        """Called every _POLL_INTERVAL_MS ms to update status bar and button states."""
        running = self._svc.is_running()
        pid = self._svc.pid()

        if running:
            self._status_dot.config(fg="#2ecc71")
            self._status_label.config(text="Running")
            self._pid_label.config(text=f"PID {pid}")
            self._start_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
            self._restart_btn.config(state="normal")
        else:
            exit_code = self._svc.exit_code()
            if exit_code is not None and exit_code != 0:
                self._status_dot.config(fg="#e74c3c")
                self._status_label.config(text=f"Crashed (exit {exit_code})")
            else:
                self._status_dot.config(fg="#e74c3c")
                self._status_label.config(text="Stopped")
            self._pid_label.config(text="")
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._restart_btn.config(state="disabled")

        self.after(self._POLL_INTERVAL_MS, self._update_status)

    def _refresh_log(self):
        """Refresh log tail text widget."""
        from launcher.service import PipelineService
        log = PipelineService.get_recent_log(50)
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.insert("end", log)
        self._log_text.see("end")
        self._log_text.config(state="disabled")
        self.after(self._LOG_REFRESH_MS, self._refresh_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_cfg_silent(self):
        """Save config without showing a messagebox (used before start/restart)."""
        try:
            self._cfg["rtsp_url"] = self._rtsp_var.get().strip()
            self._cfg["db_path"] = self._db_var.get().strip()
            self._cfg["gallery_refresh_sec"] = int(self._refresh_var.get())
            self._cfg["headless"] = self._headless_var.get()
            self._save_config(self._cfg)
        except Exception as e:
            logger.warning("Silent config save failed: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = LauncherWindow()
    app.mainloop()
