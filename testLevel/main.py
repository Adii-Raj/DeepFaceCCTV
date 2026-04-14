"""
main.py  —  CCTV Attendance System — Master Control Dashboard
Launches brain.py (AI recognition) + eyes.py (camera tracking)
as multiprocessing workers and provides a live Tkinter control panel.

Run:  python main.py
"""

import os
import sys
import time
import queue
import threading
import multiprocessing
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

# ── local modules ───────────────────────────────────────────────
from config import (
    DB_PATH, CAMERAS, CACHE_DIR, MODEL_NAME,
    MAX_QUEUE_SIZE, DISTANCE_THRESH
)
import brain
import eyes
from enroller import open_enroller

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DB_PATH,   exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  SHARED SESSION STATE
# ═══════════════════════════════════════════════════════════════
class SessionState:
    """Thread-safe store for everything the UI needs."""
    def __init__(self):
        self._lock          = threading.Lock()
        # attendance_log: list of dicts
        #   { name, cam, track_id, timestamp, confidence, snapshot_path }
        self.attendance_log = []
        # present_today: { name: { first_seen, count, best_conf, snapshot_path } }
        self.present_today  = {}

    def record(self, name, cam, track_id, confidence, snapshot: np.ndarray):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snap_path = None
        if snapshot is not None and snapshot.size > 0:
            snap_path = os.path.join(
                CACHE_DIR, f"{name}_{ts.replace(':','-').replace(' ','_')}.jpg")
            cv2.imwrite(snap_path, snapshot)

        with self._lock:
            self.attendance_log.append({
                "name":       name,
                "cam":        cam,
                "track_id":   track_id,
                "timestamp":  ts,
                "confidence": confidence,
                "snapshot":   snap_path
            })
            if name not in self.present_today:
                self.present_today[name] = {
                    "first_seen":  ts,
                    "count":       1,
                    "best_conf":   confidence,
                    "snapshot":    snap_path
                }
            else:
                p = self.present_today[name]
                p["count"] += 1
                if confidence > p["best_conf"]:
                    p["best_conf"] = confidence
                    if snap_path:
                        p["snapshot"] = snap_path

    def snapshot(self):
        with self._lock:
            return list(self.present_today.items())


SESSION = SessionState()


# ═══════════════════════════════════════════════════════════════
#  RESULT BRIDGE  (reads result_queue → updates SESSION)
# ═══════════════════════════════════════════════════════════════
def result_bridge(result_queue: queue.Queue,
                  snapshot_source: dict,
                  stop_event: threading.Event):
    """
    Runs in a thread. Pulls (cam_name, track_id, name, conf) tuples
    from result_queue and writes to SESSION.
    snapshot_source is a shared dict: { cam_name: latest_frame }
    """
    recent = {}   # name → last-logged epoch (de-duplicate at display level)
    WAIT   = 60   # seconds between logging same name

    while not stop_event.is_set():
        try:
            cam_name, track_id, name, conf = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if name == "Unknown":
            continue

        now = time.time()
        if now - recent.get(name, 0) < WAIT:
            continue
        recent[name] = now

        snap = snapshot_source.get(cam_name)
        SESSION.record(name, cam_name, track_id, conf, snap)


# ═══════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Attendance System — Dashboard")
        self.root.configure(bg="#070b14")
        self.root.geometry("1200x720")
        self.root.minsize(900, 600)

        # ── Queues & state ────────────────────────────────────
        self.face_queue    = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue  = queue.Queue(maxsize=200)
        self.cache_counter = multiprocessing.Value("i", 0)
        self.stop_event    = threading.Event()
        self.latest_frames = {}   # cam_name → np.ndarray (latest raw frame)

        # ── Build UI ──────────────────────────────────────────
        self._build_ui()

        # ── Start workers ─────────────────────────────────────
        self._start_workers()

        # ── Periodic UI refresh ───────────────────────────────
        self._refresh_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)

    # ─────────────────────────────────────────────────────────
    #  UI BUILD
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header strip ─────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#070b14", pady=10)
        hdr.pack(fill=tk.X, padx=18)

        tk.Label(hdr, text="◉  ATTENDANCE SYSTEM",
                 font=("Courier New", 19, "bold"),
                 fg="#00e5ff", bg="#070b14").pack(side=tk.LEFT)

        self.clock_var = tk.StringVar()
        tk.Label(hdr, textvariable=self.clock_var,
                 font=("Courier New", 12), fg="#546e7a",
                 bg="#070b14").pack(side=tk.LEFT, padx=20)

        # right-side action buttons
        btn_row = tk.Frame(hdr, bg="#070b14")
        btn_row.pack(side=tk.RIGHT)

        self.enroll_btn = tk.Button(
            btn_row, text="➕  Enroll Student",
            font=("Courier New", 11, "bold"),
            bg="#00695c", fg="white",
            activebackground="#00897b",
            bd=0, padx=14, pady=6, cursor="hand2",
            command=self._open_enroller
        )
        self.enroll_btn.pack(side=tk.LEFT, padx=6)

        self.report_btn = tk.Button(
            btn_row, text="📋  Full Report",
            font=("Courier New", 11, "bold"),
            bg="#1565c0", fg="white",
            activebackground="#1976d2",
            bd=0, padx=14, pady=6, cursor="hand2",
            command=self._open_report
        )
        self.report_btn.pack(side=tk.LEFT, padx=6)

        self.stop_btn = tk.Button(
            btn_row, text="⬛  Stop System",
            font=("Courier New", 11, "bold"),
            bg="#b71c1c", fg="white",
            activebackground="#c62828",
            bd=0, padx=14, pady=6, cursor="hand2",
            command=self._shutdown
        )
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        # ── Stat bar ──────────────────────────────────────────
        stat_bar = tk.Frame(self.root, bg="#0d1117", pady=8)
        stat_bar.pack(fill=tk.X, padx=18, pady=(0, 6))

        self._stat_vars = {}
        for key, label, color in [
            ("present",  "Present Today",  "#00e5ff"),
            ("total_db", "In Database",    "#69f0ae"),
            ("queue",    "AI Queue",       "#ffeb3b"),
            ("logs",     "Log Entries",    "#ff7043"),
        ]:
            box = tk.Frame(stat_bar, bg="#131a24", padx=18, pady=6,
                           bd=1, relief=tk.FLAT)
            box.pack(side=tk.LEFT, padx=6)
            v = tk.StringVar(value="0")
            self._stat_vars[key] = v
            tk.Label(box, text=label,
                     font=("Courier New", 8), fg="#455a64",
                     bg="#131a24").pack(anchor="w")
            tk.Label(box, textvariable=v,
                     font=("Courier New", 18, "bold"),
                     fg=color, bg="#131a24").pack()

        # ── Main body ─────────────────────────────────────────
        body = tk.Frame(self.root, bg="#070b14")
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=4)

        # LEFT — present students grid
        left = tk.Frame(body, bg="#0d1117", bd=1, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="  PRESENT STUDENTS",
                 font=("Courier New", 12, "bold"),
                 fg="#00e5ff", bg="#0d1117",
                 anchor="w").pack(fill=tk.X, pady=(8, 4))

        self.grid_canvas = tk.Canvas(left, bg="#0d1117",
                                     highlightthickness=0)
        grid_sb = tk.Scrollbar(left, orient="vertical",
                               command=self.grid_canvas.yview)
        self.grid_inner = tk.Frame(self.grid_canvas, bg="#0d1117")
        self.grid_inner.bind("<Configure>",
            lambda e: self.grid_canvas.configure(
                scrollregion=self.grid_canvas.bbox("all")))
        self.grid_canvas.create_window(
            (0, 0), window=self.grid_inner, anchor="nw")
        self.grid_canvas.configure(yscrollcommand=grid_sb.set)
        self.grid_canvas.pack(side="left", fill="both", expand=True)
        grid_sb.pack(side="right", fill="y")

        # RIGHT — live log feed
        right = tk.Frame(body, bg="#070b14", width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="LIVE LOG",
                 font=("Courier New", 12, "bold"),
                 fg="#69f0ae", bg="#070b14").pack(anchor="w", pady=(4, 4))

        log_wrap = tk.Frame(right, bg="#0d1117", bd=1, relief=tk.FLAT)
        log_wrap.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_wrap, bg="#0d1117", fg="#b0bec5",
            font=("Courier New", 9),
            relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD, padx=8, pady=6
        )
        log_sb = tk.Scrollbar(log_wrap, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

        # ── Status bar ────────────────────────────────────────
        self.status_var = tk.StringVar(value="⏳ Booting AI workers…")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Courier New", 9), fg="#455a64",
                 bg="#070b14", anchor="w").pack(
                     fill=tk.X, padx=18, pady=(4, 6))

        # ── Tag colors for log ────────────────────────────────
        self.log_text.tag_config("name",  foreground="#00e5ff")
        self.log_text.tag_config("cam",   foreground="#69f0ae")
        self.log_text.tag_config("conf",  foreground="#ffeb3b")
        self.log_text.tag_config("time",  foreground="#546e7a")
        self.log_text.tag_config("sep",   foreground="#1e2a38")

        self._grid_widgets = {}   # name → Frame widget in grid

    # ─────────────────────────────────────────────────────────
    #  WORKERS
    # ─────────────────────────────────────────────────────────
    def _start_workers(self):
        # ── Brain (AI recognition) in dedicated thread ────────
        threading.Thread(
            target=brain.start_ai_worker,
            args=(self.face_queue, self.result_queue, self.cache_counter),
            daemon=True
        ).start()

        # ── Eyes (camera + YOLO) one thread per camera ────────
        for cam_name, url in CAMERAS.items():
            threading.Thread(
                target=eyes.start_camera_worker,
                args=(cam_name, url,
                      self.face_queue, self.result_queue,
                      self.cache_counter),
                daemon=True
            ).start()

        # ── Result bridge ─────────────────────────────────────
        threading.Thread(
            target=result_bridge,
            args=(self.result_queue, self.latest_frames, self.stop_event),
            daemon=True
        ).start()

        self.status_var.set(
            f"✅ System running — {len(CAMERAS)} camera(s) active")

    # ─────────────────────────────────────────────────────────
    #  PERIODIC UI REFRESH  (runs on main thread via after())
    # ─────────────────────────────────────────────────────────
    def _refresh_ui(self):
        # clock
        self.clock_var.set(datetime.now().strftime("%A  %d %b %Y   %H:%M:%S"))

        # stats
        db_count = sum(
            1 for d in os.listdir(DB_PATH)
            if os.path.isdir(os.path.join(DB_PATH, d))
        ) if os.path.isdir(DB_PATH) else 0

        present_snap = SESSION.snapshot()
        self._stat_vars["present"].set(str(len(present_snap)))
        self._stat_vars["total_db"].set(str(db_count))
        self._stat_vars["queue"].set(str(self.cache_counter.value))
        self._stat_vars["logs"].set(str(len(SESSION.attendance_log)))

        # update present grid
        self._update_grid(present_snap)

        # update log
        self._update_log()

        self.root.after(1500, self._refresh_ui)

    def _update_grid(self, present_snap):
        names_now = {name for name, _ in present_snap}

        # add new cards
        for name, info in present_snap:
            if name not in self._grid_widgets:
                self._add_student_card(name, info)

        # update existing (conf / count may change)
        for name, info in present_snap:
            if name in self._grid_widgets:
                w = self._grid_widgets[name]
                w["conf_var"].set(f"{info['best_conf']*100:.1f}%")
                w["count_var"].set(f"Seen ×{info['count']}")

    def _add_student_card(self, name, info):
        card = tk.Frame(self.grid_inner, bg="#131a24",
                        bd=1, relief=tk.FLAT,
                        padx=8, pady=8)
        card.pack(fill=tk.X, padx=8, pady=4)

        # snapshot thumbnail
        snap_path = info.get("snapshot")
        if snap_path and os.path.exists(snap_path):
            img = Image.open(snap_path).resize(
                (52, 52), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl = tk.Label(card, image=imgtk, bg="#131a24")
            lbl.image = imgtk
            lbl.pack(side=tk.LEFT, padx=(0, 10))
        else:
            tk.Label(card, text="👤", font=("Courier New", 24),
                     bg="#131a24", fg="#37474f").pack(
                         side=tk.LEFT, padx=(0, 10))

        # text info
        txt = tk.Frame(card, bg="#131a24")
        txt.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(txt, text=name,
                 font=("Courier New", 13, "bold"),
                 fg="white", bg="#131a24").pack(anchor="w")

        conf_var  = tk.StringVar(value=f"{info['best_conf']*100:.1f}%")
        count_var = tk.StringVar(value=f"Seen ×{info['count']}")

        tk.Label(txt, textvariable=conf_var,
                 font=("Courier New", 10),
                 fg="#ffeb3b", bg="#131a24").pack(anchor="w")
        tk.Label(txt, text=f"First: {info['first_seen'][11:]}",
                 font=("Courier New", 9),
                 fg="#546e7a", bg="#131a24").pack(anchor="w")
        tk.Label(txt, textvariable=count_var,
                 font=("Courier New", 9),
                 fg="#69f0ae", bg="#131a24").pack(anchor="w")

        self._grid_widgets[name] = {
            "frame":     card,
            "conf_var":  conf_var,
            "count_var": count_var
        }

    def _update_log(self):
        # only append NEW entries (track via index)
        if not hasattr(self, "_log_index"):
            self._log_index = 0

        new_entries = SESSION.attendance_log[self._log_index:]
        if not new_entries:
            return

        self.log_text.configure(state=tk.NORMAL)
        for e in new_entries:
            ts   = e["timestamp"][11:]   # HH:MM:SS
            conf = f"{e['confidence']*100:.1f}%"
            self.log_text.insert("end", "─" * 34 + "\n", "sep")
            self.log_text.insert("end", f"{e['name']}\n", "name")
            self.log_text.insert("end",
                f"  {e['cam']}  ", "cam")
            self.log_text.insert("end",
                f"conf {conf}\n", "conf")
            self.log_text.insert("end",
                f"  {ts}\n", "time")

        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see("end")
        self._log_index = len(SESSION.attendance_log)

    # ─────────────────────────────────────────────────────────
    #  ENROLLER
    # ─────────────────────────────────────────────────────────
    def _open_enroller(self):
        def _on_enroll_save(name):
            """Called by enroller when a face is saved — hot-reload brain DB."""
            if name:
                print(f"[MAIN] 🔄 Hot-reloading DB for new enrollment: {name}")
                threading.Thread(
                    target=brain.process_new_faces, daemon=True).start()

        open_enroller(parent=self.root, on_close_callback=_on_enroll_save)

    # ─────────────────────────────────────────────────────────
    #  FULL REPORT WINDOW
    # ─────────────────────────────────────────────────────────
    def _open_report(self):
        ReportWindow(self.root, SESSION)

    # ─────────────────────────────────────────────────────────
    #  SHUTDOWN
    # ─────────────────────────────────────────────────────────
    def _shutdown(self):
        if not messagebox.askyesno(
                "Confirm Shutdown",
                "Stop the CCTV system and exit?"):
            return
        self.stop_event.set()
        self._save_session_csv()
        self.root.destroy()

    def _save_session_csv(self):
        if not SESSION.attendance_log:
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"attendance_{ts}.csv"
        with open(out, "w") as f:
            f.write("Name,Camera,Track ID,Timestamp,Confidence\n")
            for e in SESSION.attendance_log:
                f.write(
                    f"{e['name']},{e['cam']},{e['track_id']},"
                    f"{e['timestamp']},{e['confidence']:.4f}\n"
                )
        print(f"[MAIN] 💾 Session saved → {out}")


# ═══════════════════════════════════════════════════════════════
#  FULL REPORT WINDOW
# ═══════════════════════════════════════════════════════════════
class ReportWindow:
    def __init__(self, parent, session: SessionState):
        self.win = tk.Toplevel(parent)
        self.win.title("Session Report — All Detections")
        self.win.configure(bg="#070b14")
        self.win.geometry("1100x700")
        self.win.grab_set()
        self.session = session
        self._build(session.snapshot())

    def _build(self, present):
        # Header
        hdr = tk.Frame(self.win, bg="#070b14", pady=12)
        hdr.pack(fill=tk.X, padx=20)

        tk.Label(hdr, text="SESSION REPORT",
                 font=("Courier New", 18, "bold"),
                 fg="#00e5ff", bg="#070b14").pack(side=tk.LEFT)
        tk.Label(hdr,
                 text=f"{len(present)} students detected  ·  "
                      f"{datetime.now().strftime('%d %b %Y  %H:%M')}",
                 font=("Courier New", 11), fg="#546e7a",
                 bg="#070b14").pack(side=tk.LEFT, padx=18)

        # Export button
        tk.Button(hdr, text="💾 Export CSV",
                  font=("Courier New", 10, "bold"),
                  bg="#1565c0", fg="white", bd=0,
                  padx=12, pady=5, cursor="hand2",
                  command=self._export).pack(side=tk.RIGHT)

        # Column headers
        cols_f = tk.Frame(self.win, bg="#0d1117", pady=7)
        cols_f.pack(fill=tk.X, padx=20)
        for text, w in [("PHOTO", 8), ("NAME", 20), ("FIRST SEEN", 14),
                         ("DETECTIONS", 12), ("BEST CONF", 10)]:
            tk.Label(cols_f, text=text,
                     font=("Courier New", 9, "bold"),
                     fg="#455a64", bg="#0d1117",
                     width=w, anchor="w").pack(side=tk.LEFT, padx=6)

        # Scrollable rows
        outer = tk.Frame(self.win, bg="#070b14")
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=6)

        canvas = tk.Canvas(outer, bg="#070b14",
                           highlightthickness=0)
        sb     = tk.Scrollbar(outer, orient="vertical",
                              command=canvas.yview)
        inner  = tk.Frame(canvas, bg="#070b14")
        inner.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for name, info in sorted(present, key=lambda x: x[1]["first_seen"]):
            self._add_row(inner, name, info)

        # Close
        tk.Button(self.win, text="Close",
                  font=("Courier New", 10, "bold"),
                  bg="#263238", fg="#b0bec5",
                  bd=0, padx=20, pady=6, cursor="hand2",
                  command=self.win.destroy).pack(pady=10)

    def _add_row(self, parent, name, info):
        row = tk.Frame(parent, bg="#0d1421",
                       pady=6, padx=6)
        row.pack(fill=tk.X, pady=2, padx=4)

        # photo
        snap = info.get("snapshot")
        if snap and os.path.exists(snap):
            img = Image.open(snap).resize(
                (48, 48), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl = tk.Label(row, image=imgtk, bg="#0d1421")
            lbl.image = imgtk
            lbl.pack(side=tk.LEFT, padx=(0, 8))
        else:
            tk.Label(row, text="👤", font=("Courier New", 20),
                     bg="#0d1421", fg="#37474f").pack(
                         side=tk.LEFT, padx=(0, 8))

        # name
        tk.Label(row, text=name,
                 font=("Courier New", 12, "bold"),
                 fg="white", bg="#0d1421",
                 width=20, anchor="w").pack(side=tk.LEFT, padx=6)

        # first seen
        tk.Label(row, text=info["first_seen"][11:],
                 font=("Courier New", 11),
                 fg="#69f0ae", bg="#0d1421",
                 width=14, anchor="w").pack(side=tk.LEFT, padx=6)

        # count
        tk.Label(row, text=str(info["count"]),
                 font=("Courier New", 11, "bold"),
                 fg="#ffeb3b", bg="#0d1421",
                 width=12, anchor="w").pack(side=tk.LEFT, padx=6)

        # best conf
        conf_pct = info["best_conf"] * 100
        color = ("#00e5ff" if conf_pct >= 85
                 else "#ffeb3b" if conf_pct >= 70
                 else "#ff7043")
        tk.Label(row, text=f"{conf_pct:.1f}%",
                 font=("Courier New", 11, "bold"),
                 fg=color, bg="#0d1421",
                 width=10, anchor="w").pack(side=tk.LEFT, padx=6)

    def _export(self):
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"report_{ts}.csv"
        present = self.session.snapshot()
        with open(out, "w") as f:
            f.write("Name,First Seen,Detections,Best Confidence\n")
            for name, info in sorted(present):
                f.write(
                    f"{name},{info['first_seen']},"
                    f"{info['count']},{info['best_conf']:.4f}\n"
                )
        messagebox.showinfo("Exported", f"Report saved to:\n{out}")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app  = Dashboard(root)
    root.mainloop()