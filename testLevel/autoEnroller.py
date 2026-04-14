"""
enroller.py  ─  CCTV Auto-Capture Enroller (Multi-Photo Cache + Name Correction)
════════════════════════════════════════════════════════════════
HOW IT WORKS
────────────
Starts brain.py (AI recognition) + eyes.py (YOLO camera) in
background threads — fully automatic, no manual interaction.
Whenever brain recognises someone at ≥ ENROLL_CONF_THRESH
confidence, the CCTV crop is SAVED to a cache (no longer overridden).
Click ⬛ STOP → a Review window opens showing EVERY detected
photo as a side-by-side card.
LEFT  = original DB photo  (what the system thinks it is)
RIGHT = CCTV snapshot      (what the camera actually saw)
You click ✅ CONFIRM, ❌ REJECT, or ✏️ CORRECT NAME for EACH card.
✅ Confirmed images are saved into my_database/<name>/
✏️ Corrected images are saved into my_database/<your_input>/
❌ Rejected images are discarded.
Run:
python enroller.py
"""
import os
import cv2
import time
import queue
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
from PIL import Image, ImageTk
import multiprocessing
import ctypes
from config import (
    DB_PATH, CAMERAS,
    MAX_QUEUE_SIZE,
)
import brain
import eyes

#════════════════════════════════════════════════════════════
#ENROLLER CONFIG
#════════════════════════════════════════════════════════════
ENROLL_CONF_THRESH = 0.65   # 65% minimum confidence to cache a snapshot

#════════════════════════════════════════════════════════════
#PENDING CACHE  (thread-safe, stores ALL photos)
#════════════════════════════════════════════════════════════
class PendingCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._items = []
        self._id_counter = 0

    def add(self, name, confidence, cctv_img, db_img_path):
        with self._lock:
            self._id_counter += 1
            self._items.append({
                "id": self._id_counter,
                "name": name,
                "confidence": confidence,
                "cctv_img": cctv_img.copy(),
                "db_img_path": db_img_path,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            return True

    def snapshot(self):
        with self._lock:
            return list(self._items)

    def count(self):
        with self._lock:
            return len(self._items)

    def latest_per_person(self):
        with self._lock:
            best = {}
            for item in self._items:
                name = item["name"]
                if name not in best or item["confidence"] > best[name]["confidence"]:
                    best[name] = item
            return best

#════════════════════════════════════════════════════════════
#HELPERS
#════════════════════════════════════════════════════════════
def _first_db_image(name):
    folder = os.path.join(DB_PATH, name)
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder, f)
    return None

def _next_img_path(name):
    folder = os.path.join(DB_PATH, name)
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return os.path.join(folder, f"img{len(existing)+1}.jpg")

def _tk_img(np_bgr, size):
    rgb = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image=img)

def _tk_img_path(path, size):
    img = Image.open(path).convert("RGB").resize(
        size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image=img)

#════════════════════════════════════════════════════════════
#BRAIN WRAPPER
#════════════════════════════════════════════════════════════
class _EnrollerBrain:
    def __init__(self, pending, stop_event, on_update):
        self.pending     = pending
        self.stop_event  = stop_event
        self.on_update   = on_update
        self.face_queue    = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue  = queue.Queue(maxsize=400)
        self.cache_counter = multiprocessing.Value(ctypes.c_int, 0)
        self._crop_buf      = {}
        self._crop_buf_lock = threading.Lock()

    def _tap_worker(self, real_q):
        while not self.stop_event.is_set():
            try:
                item = self.face_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            cam_name, face_crop, track_id, timestamp = item
            with self._crop_buf_lock:
                self._crop_buf[track_id] = face_crop.copy()
            try:
                real_q.put_nowait(item)
            except queue.Full:
                pass

    def _result_watcher(self):
        while not self.stop_event.is_set():
            try:
                cam_name, track_id, name, conf = \
                    self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if name == "Unknown" or conf < ENROLL_CONF_THRESH:
                continue

            with self._crop_buf_lock:
                crop = self._crop_buf.get(track_id)
            if crop is None:
                continue

            db_path = _first_db_image(name)
            if db_path is None:
                continue

            self.pending.add(name, conf, crop, db_path)
            self.on_update()

    def start(self):
        real_face_q = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        threading.Thread(target=self._tap_worker,
                         args=(real_face_q,), daemon=True).start()
        threading.Thread(target=brain.start_ai_worker,
                         args=(real_face_q,
                               self.result_queue,
                               self.cache_counter),
                         daemon=True).start()
        threading.Thread(target=self._result_watcher,
                         daemon=True).start()
        for cam_name, url in CAMERAS.items():
            threading.Thread(
                target=eyes.start_camera_worker,
                args=(cam_name, url,
                      self.face_queue,
                      self.result_queue,
                      self.cache_counter),
                daemon=True).start()

#════════════════════════════════════════════════════════════
#REVIEW WINDOW (Now with ✏️ Correct Name)
#════════════════════════════════════════════════════════════
class ReviewWindow:
    SZ = (190, 190)
    def __init__(self, parent, data, on_done):
        self.win     = tk.Toplevel(parent)
        self.win.title("Review Detections — Confirm or Reject")
        self.win.configure(bg="#07080f")
        self.win.geometry("1100x720")
        self.win.grab_set()
        self.data    = data
        self.on_done = on_done
        self._dec    = {}           # item_id → StringVar
        self._status_lbls = {}      # item_id → Label
        self._name_lbls = {}        # item_id → Label (for text updates)
        self._name_overrides = {}   # item_id → corrected_name
        self._build()

    def _build(self):
        hdr = tk.Frame(self.win, bg="#07080f", pady=12)
        hdr.pack(fill=tk.X, padx=20)
        tk.Label(hdr, text="CONFIRM DETECTIONS",
                 font=("Courier New", 18, "bold"),
                 fg="#00e5ff", bg="#07080f").pack(side=tk.LEFT)
        tk.Label(hdr,
                 text=f"{len(self.data)} photo(s) detected",
                 font=("Courier New", 11), fg="#546e7a",
                 bg="#07080f").pack(side=tk.LEFT, padx=14)

        bulk = tk.Frame(hdr, bg="#07080f")
        bulk.pack(side=tk.RIGHT)
        tk.Button(bulk, text="✅ Confirm All",
                  font=("Courier New", 10, "bold"),
                  bg="#1b5e20", fg="white", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self._confirm_all).pack(side=tk.LEFT, padx=4)
        tk.Button(bulk, text="❌ Reject All",
                  font=("Courier New", 10, "bold"),
                  bg="#7f0000", fg="white", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self._reject_all).pack(side=tk.LEFT, padx=4)

        ch = tk.Frame(self.win, bg="#0d1117", pady=6)
        ch.pack(fill=tk.X, padx=20)
        for txt in ["NAME & CONFIDENCE",
                     "◀  DB PHOTO  (system matched this)",
                     "▶  CCTV SNAPSHOT  (camera saw this)",
                     "DECISION"]:
            tk.Label(ch, text=txt,
                     font=("Courier New", 9, "bold"),
                     fg="#37474f", bg="#0d1117",
                     width=24, anchor="w").pack(
                         side=tk.LEFT, padx=6)

        outer = tk.Frame(self.win, bg="#07080f")
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=6)

        canvas = tk.Canvas(outer, bg="#07080f",
                           highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical",
                          command=canvas.yview)
        inner = tk.Frame(canvas, bg="#07080f")
        inner.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(
                -1*(e.delta//120), "units"))

        if not self.data:
            tk.Label(inner,
                     text="No students detected above the  "
                          f"{ENROLL_CONF_THRESH*100:.0f}% threshold.",
                     font=("Courier New", 13), fg="#546e7a",
                     bg="#07080f", pady=40).pack()
        else:
            for info in sorted(self.data, key=lambda x: (x["name"], x["id"])):
                self._add_card(inner, info)

        foot = tk.Frame(self.win, bg="#07080f", pady=8)
        foot.pack(fill=tk.X, padx=20)
        self._summary_var = tk.StringVar(value="")
        tk.Label(foot, textvariable=self._summary_var,
                 font=("Courier New", 10), fg="#69f0ae",
                 bg="#07080f").pack(side=tk.LEFT)
        tk.Button(foot, text="💾  Save Confirmed  & Close",
                  font=("Courier New", 12, "bold"),
                  bg="#0d47a1", fg="white",
                  activebackground="#1565c0",
                  bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._save_and_close).pack(side=tk.RIGHT)
        self._refresh_summary()

    def _add_card(self, parent, info):
        item_id = info["id"]
        name = info["name"]
        conf_pct = info["confidence"] * 100
        c_col = ("#00e5ff" if conf_pct >= 90
                 else "#69f0ae" if conf_pct >= 85
                 else "#ffeb3b")

        row = tk.Frame(parent, bg="#0d1421", pady=10, padx=10)
        row.pack(fill=tk.X, pady=4, padx=4)

        # ── col 1: name + meta ──────────────────────────────
        meta = tk.Frame(row, bg="#0d1421", width=175)
        meta.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        meta.pack_propagate(False)
        name_lbl = tk.Label(meta, text=name,
                 font=("Courier New", 13, "bold"),
                 fg="white", bg="#0d1421",
                 wraplength=160, justify="left")
        name_lbl.pack(anchor="w")
        self._name_lbls[item_id] = name_lbl

        tk.Label(meta, text=f"Conf: {conf_pct:.1f}%",
                 font=("Courier New", 10, "bold"),
                 fg=c_col, bg="#0d1421").pack(anchor="w", pady=(4, 0))
        tk.Label(meta, text=info["timestamp"],
                 font=("Courier New", 9),
                 fg="#455a64", bg="#0d1421").pack(anchor="w")

        # ── col 2: DB photo ──────────────────────────────────
        db_f = tk.Frame(row, bg="#0d1421")
        db_f.pack(side=tk.LEFT, padx=8)
        try:
            db_tk = _tk_img_path(info["db_img_path"], self.SZ)
            l = tk.Label(db_f, image=db_tk, bg="#0d1421",
                         bd=2, relief=tk.SOLID)
            l.image = db_tk
            l.pack()
        except Exception:
            tk.Label(db_f, text="[no DB\nphoto]",
                     font=("Courier New", 9), fg="#546e7a",
                     bg="#0d1421", width=12, height=8).pack()
        tk.Label(db_f, text="📁 From Database",
                 font=("Courier New", 8), fg="#455a64",
                 bg="#0d1421").pack(pady=(3, 0))

        tk.Label(row, text="≈",
                 font=("Courier New", 26, "bold"),
                 fg="#263238", bg="#0d1421").pack(
                     side=tk.LEFT, padx=4)

        # ── col 3: CCTV snapshot ──────────────────────────────
        cc_f = tk.Frame(row, bg="#0d1421")
        cc_f.pack(side=tk.LEFT, padx=8)
        try:
            cc_tk = _tk_img(info["cctv_img"], self.SZ)
            l2 = tk.Label(cc_f, image=cc_tk, bg="#0d1421",
                          bd=2, relief=tk.SOLID)
            l2.image = cc_tk
            l2.pack()
        except Exception:
            tk.Label(cc_f, text="[no CCTV\ncrop]",
                     font=("Courier New", 9), fg="#546e7a",
                     bg="#0d1421", width=12, height=8).pack()
        tk.Label(cc_f, text="📷 CCTV Capture",
                 font=("Courier New", 8), fg="#455a64",
                 bg="#0d1421").pack(pady=(3, 0))

        # ── col 4: decision ──────────────────────────────────
        dec_var = tk.StringVar(value="pending")
        self._dec[item_id] = dec_var

        btn_f = tk.Frame(row, bg="#0d1421")
        btn_f.pack(side=tk.LEFT, padx=(16, 0), fill=tk.Y)

        status_lbl = tk.Label(btn_f,
                              text="●  PENDING",
                              font=("Courier New", 10, "bold"),
                              fg="#546e7a", bg="#0d1421")
        status_lbl.pack(pady=(0, 8))
        self._status_lbls[item_id] = status_lbl

        def _confirm(lbl=status_lbl, v=dec_var):
            v.set("confirm")
            lbl.configure(text="✅  CONFIRMED", fg="#69f0ae")
            self._refresh_summary()

        def _reject(lbl=status_lbl, v=dec_var):
            v.set("reject")
            lbl.configure(text="❌  REJECTED", fg="#ef5350")
            self._refresh_summary()

        def _edit_name():
            new_name = simpledialog.askstring(
                "Correct Student Name",
                f"AI guessed: '{info['name']}'\n"
                "Enter the correct name (will auto-confirm):",
                parent=self.win
            )
            if new_name:
                new_name = new_name.strip()
                if new_name:
                    self._name_overrides[item_id] = new_name
                    name_lbl.configure(text=new_name, fg="#00e5ff")
                    dec_var.set("confirm")
                    status_lbl.configure(text="✅ CORRECTED", fg="#00e5ff")
                    self._refresh_summary()

        tk.Button(btn_f, text="✅  Confirm",
                  font=("Courier New", 11, "bold"),
                  bg="#1b5e20", fg="white",
                  activebackground="#2e7d32",
                  bd=0, padx=14, pady=7, cursor="hand2",
                  command=_confirm).pack(fill=tk.X, pady=3)

        tk.Button(btn_f, text="❌  Reject",
                  font=("Courier New", 11, "bold"),
                  bg="#7f0000", fg="white",
                  activebackground="#b71c1c",
                  bd=0, padx=14, pady=7, cursor="hand2",
                  command=_reject).pack(fill=tk.X, pady=3)

        tk.Button(btn_f, text="✏️  Correct Name",
                  font=("Courier New", 10, "bold"),
                  bg="#0d47a1", fg="white",
                  activebackground="#1565c0",
                  bd=0, padx=14, pady=5, cursor="hand2",
                  command=_edit_name).pack(fill=tk.X, pady=(6,0))

    def _confirm_all(self):
        for var in self._dec.values():
            var.set("confirm")
        for lbl in self._status_lbls.values():
            lbl.configure(text="✅  CONFIRMED", fg="#69f0ae")
        self._refresh_summary()

    def _reject_all(self):
        for var in self._dec.values():
            var.set("reject")
        for lbl in self._status_lbls.values():
            lbl.configure(text="❌  REJECTED", fg="#ef5350")
        self._refresh_summary()

    def _refresh_summary(self):
        c = sum(1 for v in self._dec.values() if v.get() == "confirm")
        r = sum(1 for v in self._dec.values() if v.get() == "reject")
        p = sum(1 for v in self._dec.values() if v.get() == "pending")
        self._summary_var.set(
            f"✅ {c} confirmed   ❌ {r} rejected   ● {p} pending")

    def _save_and_close(self):
        pending = sum(1 for v in self._dec.values() if v.get() == "pending")
        if pending > 0:
            if not messagebox.askyesno(
                     "Pending",
                    f"{pending} photo(s) still undecided.\n"
                     "They will be SKIPPED. Continue?"):
                return

        saved = 0
        for item in self.data:
            item_id = item["id"]
            if self._dec[item_id].get() != "confirm":
                continue
            # ✅ USE CORRECTED NAME IF AVAILABLE
            save_name = self._name_overrides.get(item_id, item["name"])
            path = _next_img_path(save_name)
            cv2.imwrite(path, item["cctv_img"])
            print(f"[ENROLLER] ✅ Saved '{save_name}' → {path}")
            saved += 1

        messagebox.showinfo(
             "Done",
            f"✅ {saved} image(s) saved to database.\n"
            f"❌ {len(self._dec) - saved} skipped/rejected."
        )
        self.win.destroy()
        self.on_done()

#════════════════════════════════════════════════════════════
#MAIN ENROLLER WINDOW
#════════════════════════════════════════════════════════════
class EnrollerApp:
    def __init__(self, root):
        self.root    = root
        self.root.title("CCTV Auto-Enroller — Leave Running")
        self.root.configure(bg="#07080f")
        self.root.geometry("900x580")
        self._pending    = PendingCache()
        self._stop_event = threading.Event()
        self._running    = True
        os.makedirs(DB_PATH, exist_ok=True)
        self._build_ui()
        self._brain = _EnrollerBrain(
            pending    = self._pending,
            stop_event = self._stop_event,
            on_update  = lambda: self.root.after(0, self._refresh_ui),
        )
        self._brain.start()
        self._refresh_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#07080f", pady=12)
        hdr.pack(fill=tk.X, padx=18)
        tk.Label(hdr, text="◉  CCTV AUTO-ENROLLER",
                 font=("Courier New", 17, "bold"),
                 fg="#00e5ff", bg="#07080f").pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="⏳ Booting AI & camera…")
        tk.Label(hdr, textvariable=self._status_var,
                 font=("Courier New", 10), fg="#546e7a",
                 bg="#07080f").pack(side=tk.LEFT, padx=14)
        self._stop_btn = tk.Button(
            hdr, text="⬛  STOP & REVIEW",
            font=("Courier New", 12, "bold"),
            bg="#b71c1c", fg="white",
            activebackground="#c62828",
            bd=0, padx=18, pady=7, cursor="hand2",
            command=self._stop_and_review
        )
        self._stop_btn.pack(side=tk.RIGHT)

        stats = tk.Frame(self.root, bg="#0d1117", pady=8)
        stats.pack(fill=tk.X, padx=18, pady=(0, 6))
        self._stat_vars = {}
        for key, label, color, init in [
            ("detected",   "Cached Photos",   "#00e5ff", "0"),
            ("db_people",  "People in DB",    "#69f0ae", "—"),
            ("min_conf",   "Min Confidence",  "#ffeb3b",
             f"{ENROLL_CONF_THRESH*100:.0f}%"),
            ("queue",      "AI Queue Depth",  "#ff7043", "0"),
        ]:
            box = tk.Frame(stats, bg="#131a24", padx=16, pady=6)
            box.pack(side=tk.LEFT, padx=6)
            v = tk.StringVar(value=init)
            self._stat_vars[key] = v
            tk.Label(box, text=label,
                     font=("Courier New", 8), fg="#455a64",
                     bg="#131a24").pack(anchor="w")
            tk.Label(box, textvariable=v,
                     font=("Courier New", 17, "bold"),
                     fg=color, bg="#131a24").pack()

        fl = tk.Frame(self.root, bg="#07080f")
        fl.pack(fill=tk.X, padx=18, pady=(4, 2))
        tk.Label(fl, text="LIVE DETECTION FEED",
                 font=("Courier New", 11, "bold"),
                 fg="#b0bec5", bg="#07080f").pack(side=tk.LEFT)
        tk.Label(fl,
                 text="best snapshot per person  ·  auto-updates",
                 font=("Courier New", 9), fg="#263238",
                 bg="#07080f").pack(side=tk.LEFT, padx=8)

        outer = tk.Frame(self.root, bg="#0d1117", bd=1,
                         relief=tk.FLAT)
        outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 8))
        self._canvas = tk.Canvas(outer, bg="#0d1117",
                                 highlightthickness=0)
        sb = tk.Scrollbar(outer, orient="vertical",
                          command=self._canvas.yview)
        self._feed = tk.Frame(self._canvas, bg="#0d1117")
        self._feed.bind("<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window(
            (0, 0), window=self._feed, anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._canvas.bind_all("<MouseWheel>",
            lambda e: self._canvas.yview_scroll(
                -1*(e.delta//120), "units"))

        tk.Label(self.root,
                 text=f"Database: {os.path.abspath(DB_PATH)}",
                 font=("Courier New", 8), fg="#1c2733",
                 bg="#07080f", anchor="w").pack(
                     fill=tk.X, padx=18, pady=(0, 4))
        self._feed_cards = {}

    def _refresh_ui(self):
        snap = self._pending.latest_per_person()
        db_count = (
            sum(1 for d in os.listdir(DB_PATH)
                if os.path.isdir(os.path.join(DB_PATH, d)))
            if os.path.isdir(DB_PATH) else 0
        )
        self._stat_vars["detected"].set(str(self._pending.count()))
        self._stat_vars["db_people"].set(str(db_count))
        self._stat_vars["queue"].set(
            str(self._brain.cache_counter.value))
        self._status_var.set(
            "✅ Running — leave it, I'm watching…"
            if self._pending.count() > 0 else
            "⏳ Waiting for first high-confidence match…"
        )
        for name, info in snap.items():
            if name not in self._feed_cards:
                self._add_card(name, info)
            else:
                self._feed_cards[name]["conf_var"].set(
                    f"{info['confidence']*100:.1f}%")
                try:
                    imgtk = _tk_img(info["cctv_img"], (90, 90))
                    self._feed_cards[name]["img_lbl"].configure(
                        image=imgtk)
                    self._feed_cards[name]["img_lbl"].image = imgtk
                except Exception:
                    pass
        if self._running:
            self.root.after(2500, self._refresh_ui)

    def _add_card(self, name, info):
        card = tk.Frame(self._feed, bg="#131a24",
                        bd=1, relief=tk.FLAT,
                        padx=8, pady=8)
        card.pack(side=tk.LEFT, padx=8, pady=8, anchor="nw")
        try:
            imgtk = _tk_img(info["cctv_img"], (90, 90))
            img_lbl = tk.Label(card, image=imgtk,
                               bg="#131a24",
                               bd=1, relief=tk.SOLID)
            img_lbl.image = imgtk
            img_lbl.pack()
        except Exception:
            img_lbl = tk.Label(card, text="📷",
                               font=("Courier New", 24),
                               bg="#131a24")
            img_lbl.pack()
        tk.Label(card, text=name,
                 font=("Courier New", 10, "bold"),
                 fg="white", bg="#131a24",
                 wraplength=100).pack(pady=(4, 0))
        conf_var = tk.StringVar(
            value=f"{info['confidence']*100:.1f}%")
        tk.Label(card, textvariable=conf_var,
                 font=("Courier New", 9),
                 fg="#ffeb3b", bg="#131a24").pack()
        tk.Label(card, text=info["timestamp"],
                 font=("Courier New", 8),
                 fg="#37474f", bg="#131a24").pack()
        self._feed_cards[name] = {
            "frame":    card,
            "conf_var": conf_var,
            "img_lbl":  img_lbl,
        }

    def _stop_and_review(self):
        self._stop_event.set()
        self._running = False
        self._stop_btn.configure(state=tk.DISABLED, text="⬛  Stopped")
        self._status_var.set("⏹ Stopped — opening review…")
        def _on_done():
            threading.Thread(target=brain.process_new_faces,
                             daemon=True).start()
            self._status_var.set(
                "✅ Database updated. You may close this window.")
        ReviewWindow(self.root, self._pending.snapshot(), _on_done)

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                     "Exit?",
                     "Stop the enroller without reviewing detections?"):
                return
        self._stop_event.set()
        self._running = False
        self.root.destroy()

#════════════════════════════════════════════════════════════
#ENTRY POINT
#════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = EnrollerApp(root)
    root.mainloop()