"""
builder/buildDataSet.py
-----------------------
Tkinter UI for building and managing the face dataset.
Single responsibility: UI only.
All logic is delegated to:
    builder/embedder.py   — embedding extraction
    builder/db_ops.py     — ChromaDB read/write

Entry point: open_build_dataset_window(parent)
Called from launcher/launcher.py when user clicks "Build Dataset".
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

logger = logging.getLogger(__name__)

# Project config defaults
_DEFAULT_DB_PATH = "data/face_db"
_DEFAULT_MODEL_DIR = "models/buffalo_l"
_DEFAULT_COLLECTION = "face_gallery"
_MAX_IMAGES_PER_PERSON = 10


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BuildDatasetWindow(tk.Toplevel):
    """
    Modal-style window for building / managing the face gallery.

    Layout
    ------
    ┌──────────────────────────────────┐
    │  Mode selector (Create/Update)   │
    ├──────────────────────────────────┤
    │  [CREATE mode panel]              │
    │  [UPDATE mode panel]              │
    ├──────────────────────────────────┤
    │  GPU / Processing options        │
    ├──────────────────────────────────┤
    │  Progress bar + log              │
    ├──────────────────────────────────┤
    │  Action buttons                  │
    └──────────────────────────────────┘
    """

    def __init__(
        self,
        parent: tk.Misc,
        db_path: str = _DEFAULT_DB_PATH,
        model_dir: str = _DEFAULT_MODEL_DIR,
    ):
        super().__init__(parent)
        self.title("Build Dataset — Survil")
        self.resizable(True, True)
        self.minsize(680, 750)
        self.grab_set()

        self._db_path = db_path
        self._model_dir = model_dir
        self._embedder = None
        self._worker_thread = None
        self._is_running = False

        # GPU state
        self._gpu_status = None
        self._gpu_check_done = False
        self._use_gpu = tk.BooleanVar(value=False)

        # Mode & variables
        self._mode = tk.StringVar(value="create")
        self._folder_var = tk.StringVar(value="")
        self._db_create_name = tk.StringVar(value="")
        self._db_select_var = tk.StringVar(value="")
        self._people_vars: dict[str, tk.BooleanVar] = {}

        self._build_ui()
        self._refresh_databases()

        # Auto-run GPU check
        threading.Thread(target=self._run_gpu_check, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = {"padx": 10, "pady": 6}
        
        # ── Mode selector ────────────────────────────────────────────────────
        mode_frame = ttk.LabelFrame(self, text="Action", padding=10)
        mode_frame.grid(row=0, column=0, sticky="ew", **PAD)
        mode_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            mode_frame, text="Create new database",
            variable=self._mode, value="create",
            command=self._on_mode_change,
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            mode_frame, text="Update existing database",
            variable=self._mode, value="update",
            command=self._on_mode_change,
        ).grid(row=0, column=1, sticky="w")

        # ── CREATE panel ─────────────────────────────────────────────────────
        self._create_frame = ttk.LabelFrame(
            self, text="Create New Database", padding=10)
        self._create_frame.grid(row=1, column=0, sticky="ew", **PAD)
        self._create_frame.columnconfigure(1, weight=1)

        ttk.Label(self._create_frame, text="Database name:").grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self._create_frame, textvariable=self._db_create_name,
                  width=30).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(self._create_frame, text="Photos root folder:").grid(
            row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self._create_frame, textvariable=self._folder_var,
                  width=30, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(self._create_frame, text="Browse…",
                   command=self._browse_folder).grid(
            row=1, column=2, padx=4, pady=4)

        ttk.Label(
            self._create_frame,
            text="Folder should contain one subfolder per person.",
            foreground="gray", font=("", 8),
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8)

        # ── UPDATE panel ─────────────────────────────────────────────────────
        self._update_frame = ttk.LabelFrame(
            self, text="Update Existing Database", padding=10)
        self._update_frame.columnconfigure(1, weight=1)

        ttk.Label(self._update_frame, text="Select database:").grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        self._db_combo = ttk.Combobox(
            self._update_frame, textvariable=self._db_select_var,
            state="readonly", width=30)
        self._db_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self._db_combo.bind("<<ComboboxSelected>>", self._on_db_selected)

        ttk.Button(self._update_frame, text="↻ Refresh",
                   command=self._refresh_databases).grid(
            row=0, column=2, padx=4, pady=4)
        ttk.Button(self._update_frame, text="🗑 Delete DB",
                   command=self._delete_database).grid(
            row=0, column=3, padx=4, pady=4)

        ttk.Label(self._update_frame, text="People in database:").grid(
            row=1, column=0, sticky="nw", padx=8, pady=(10, 0))

        people_outer = ttk.Frame(self._update_frame)
        people_outer.grid(row=1, column=1, columnspan=3, sticky="nsew",
                          padx=4, pady=(10, 0))
        people_outer.columnconfigure(0, weight=1)
        people_outer.rowconfigure(0, weight=1)

        self._people_canvas = tk.Canvas(
            people_outer, height=100, width=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            people_outer, orient="vertical",
            command=self._people_canvas.yview)
        self._people_canvas.configure(yscrollcommand=scrollbar.set)
        self._people_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self._people_inner = ttk.Frame(self._people_canvas)
        self._people_canvas.create_window(
            (0, 0), window=self._people_inner, anchor="nw")
        self._people_inner.bind(
            "<Configure>",
            lambda e: self._people_canvas.configure(
                scrollregion=self._people_canvas.bbox("all")))

        self._no_people_label = ttk.Label(
            self._people_inner, text="← Select a database first",
            foreground="gray")
        self._no_people_label.pack(anchor="w", padx=4, pady=4)

        sel_frame = ttk.Frame(self._update_frame)
        sel_frame.grid(row=2, column=1, columnspan=2, sticky="w",
                       padx=4, pady=(4, 0))
        ttk.Button(sel_frame, text="Select all",
                   command=self._select_all_people).pack(
            side="left", padx=(0, 6))
        ttk.Button(sel_frame, text="Clear all",
                   command=self._clear_all_people).pack(
            side="left", padx=(0, 6))
        ttk.Button(sel_frame, text="🗑 Delete Selected",
                   command=self._delete_selected_people).pack(side="left")

        ttk.Label(self._update_frame, text="Photos root folder:").grid(
            row=3, column=0, sticky="w", padx=8, pady=(10, 0))
        ttk.Entry(self._update_frame, textvariable=self._folder_var,
                  width=30, state="readonly").grid(
            row=3, column=1, sticky="ew", padx=4, pady=(10, 0))
        ttk.Button(self._update_frame, text="Browse…",
                   command=self._browse_folder).grid(
            row=3, column=2, padx=4, pady=(10, 0))

        # ── GPU section ──────────────────────────────────────────────────────
        gpu_frame = ttk.LabelFrame(self, text="Processing Device", padding=10)
        gpu_frame.grid(row=2, column=0, sticky="ew", **PAD)
        gpu_frame.columnconfigure(1, weight=1)

        self._gpu_check = ttk.Checkbutton(
            gpu_frame,
            text="Use GPU (CUDA) — faster for large datasets",
            variable=self._use_gpu,
            state="disabled",
        )
        self._gpu_check.grid(row=0, column=0, sticky="w", padx=8)

        ttk.Button(
            gpu_frame, text="Check GPU",
            command=self._on_check_gpu_clicked, width=12).grid(
            row=0, column=1, sticky="e", padx=8)

        self._gpu_status_label = ttk.Label(
            gpu_frame,
            text="⏳ Checking GPU…",
            foreground="gray", font=("", 8),
            wraplength=600, justify="left",
        )
        self._gpu_status_label.grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0))

        # ── Progress section ─────────────────────────────────────────────────
        prog_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        prog_frame.grid(row=3, column=0, sticky="ew", **PAD)
        prog_frame.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame, variable=self._progress_var, maximum=100)
        self._progress_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        log_frame = ttk.Frame(prog_frame)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self._log_text = tk.Text(
            log_frame, height=8, width=80,
            state="disabled", wrap="word",
            font=("Courier", 9),
            background="#1e1e1e", foreground="#d4d4d4", relief="flat")
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        btn_frame.grid(row=4, column=0, sticky="e")

        self._run_btn = ttk.Button(
            btn_frame, text="▶  Build", command=self._run)
        self._run_btn.pack(side="right", padx=(6, 0))

        ttk.Button(btn_frame, text="Close",
                   command=self.destroy).pack(side="right")

        self._on_mode_change()

    # ──────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        """Switch between CREATE and UPDATE panels."""
        mode = self._mode.get()
        if mode == "create":
            self._update_frame.grid_remove()
            self._create_frame.grid(row=1, column=0, sticky="ew",
                                     padx=10, pady=6)
            self._run_btn.configure(text="▶  Create Database")
        else:
            self._create_frame.grid_remove()
            self._update_frame.grid(row=1, column=0, sticky="ew",
                                     padx=10, pady=6)
            self._run_btn.configure(text="▶  Update Database")
        self._folder_var.set("")

    def _browse_folder(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title="Select photos root folder")
        if folder:
            self._folder_var.set(folder)

    def _on_db_selected(self, event=None):
        """Load people when database is selected."""
        db_name = self._db_select_var.get()
        if db_name:
            self._load_people_for_db(db_name)

    def _load_people_for_db(self, db_name: str):
        """Populate people checkboxes for selected database."""
        self._clear_people_checkboxes()
        try:
            from builder.db_ops import list_people
            people = list_people(db_path=self._db_path)
            people = [p["name"] for p in people]
        except Exception as e:
            self._log(f"❌ Could not load people: {e}")
            return

        self._no_people_label.pack_forget()
        self._people_vars.clear()

        for person in sorted(people):
            var = tk.BooleanVar(value=False)
            self._people_vars[person] = var
            ttk.Checkbutton(
                self._people_inner, text=person, variable=var
            ).pack(anchor="w", padx=4, pady=2)

        if not people:
            ttk.Label(self._people_inner, text="(database is empty)",
                      foreground="gray").pack(anchor="w", padx=4, pady=4)

    def _clear_people_checkboxes(self):
        """Clear and reset people list."""
        for widget in self._people_inner.winfo_children():
            widget.destroy()
        self._no_people_label = ttk.Label(
            self._people_inner, text="← Select a database first",
            foreground="gray")
        self._no_people_label.pack(anchor="w", padx=4, pady=4)
        self._people_vars.clear()

    def _select_all_people(self):
        for var in self._people_vars.values():
            var.set(True)

    def _clear_all_people(self):
        for var in self._people_vars.values():
            var.set(False)

    def _refresh_databases(self):
        """Refresh the database list."""
        try:
            from builder.db_ops import list_all_collections
            dbs = list_all_collections(db_path=self._db_path)
            self._db_combo["values"] = dbs
            if dbs:
                self._db_combo.set(dbs[0])
                self._load_people_for_db(dbs[0])
        except Exception as e:
            self._log(f"❌ Could not list databases: {e}")

    def _delete_database(self):
        """Delete entire database with confirmation."""
        db_name = self._db_select_var.get()
        if not db_name:
            messagebox.showwarning("No database",
                "Please select a database first.", parent=self)
            return

        if not messagebox.askyesno("Delete entire database?",
                f"Permanently delete database:\n\n  '{db_name}'\n\n"
                "ALL people and vectors will be removed.\n"
                "This cannot be undone.",
                parent=self):
            return

        try:
            from builder.db_ops import delete_entire_collection
            delete_entire_collection(db_name, db_path=self._db_path)
            self._log(f"🗑  Database '{db_name}' deleted.")
            self._refresh_databases()
        except Exception as e:
            messagebox.showerror("Delete failed", str(e), parent=self)

    def _delete_selected_people(self):
        """Delete selected people from database."""
        db_name = self._db_select_var.get()
        if not db_name:
            messagebox.showwarning("No database",
                "Please select a database first.", parent=self)
            return

        selected = [n for n, v in self._people_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Nothing selected",
                "Please tick people you want to delete.", parent=self)
            return

        preview = ", ".join(selected[:3]) + ("…" if len(selected) > 3 else "")
        if not messagebox.askyesno("Confirm delete",
                f"Delete {len(selected)} person(s):\n\n  {preview}\n\n"
                "This cannot be undone.",
                parent=self):
            return

        try:
            from builder.db_ops import delete_multiple_people
            count = delete_multiple_people(selected, db_path=self._db_path)
            self._log(f"🗑  Deleted {count} person(s).")
            self._load_people_for_db(db_name)
        except Exception as e:
            messagebox.showerror("Delete failed", str(e), parent=self)

    # ──────────────────────────────────────────────────────────────────────────
    # GPU Check
    # ──────────────────────────────────────────────────────────────────────────

    def _run_gpu_check(self):
        """Run GPU check in background thread."""
        from builder.embedder import check_gpu_support, GPU_MESSAGES, GPU_OK

        code, detail = check_gpu_support()
        self._gpu_status = code

        def _update_ui():
            msg = GPU_MESSAGES.get(code, detail)
            if code == GPU_OK:
                self._gpu_status_label.configure(
                    text=msg, foreground="#1a7a1a")
                self._gpu_check.configure(state="normal")
            else:
                self._gpu_status_label.configure(
                    text=msg, foreground="#c0504d")
                self._use_gpu.set(False)
                self._gpu_check.configure(state="disabled")

        self.after(0, _update_ui)

    def _on_check_gpu_clicked(self):
        """Manual GPU check."""
        self._use_gpu.set(False)
        threading.Thread(target=self._run_gpu_check, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Logging & Progress
    # ──────────────────────────────────────────────────────────────────────────

    def _log(self, text: str):
        """Thread-safe logging to text widget."""
        def _append():
            self._log_text.configure(state="normal")
            self._log_text.insert("end", text + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        self.after(0, _append)

    def _set_progress(self, value: float):
        """Thread-safe progress update."""
        self.after(0, lambda: self._progress_var.set(value))

    # ──────────────────────────────────────────────────────────────────────────
    # Build / Update workers
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self):
        """Dispatch to CREATE or UPDATE worker."""
        if self._is_running:
            messagebox.showwarning("Already running",
                "A build/update is already in progress.", parent=self)
            return

        if self._mode.get() == "create":
            self._run_create()
        else:
            self._run_update()

    def _run_create(self):
        """Validate and start CREATE NEW DATABASE build."""
        db_name = self._db_create_name.get().strip()
        folder = self._folder_var.get().strip()

        if not db_name:
            messagebox.showwarning("Missing name",
                "Please enter a database name.", parent=self)
            return
        if not folder or not Path(folder).is_dir():
            messagebox.showwarning("Invalid folder",
                "Please select a valid photos folder.", parent=self)
            return

        # Check if folder has subfolders
        subdirs = [d for d in Path(folder).iterdir() if d.is_dir()]
        if not subdirs:
            messagebox.showwarning("No subfolders",
                "Folder must contain subfolders (one per person).", parent=self)
            return

        # Check if database exists
        from builder.db_ops import list_all_collections
        existing = list_all_collections(db_path=self._db_path)
        if db_name in existing:
            if not messagebox.askyesno("Database exists",
                    f"'{db_name}' already exists. Overwrite?",
                    parent=self):
                return

        self._is_running = True
        self._run_btn.configure(state="disabled")
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._progress_var.set(0)

        self._worker_thread = threading.Thread(
            target=self._create_worker,
            args=(db_name, folder),
            daemon=True,
        )
        self._worker_thread.start()

    def _create_worker(self, db_name: str, folder: str):
        """Worker thread for CREATE NEW DATABASE."""
        try:
            from builder.embedder import ArcFaceEmbedder, extract_embeddings_from_folder
            from builder.db_ops import add_embeddings, delete_entire_collection

            self._log(f"── Creating database: '{db_name}' ──")

            # Delete if exists
            try:
                delete_entire_collection(db_name, db_path=self._db_path)
                self._log(f"Cleared existing '{db_name}'.")
            except:
                pass

            # Initialize embedder
            if self._embedder is None:
                self._log("Loading InsightFace model…")
                self._embedder = ArcFaceEmbedder(
                    model_dir=self._model_dir,
                    use_gpu=self._use_gpu.get()
                )

            # Process each person folder
            person_dirs = sorted([
                d for d in Path(folder).iterdir() if d.is_dir()
            ])
            self._log(f"Found {len(person_dirs)} person folder(s).\n")

            ok_count = failed_count = total_emb = 0

            for idx, person_dir in enumerate(person_dirs):
                person_name = person_dir.name
                images = [
                    f for f in person_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]

                if not images:
                    self._log(f"  SKIP  '{person_name}' — no images")
                    failed_count += 1
                    self._set_progress((idx + 1) / len(person_dirs) * 100)
                    continue

                def progress_cb(cur, tot, path):
                    pct = (idx + (cur / tot)) / len(person_dirs) * 100
                    self._set_progress(pct)

                embeddings, img_paths, failed = extract_embeddings_from_folder(
                    person_dir,
                    self._embedder,
                    max_images=_MAX_IMAGES_PER_PERSON,
                    progress_callback=progress_cb,
                )

                if not embeddings:
                    self._log(
                        f"  SKIP  '{person_name}' — no valid faces detected "
                        f"({len(failed)} image(s))")
                    failed_count += 1
                else:
                    add_embeddings(
                        name=person_name,
                        embeddings=[e.tolist() for e in embeddings],
                        img_paths=img_paths,
                        db_path=self._db_path,
                    )
                    self._log(
                        f"  OK    '{person_name}' — {len(embeddings)} embedding(s) added "
                        f"({len(failed)} image(s) skipped)")
                    ok_count += 1
                    total_emb += len(embeddings)

                self._set_progress((idx + 1) / len(person_dirs) * 100)

            self._log("\n" + "=" * 50)
            self._log(f"Database    : {db_name}")
            self._log(f"People added: {ok_count}")
            self._log(f"People skip : {failed_count}")
            self._log(f"Total embed : {total_emb}")
            self._log("=" * 50)
            self._log("✅ Done!")

        except Exception as e:
            logger.exception("Create worker failed")
            self._log(f"\n❌ ERROR: {e}")
        finally:
            self._is_running = False
            self.after(0, lambda: self._run_btn.configure(state="normal"))
            self._refresh_databases()

    def _run_update(self):
        """Validate and start UPDATE DATABASE."""
        db_name = self._db_select_var.get().strip()
        folder = self._folder_var.get().strip()

        if not db_name:
            messagebox.showwarning("No database",
                "Please select a database.", parent=self)
            return
        if not folder or not Path(folder).is_dir():
            messagebox.showwarning("Invalid folder",
                "Please select a valid photos folder.", parent=self)
            return

        selected = [n for n, v in self._people_vars.items() if v.get()]
        all_dirs = {d.name for d in Path(folder).iterdir() if d.is_dir()}

        from builder.db_ops import list_people
        existing_names = {p["name"] for p in list_people(db_path=self._db_path)}
        new_people = [n for n in all_dirs if n not in existing_names]

        if not selected and not new_people:
            messagebox.showwarning("Nothing to do",
                "No people selected and no new person folders found.",
                parent=self)
            return

        self._is_running = True
        self._run_btn.configure(state="disabled")
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._progress_var.set(0)

        self._worker_thread = threading.Thread(
            target=self._update_worker,
            args=(db_name, folder, selected, all_dirs),
            daemon=True,
        )
        self._worker_thread.start()

    def _update_worker(self, db_name: str, folder: str, selected_people: list, all_dirs: set):
        """Worker thread for UPDATE DATABASE."""
        try:
            from builder.embedder import ArcFaceEmbedder, extract_embeddings_from_folder
            from builder.db_ops import add_embeddings, delete_person, list_people

            self._log(f"── Updating database: '{db_name}' ──")

            # Initialize embedder
            if self._embedder is None:
                self._log("Loading InsightFace model…")
                self._embedder = ArcFaceEmbedder(
                    model_dir=self._model_dir,
                    use_gpu=self._use_gpu.get()
                )

            existing_in_db = {p["name"] for p in list_people(db_path=self._db_path)}
            selected_set = set(selected_people)

            person_dirs = sorted([
                d for d in Path(folder).iterdir() if d.is_dir()
            ])
            self._log(f"Found {len(person_dirs)} subfolder(s).\n")

            updated = added = skipped = 0

            for idx, person_dir in enumerate(person_dirs):
                person_name = person_dir.name
                images = [
                    f for f in person_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]

                if not images:
                    self._log(f"  SKIP  '{person_name}' — folder is empty")
                    skipped += 1
                    self._set_progress((idx + 1) / len(person_dirs) * 100)
                    continue

                is_existing = person_name in existing_in_db
                is_selected = person_name in selected_set

                # Decide: skip or process
                if is_existing and not is_selected:
                    self._log(f"  SKIP  '{person_name}' — exists but not selected")
                    skipped += 1
                    self._set_progress((idx + 1) / len(person_dirs) * 100)
                    continue

                def progress_cb(cur, tot, path):
                    pct = (idx + (cur / tot)) / len(person_dirs) * 100
                    self._set_progress(pct)

                embeddings, img_paths, failed = extract_embeddings_from_folder(
                    person_dir,
                    self._embedder,
                    max_images=_MAX_IMAGES_PER_PERSON,
                    progress_callback=progress_cb,
                )

                if not embeddings:
                    self._log(f"  SKIP  '{person_name}' — no valid faces detected")
                    skipped += 1
                    self._set_progress((idx + 1) / len(person_dirs) * 100)
                    continue

                if is_existing and is_selected:
                    # Replace old embeddings
                    delete_person(person_name, db_path=self._db_path)
                    add_embeddings(
                        name=person_name,
                        embeddings=[e.tolist() for e in embeddings],
                        img_paths=img_paths,
                        db_path=self._db_path,
                    )
                    self._log(f"  UPD   '{person_name}' — {len(embeddings)} embedding(s) "
                            f"({len(failed)} skipped)")
                    updated += 1
                else:
                    # New person
                    add_embeddings(
                        name=person_name,
                        embeddings=[e.tolist() for e in embeddings],
                        img_paths=img_paths,
                        db_path=self._db_path,
                    )
                    self._log(f"  NEW   '{person_name}' — {len(embeddings)} embedding(s) "
                            f"({len(failed)} skipped)")
                    added += 1

                self._set_progress((idx + 1) / len(person_dirs) * 100)

            self._log("\n" + "=" * 50)
            self._log(f"Database   : {db_name}")
            self._log(f"Updated    : {updated}")
            self._log(f"New added  : {added}")
            self._log(f"Skipped    : {skipped}")
            self._log("=" * 50)
            self._log("✅ Done!")

        except Exception as e:
            logger.exception("Update worker failed")
            self._log(f"\n❌ ERROR: {e}")
        finally:
            self._is_running = False
            self.after(0, lambda: self._run_btn.configure(state="normal"))
            self._refresh_databases()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def open_build_dataset_window(
    parent: tk.Misc = None,
    db_path: str = _DEFAULT_DB_PATH,
    model_dir: str = _DEFAULT_MODEL_DIR,
) -> BuildDatasetWindow:
    """
    Open the Build Dataset window as a child of parent.

    Usage in launcher.py:
        from builder.buildDataSet import open_build_dataset_window
        open_build_dataset_window(parent=root)
    """
    win = BuildDatasetWindow(parent, db_path=db_path, model_dir=model_dir)
    return win


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    root.withdraw()
    win = open_build_dataset_window(root)
    root.wait_window(win)