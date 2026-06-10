"""
core/gallery.py
───────────────
Face gallery backed by ChromaDB with in-memory cache.

Single responsibility: given a query embedding, return the best matching
identity (name, score, status) by cosine similarity.

Architecture:
  - ChromaDB is the persistent store (written by builder/buildDataSet.py)
  - An in-memory numpy cache is kept for fast matching (dot product)
  - A background thread refreshes the cache every `refresh_interval` seconds
    so new people added via buildDataSet.py appear within ~60 seconds
    without restarting the detection process

Collection name default: "face_gallery"
ChromaDB path default  : "data/face_db"
Both are configurable via constructor args (read from config.json by pipeline.py)
"""

import threading
import time

import numpy as np

# ── Status constants ──────────────────────────────────────────────────────────

STATUS_KNOWN   = "known"
STATUS_UNSURE  = "unsure"
STATUS_UNKNOWN = "unknown"


# ── FaceGallery ───────────────────────────────────────────────────────────────

class FaceGallery:
    """
    ChromaDB-backed face gallery with auto-refreshing in-memory cache.

    Usage:
        gallery = FaceGallery(db_path="data/face_db")
        gallery.start()                        # starts background refresh
        name, score, status = gallery.match(embedding, 0.48, 0.32)
        gallery.stop()                         # call on shutdown
    """

    def __init__(
        self,
        db_path:          str   = "data/face_db",
        collection_name:  str   = "face_gallery",
        refresh_interval: int   = 60,
    ):
        """
        Args:
            db_path          : path to ChromaDB persistent storage directory
            collection_name  : ChromaDB collection name
            refresh_interval : seconds between cache refreshes (default 60)
        """
        self._db_path         = db_path
        self._collection_name = collection_name
        self._refresh_interval = refresh_interval

        # In-memory cache — protected by a read-write lock (threading.Lock)
        self._cache_lock   = threading.Lock()
        self._emb_arr:     np.ndarray | None = None   # shape (N, D) float32
        self._labels:      list[str]          = []
        self._names:       list[str]          = []    # unique sorted names

        # Background refresh thread
        self._stop_event   = threading.Event()
        self._refresh_thread: threading.Thread | None = None

        # ChromaDB client (lazy init)
        self._client     = None
        self._collection = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """
        Connect to ChromaDB, load initial cache, start background refresh.
        Call once before using match().
        """
        self._connect()
        self._refresh_cache()
        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="gallery-refresh",
        )
        self._refresh_thread.start()
        print(f"[gallery] Started - refresh every {self._refresh_interval}s")

    def stop(self):
        """Stop background refresh thread. Call on shutdown."""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        print("[gallery] Stopped.")

    # ── ChromaDB connection ───────────────────────────────────────────────────

    def _connect(self):
        try:
            import chromadb
        except ImportError:
            raise RuntimeError(
                "[gallery] chromadb is not installed.\n"
                "Run: pip install chromadb"
            )
        self._client = chromadb.PersistentClient(path=self._db_path)
        # get_or_create so it doesn't crash if collection doesn't exist yet
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[gallery] Connected to ChromaDB at '{self._db_path}' "
              f"collection='{self._collection_name}'")

    # ── Cache refresh ─────────────────────────────────────────────────────────

    def _refresh_cache(self):
        try:
            result = self._collection.get(include=["embeddings", "metadatas"])
            embs   = result.get("embeddings")
            metas  = result.get("metadatas") or []

            if embs is None or len(embs) == 0:
                with self._cache_lock:
                    self._emb_arr = None
                    self._labels  = []
                    self._names   = []
                print("[gallery] Cache refreshed - collection is empty")
                return

            emb_arr = np.array(embs, dtype=np.float32)
            if emb_arr.size == 0:
                with self._cache_lock:
                    self._emb_arr = None
                    self._labels  = []
                    self._names   = []
                print("[gallery] Cache refreshed - no embeddings")
                return

            labels = [m.get("name", "Unknown") for m in metas]

            # L2-normalise
            norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
            emb_arr = emb_arr / np.where(norms < 1e-10, 1e-10, norms)

            with self._cache_lock:
                self._emb_arr = emb_arr
                self._labels  = labels
                self._names   = sorted(set(labels))

            print(f"[gallery] Cache refreshed - {len(labels)} embeddings, {len(self._names)} identities")
        except Exception as e:
            print(f"[gallery] Cache refresh failed: {e}")

    def _refresh_loop(self):
        """Background thread: refresh cache every refresh_interval seconds."""
        while not self._stop_event.wait(timeout=self._refresh_interval):
            self._refresh_cache()

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(
        self,
        query:             np.ndarray,
        threshold_accept:  float,
        threshold_reject:  float,
        border_margin:     float = 0.05,
    ) -> tuple[str, float, str]:
        """
        Find the best matching identity for a query embedding.

        Args:
            query            : L2-normalised embedding (float32 vector)
            threshold_accept : cosine sim above which identity is confirmed (green)
            threshold_reject : cosine sim below which face is Unknown (red)
            border_margin    : scores within this margin above threshold_accept
                               are demoted to 'unsure' (catches borderline matches)

        Returns:
            (name, score, status)
            status ∈ {'known', 'unsure', 'unknown'}
        """
        with self._cache_lock:
            emb_arr = self._emb_arr
            labels  = self._labels

        if emb_arr is None or len(emb_arr) == 0:
            return "Unknown", 0.0, STATUS_UNKNOWN

        # L2-normalise query
        n = np.linalg.norm(query)
        if n < 1e-10:
            return "Unknown", 0.0, STATUS_UNKNOWN
        q = query / n

        # Cosine similarity via dot product (both sides L2-normalised)
        sims = emb_arr @ q
        idx  = int(np.argmax(sims))
        sim  = float(sims[idx])
        lbl  = labels[idx]

        if sim >= threshold_accept:
            # Borderline accept -> unsure
            if sim < threshold_accept + border_margin:
                return lbl, sim, STATUS_UNSURE
            return lbl, sim, STATUS_KNOWN
        if sim >= threshold_reject:
            return lbl, sim, STATUS_UNSURE
        return "Unknown", sim, STATUS_UNKNOWN

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def identity_count(self) -> int:
        """Number of unique identities currently in cache."""
        with self._cache_lock:
            return len(self._names)

    @property
    def embedding_count(self) -> int:
        """Total number of embeddings in cache."""
        with self._cache_lock:
            return len(self._labels)

    @property
    def names(self) -> list[str]:
        """Sorted list of unique identity names currently in cache."""
        with self._cache_lock:
            return list(self._names)

    def force_refresh(self):
        """Manually trigger a cache refresh outside the normal schedule."""
        self._refresh_cache()