# ═══════════════════════════════════════════════════════════════
#  config.py  —  Single source of truth for all system settings
# ═══════════════════════════════════════════════════════════════

# ── Paths ────────────────────────────────────────────────────
DB_PATH     = "my_database"   # folder: my_database/<PersonName>/*.jpg
CACHE_DIR   = "session_cache" # temp folder for session snapshots

# ── AI / Recognition ─────────────────────────────────────────
MODEL_NAME      = "ArcFace"   # DeepFace backbone
DISTANCE_THRESH = 0.35        # cosine distance  (lower = stricter)
WAITING_TIME    = 60          # seconds before logging same person again

# ── Camera / YOLO ────────────────────────────────────────────
YOLO_MODEL              = "yolo8n-face.pt"
YOLO_CONF               = 0.50
YOLO_RUN_EVERY_N_FRAMES = 2
CAMERA_WIDTH            = 1280
CAMERA_HEIGHT           = 720

# ── Queue / Throughput ────────────────────────────────────────
TRACK_SEND_COOLDOWN = 2.5   # seconds between re-sending same track to AI
TRACK_EXPIRY_SECONDS = 5.0  # seconds before a lost track is cleaned up
MAX_QUEUE_SIZE       = 30   # max items in face_queue before dropping

# ── Voting / Confidence ──────────────────────────────────────
VOTE_WINDOW_SECONDS   = 6.0   # sliding window for recency-weighted voting
HIGH_CONF_LOCK_THRESH = 0.80  # instant-lock threshold (0.0–1.0)

# ── Low-light Enhancement ────────────────────────────────────
ENHANCE_LOW_LIGHT     = True
ENHANCE_MIN_BRIGHTNESS = 80   # skip enhancement if mean brightness > this

# ── Multi-camera sources ─────────────────────────────────────
# Add as many cameras as needed:  { "name": url_or_index }
CAMERAS = {
    "CAM-01": 0,           # default webcam
    # "CAM-02": "rtsp://192.168.1.101:554/stream",
}