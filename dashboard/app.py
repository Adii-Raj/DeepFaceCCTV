"""
dashboard/app.py
----------------
Flask dashboard server for Survil.
Single responsibility: routes only.
Reads data/detections.db and serves dashboard/static/crops/.
No detection logic, no ChromaDB writes, no Tkinter.

Run from project root:
    python dashboard/app.py

Or start it programmatically from launcher/service.py.
Accessible on LAN: http://<machine-ip>:5002
"""

from __future__ import annotations

import sqlite3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


from flask import Flask, jsonify, render_template, send_from_directory  # type: ignore[import]

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

# NEW - use absolute paths
_TEMPLATE_DIR = str(_PROJECT_ROOT / "dashboard" / "templates")
_STATIC_DIR = str(_PROJECT_ROOT / "dashboard" / "static")

app = Flask(
    __name__,
    template_folder=_TEMPLATE_DIR,
    static_folder=_STATIC_DIR,
)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_DEFAULT_DB = _PROJECT_ROOT / "data" / "detections.db"


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# DB reader
# ---------------------------------------------------------------------------
import sqlite3
from pathlib import Path

# ── Database path ────────────────────────────────────────────────────────────

def _db_path() -> Path:
    """Return the path to the detections SQLite database."""
    cfg = _load_config()
    return Path(cfg.get("detections_db", str(_DEFAULT_DB)))


# ── Read detections ───────────────────────────────────────────────────────────


def _read_detections(limit: int = 200) -> list[dict]:
    """
    Read the last `limit` rows from detections.db (SQLite).
    Returns list of dicts, newest first.
    """
    db_file = _db_path()
    if not db_file.exists():
        return []

    rows = []
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, video_time_s, name, confidence, status
            FROM detections
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        columns = ["timestamp", "video_time_s", "name", "confidence", "status"]
        for row in cursor.fetchall():
            row_dict = dict(zip(columns, row))
            row_dict["crop_url"] = ""  # Empty for compatibility
            rows.append(row_dict)

        conn.close()
    except Exception as e:
        print(f"[dashboard] Error reading detections DB: {e}")
        return []

    return rows


# ── Summary stats ───────────────────────────────────────────────────────────


def _summary_stats(rows: list[dict]) -> dict:
    """Compute summary counts from detection rows."""
    total = len(rows)
    known = sum(1 for r in rows if r.get("status", "").lower() == "known")
    unknown = total - known

    people: dict[str, int] = {}
    for r in rows:
        name = r.get("name", "unknown")
        if name and name.lower() not in ("unknown", "", "?"):
            people[name] = people.get(name, 0) + 1

    top_people = sorted(people.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total": total,
        "known": known,
        "unknown": unknown,
        "top_people": [{"name": n, "count": c} for n, c in top_people],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# new : absolute path
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detections")
def api_detections():
    """
    GET /api/detections
    Returns last 200 detections as JSON, newest first.
    Polled by the frontend every 5 seconds.
    """
    rows = _read_detections(200)
    stats = _summary_stats(rows)
    return jsonify(
        {
            "detections": rows,
            "stats": stats,
            "pipeline_log_exists": Path("pipeline.log").exists(),
            "db_exists": _db_path().exists(),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/api/stats")
def api_stats():
    """GET /api/stats — summary counts only (lighter than full detections)."""
    rows = _read_detections(500)
    return jsonify(_summary_stats(rows))


@app.route("/api/people")
def api_people():
    """
    GET /api/people
    Returns list of known people from ChromaDB (for gallery management).
    """
    try:
        from builder.db_ops import list_people

        cfg = _load_config()
        people = list_people(db_path=cfg.get("db_path", "data/face_db"))
        return jsonify({"people": people})
    except Exception as e:
        return jsonify({"people": [], "error": str(e)}), 200


@app.route("/api/log")
def api_log():
    """GET /api/log — last 80 lines of pipeline.log."""
    log_path = _PROJECT_ROOT / "pipeline.log"
    if not log_path.exists():
        return jsonify({"log": "(no log yet)"})
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return jsonify({"log": "".join(lines[-80:])})
    except Exception as e:
        return jsonify({"log": f"Error reading log: {e}"})


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate, public, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/debugfile")
def debugfile():
    import os

    path = os.path.join(app.root_path, "templates", "index.html")
    return {
        "template_path": path,
        "exists": os.path.exists(path),
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
    }


@app.route("/health")
def health():
    """
    Health check endpoint.
    Verifies server is alive and database is accessible.
    Returns unique people count (not total detections).
    """
    try:
        # Check database is accessible
        conn = sqlite3.connect(str(_db_path()), timeout=2)
        cursor = conn.cursor()
        
        # Count unique people (not total detections)
        cursor.execute("SELECT COUNT(DISTINCT name) FROM detections")
        unique_count = cursor.fetchone()[0]
        
        # Also get total detections for reference
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "status": "ok",
            "db": "connected",
            "unique_people": unique_count,   # Unique names, not total rows
            "total_detections": total_count,  # For reference
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "db": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_config()
    port = 5002

    print(f"Survil dashboard -> http://localhost:{port}")
    print(f"LAN access       -> http://<your-ip>:{port}")
    print("Press Ctrl+C to stop.")

    app.run(host="0.0.0.0", port=port, debug=True)
