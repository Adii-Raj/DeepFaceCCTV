"""
dashboard/app.py
----------------
Flask dashboard server for Survil.
Single responsibility: routes only.
Reads data/detections.csv and serves dashboard/static/crops/.
No detection logic, no ChromaDB writes, no Tkinter.

Run from project root:
    python dashboard/app.py

Or start it programmatically from launcher/service.py.
Accessible on LAN: http://<machine-ip>:5000
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "detections.csv"
_DEFAULT_CROPS = _PROJECT_ROOT / "dashboard" / "static" / "crops"


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _csv_path() -> Path:
    cfg = _load_config()
    return Path(cfg.get("detections_csv", str(_DEFAULT_CSV)))


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_detections(limit: int = 200) -> list[dict]:
    """
    Read the last `limit` rows from detections.csv.
    Returns list of dicts, newest first.

    Expected CSV columns (written by core/logger.py):
        timestamp, name, score, status, crop_path
    """
    path = _csv_path()
    if not path.exists():
        return []

    rows = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return []

    # Newest first, capped at limit
    rows = rows[-limit:][::-1]

    # Normalise crop_path to a URL-friendly path under /static/crops/
    for row in rows:
        crop = row.get("crop_path", "")
        if crop:
            crop_file = Path(crop).name
            row["crop_url"] = f"/static/crops/{crop_file}"
        else:
            row["crop_url"] = ""

    return rows


def _summary_stats(rows: list[dict]) -> dict:
    """Compute summary counts from detection rows."""
    total = len(rows)
    known = sum(1 for r in rows if r.get("status", "").lower() == "known")
    unknown = total - known

    people: dict[str, int] = {}
    for r in rows:
        name = r.get("name", "unknown")
        if name and name.lower() not in ("unknown", ""):
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

@app.route("/")
def index():
    """Main dashboard page."""
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
    return jsonify({
        "detections": rows,
        "stats": stats,
        "pipeline_log_exists": Path("pipeline.log").exists(),
        "csv_exists": _csv_path().exists(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


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


@app.route("/static/crops/<path:filename>")
def serve_crop(filename):
    """Serve face crop images saved by core/logger.py."""
    crops_dir = str(_DEFAULT_CROPS)
    return send_from_directory(crops_dir, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_config()
    port = cfg.get("flask_port", 5000)

    print(f"Survil dashboard -> http://localhost:{port}")
    print(f"LAN access       -> http://<your-ip>:{port}")
    print("Press Ctrl+C to stop.")

    app.run(host="0.0.0.0", port=port, debug=False)