import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import cv2

import numpy as np

import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except Exception:
    pass  # Fallback: use default encoding on Windows console

# ── Add project root to path so 'core' imports work ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.detector    import YuNetDetector
from core.recogniser  import load_recogniser, InsightFaceRecogniser
from core.tracker     import FaceTracker, POSE_SKIP, QUALITY_SKIP, SKIP_TOKENS
from core.gallery     import FaceGallery
from core.quality     import passes_quality_gates, resolve_vote
from core.drawing     import (draw_face_label, draw_hud,
                               draw_loading_overlay, draw_reconnect_overlay)
from core.logger      import DetectionLogger

# ── Global stop event (set externally to stop run() without killing process) ──
_global_stop = threading.Event()

# ── Embed thread pool (1 worker — serial embedding avoids race conditions) ────
_embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

# ── Default config values ─────────────────────────────────────────────────────
DEFAULTS = {
    "rtsp":               None,
    "video":              None,
    "transport":          "tcp",
    "headless":           False,
    "db_path":            "data/face_db",
    "collection_name":    "face_gallery",
    "refresh_interval":   60,
    "output_db":         "data/detections.db",
    "yunet_model":        "models/face_detection_yunet_2023mar.onnx",
    "sface_model":        "models/face_recognition_sface_2021dec.onnx",
    "threshold_accept":   0.48,
    "threshold_reject":   0.32,
    "border_margin":      0.05,
    "max_yaw":            45.0,
    "min_face_size":      25,
    "blur_threshold":     20.0,
    "det_scale":          0.75,
    "det_confidence":     0.55,
    "recognition_interval": 4,
    "log_cooldown":       3.0,
}


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.json") -> dict:
    """
    Load config.json. Returns DEFAULTS merged with file values.
    Missing keys fall back to DEFAULTS silently.
    """
    cfg = dict(DEFAULTS)
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            cfg.update(file_cfg)
            print(f"[pipeline] Config loaded from {config_path}")
        except Exception as e:
            print(f"[pipeline] Config read error ({config_path}): {e} — using defaults")
    else:
        print(f"[pipeline] No config.json found at '{config_path}' — using defaults")
    return cfg


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Survil — CCTV Face Identification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     default="config.json",
                   help="Path to config.json")

    # Source
    src = p.add_mutually_exclusive_group()
    src.add_argument("--rtsp",  help="RTSP URL")
    src.add_argument("--video", help="Video file path or 0 for webcam")
    p.add_argument("--transport", choices=["tcp", "udp", "auto"],
                   help="RTSP transport protocol")

    # Mode
    p.add_argument("--headless", action="store_true",
                   help="Run without cv2.imshow (server / background mode)")

    # ChromaDB
    p.add_argument("--db-path",         help="ChromaDB persistent storage path")
    p.add_argument("--collection-name", help="ChromaDB collection name")
    p.add_argument("--refresh-interval", type=int,
                   help="Gallery cache refresh interval (seconds)")

    # Output
    p.add_argument("--output-db",  help="Path to detections db")

    # Models
    p.add_argument("--yunet-model", help="Path to YuNet ONNX model")
    p.add_argument("--sface-model", help="Path to SFace ONNX model (fallback)")

    # Thresholds
    p.add_argument("--threshold-accept", type=float)
    p.add_argument("--threshold-reject", type=float)
    p.add_argument("--border-margin",    type=float)
    p.add_argument("--max-yaw",          type=float)
    p.add_argument("--min-face-size",    type=int)
    p.add_argument("--blur-threshold",   type=float)
    p.add_argument("--det-scale",        type=float)
    p.add_argument("--det-confidence",   type=float)
    p.add_argument("--recognition-interval", type=int)

    return p.parse_args(args)


def _merge_args_into_config(args: argparse.Namespace, cfg: dict) -> dict:
    """
    CLI args override config.json values.
    Only override if the CLI arg was explicitly provided (not None).
    """
    mapping = {
        "rtsp":                 args.rtsp,
        "video":                args.video,
        "transport":            args.transport,
        "headless":             True if args.headless else None,
        "db_path":              args.db_path,
        "collection_name":      args.collection_name,
        "refresh_interval":     args.refresh_interval,
        "output_db":            args.output_db,
        "yunet_model":          args.yunet_model,
        "sface_model":          args.sface_model,
        "threshold_accept":     args.threshold_accept,
        "threshold_reject":     args.threshold_reject,
        "border_margin":        args.border_margin,
        "max_yaw":              args.max_yaw,
        "min_face_size":        args.min_face_size,
        "blur_threshold":       args.blur_threshold,
        "det_scale":            args.det_scale,
        "det_confidence":       args.det_confidence,
        "recognition_interval": args.recognition_interval,
    }
    for key, val in mapping.items():
        if val is not None:
            cfg[key] = val
    return cfg


# ── Video capture ─────────────────────────────────────────────────────────────

def _open_capture(src, is_rtsp: bool, transport: str) -> cv2.VideoCapture:
    if is_rtsp:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{transport}|buffer_size;0|"
            "stimeout;5000000|analyzeduration;500000|probesize;131072",
        )
        cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    else:
        src_id = int(src) if str(src).isdigit() else src
        cap = cv2.VideoCapture(src_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _build_source(cfg: dict):
    """Returns (src, is_rtsp, label)."""
    if cfg.get("rtsp"):
        return cfg["rtsp"], True, cfg["rtsp"]
    src = cfg.get("video", "0")
    is_rtsp = isinstance(src, str) and src.lower().startswith("rtsp://")
    return src, is_rtsp, str(src)


# ── Black frame detection ──────────────────────────────────────────────────────

def _is_frame_black(frame: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Detect if a frame is mostly black (all pixels near 0).
    Indicates network lag / stream dropout.
    
    Args:
        frame: BGR frame
        threshold: mean pixel value above which frame is NOT black (0-255)
    
    Returns:
        True if frame is black/near-black
    """
    if frame is None or frame.size == 0:
        return True
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(grey.mean())
    return mean_val < threshold


# ── Capture worker (thread) ───────────────────────────────────────────────────

def _capture_worker(src, is_rtsp, transport, fq, stop, reconnect_event):
    backoff = 1.0
    consecutive_black_frames = 0
    MAX_BLACK_FRAMES = 10  # Trigger reconnect after 10 consecutive black frames

    cap = _open_capture(src, is_rtsp, transport)
    if not cap.isOpened():
        print(f"[pipeline] ERROR: Cannot open source: {src}")
        stop.set()
        return
    print(f"[pipeline] Capture opened - FPS:{cap.get(cv2.CAP_PROP_FPS):.1f}")

    while not stop.is_set():
        ret, frame = cap.read()

        if not ret or _is_frame_black(frame):
            consecutive_black_frames += 1

            if ret and _is_frame_black(frame):
                # Frame read OK but is black - network lag
                if consecutive_black_frames == 1:
                    print(f"[pipeline] Black frames detected (network lag) …")
            else:
                # Frame read failed
                if not is_rtsp and str(src) != "0":
                    print("[pipeline] End of file.")
                    stop.set()
                    break

            if consecutive_black_frames >= MAX_BLACK_FRAMES:
                if not reconnect_event.is_set():
                    reconnect_event.set()
                    print(f"[pipeline] Stream stalled. Reconnecting in {backoff:.0f}s …")
                cap.release()
                time.sleep(backoff)
                backoff = min(backoff * 2, 16.0)
                cap = _open_capture(src, is_rtsp, transport)
                if cap.isOpened():
                    reconnect_event.clear()
                    backoff = 1.0
                    consecutive_black_frames = 0
                    print("[pipeline] Stream reconnected.")
                continue

            # Skip putting black frame in queue
            continue

        # Good frame - reset black frame counter and backoff
        consecutive_black_frames = 0
        backoff = 1.0
        vt = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        try:
            fq.put_nowait((frame, vt))
        except queue.Full:
            pass

    cap.release()


# ── Embed job (runs in thread pool) ──────────────────────────────────────────


def _submit_embed(
    recogniser,
    use_insightface,
    frame_snap,
    full_row,
    disp_w,
    disp_h,
    track,
    gallery,
    cfg,
    logger,
):
    """
    Extract embedding and cast a vote on the track.
    Runs in _embed_pool (off the main process thread).
    """
    try:
        # Extract crop for embedding ONLY (not saved)
        x, y, w, h = [int(v) for v in full_row[:4]]
        if use_insightface:
            px = int(w * 0.20)
            py = int(h * 0.20)
            crop = frame_snap[
                max(0, y - py) : min(disp_h, y + h + py),
                max(0, x - px) : min(disp_w, x + w + px),
            ]
        else:
            crop = frame_snap[
                max(0, y) : min(disp_h, y + h),
                max(0, x) : min(disp_w, x + w),
            ]

        # Quality gates — check if face is good enough
        passes, skip_reason = passes_quality_gates(
            full_row,
            crop,
            max_yaw=cfg["max_yaw"],
            min_face_size=cfg["min_face_size"],
            blur_threshold=cfg["blur_threshold"],
        )
        if not passes:
            track.votes.append((skip_reason, "skip"))
            return

        # Embed — convert face crop to 512-number fingerprint
        if use_insightface:
            emb = recogniser.embed(crop) if crop.size > 0 else np.zeros(512, np.float32)
        else:
            emb = recogniser.embed_with_landmarks(frame_snap, full_row)

        # Match against known people in gallery
        name, score, status = gallery.match(
            emb,
            cfg["threshold_accept"],
            cfg["threshold_reject"],
            border_margin=cfg["border_margin"],
        )
        track.score = score

        # Vote system — builds confidence over multiple frames
        track.votes.append((name, status))
        d_name, d_status = resolve_vote(track.votes)
        track.display_name = d_name
        track.display_status = d_status

        # Log to db — NO crop image passed, NO crop saved
        logger.log(d_name, score, getattr(track, "_last_vt", 0.0), d_status)

    except Exception as e:
        print(f"[pipeline] Embed job error: {e}")
    finally:
        track.embed_pending = False


# ── Process worker (thread) ───────────────────────────────────────────────────

def _process_worker(
    fq, rq, stop,
    recogniser_container, model_ready_event,
    detector, tracker, gallery, cfg, logger,
):
    fps_smooth = 0.0
    t_prev     = time.time()
    fidx       = 0
    ds         = cfg["det_scale"]

    model_ready_event.wait()
    recogniser      = recogniser_container[0]
    use_insightface = isinstance(recogniser, InsightFaceRecogniser)

    while not stop.is_set():
        try:
            frame, vt = fq.get(timeout=0.1)
        except queue.Empty:
            continue

        fidx += 1

        # Scale frame for detection
        det_frame = (cv2.resize(frame, (0, 0), fx=ds, fy=ds)
                     if ds != 1.0 else frame)
        det_h, det_w = det_frame.shape[:2]
        disp_h, disp_w = frame.shape[:2]
        sx = disp_w / det_w
        sy = disp_h / det_h

        # Detect + track
        faces  = detector.detect_and_deduplicate(det_frame)
        tracks = tracker.update(faces)

        for track in tracks:
            track._last_vt = vt
            should_embed = (
                track.frames_no_recog >= cfg["recognition_interval"]
                and not track.embed_pending
            )
            if should_embed:
                full_row = track.face_row.copy()
                full_row[0:14:2] *= sx
                full_row[1:14:2] *= sy
                track.embed_pending   = True
                track.frames_no_recog = 0
                frame_snap            = frame.copy()
                _embed_pool.submit(
                    _submit_embed,
                    recogniser, use_insightface,
                    frame_snap, full_row,
                    disp_w, disp_h,
                    track, gallery, cfg, logger,
                )
            else:
                track.frames_no_recog += 1

        # FPS smoothing
        t_now      = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev     = t_now

        # Build annotated frame for display queue
        annotated = frame.copy()
        for track in tracks:
            x, y, w, h = track.face_row[:4]
            x1 = int(x * sx); y1 = int(y * sy)
            x2 = int((x + w) * sx); y2 = int((y + h) * sy)
            draw_face_label(annotated, x1, y1, x2, y2,
                            track.display_name,
                            track.score,
                            track.display_status)

        recog_label = "ArcFace" if use_insightface else "SFace"
        draw_hud(
            annotated,
            fps            = fps_smooth,
            frame_idx      = fidx,
            recogniser_label = recog_label,
            vote_window    = 5,
            max_yaw        = cfg["max_yaw"],
            threshold_accept = cfg["threshold_accept"],
            threshold_reject = cfg["threshold_reject"],
            blur_threshold = cfg["blur_threshold"],
            db_size        = gallery.embedding_count,
        )

        try:
            rq.put_nowait(annotated)
        except queue.Full:
            pass


# ── Main run loop ─────────────────────────────────────────────────────────────

def run(cfg: dict):
    """
    Start the full detection pipeline.

    Args:
        cfg: merged config dict (from load_config + CLI args)
    """
    headless = cfg.get("headless", False)

    # ── Load model in background ──────────────────────────────────────────────
    model_ready     = threading.Event()
    rec_container   = [None]
    load_err        = [None]

    def _load_model():
        try:
            rec_container[0] = load_recogniser(cfg["sface_model"])
        except Exception as e:
            load_err[0] = e
        finally:
            model_ready.set()

    threading.Thread(target=_load_model, daemon=True, name="model-load").start()

    # ── Init components ───────────────────────────────────────────────────────
    detector = YuNetDetector.from_path(
        cfg["yunet_model"],
        score_threshold=cfg["det_confidence"],
    )
    tracker  = FaceTracker(max_age=10)
    gallery  = FaceGallery(
        db_path          = cfg["db_path"],
        collection_name  = cfg["collection_name"],
        refresh_interval = cfg["refresh_interval"],
    )
    gallery.start()

    logger = DetectionLogger(
        db_path   = cfg["output_db"],
        cooldown   = cfg.get("log_cooldown", 3.0),
    )

    src, is_rtsp, src_label = _build_source(cfg)
    print(f"[pipeline] Source: {src_label}  RTSP:{is_rtsp}  Headless:{headless}")

    stop            = threading.Event()
    reconnect_event = threading.Event()
    fq = queue.Queue(maxsize=4)
    rq = queue.Queue(maxsize=4)

    t_cap  = threading.Thread(
        target=_capture_worker,
        args=(src, is_rtsp, cfg["transport"], fq, stop, reconnect_event),
        daemon=True, name="capture",
    )
    t_proc = threading.Thread(
        target=_process_worker,
        args=(fq, rq, stop, rec_container, model_ready,
              detector, tracker, gallery, cfg, logger),
        daemon=True, name="process",
    )
    t_cap.start()
    t_proc.start()

    print("[pipeline] Running — press Q to quit (GUI mode) or Ctrl+C (headless)")

    if not headless:
        cv2.namedWindow("Survil v1", cv2.WINDOW_NORMAL)

    last_frame      = None
    reconnect_start = 0.0
    blank           = np.zeros((480, 640, 3), np.uint8)

    try:
        while not stop.is_set() and not _global_stop.is_set():
            # Drain display queue — keep only the latest frame
            try:
                while True:
                    last_frame = rq.get_nowait()
            except queue.Empty:
                pass

            if not headless:
                display = last_frame.copy() if last_frame is not None \
                          else blank.copy()

                if not model_ready.is_set():
                    draw_loading_overlay(display, "Loading ArcFace model…")
                elif reconnect_event.is_set():
                    if reconnect_start == 0.0:
                        reconnect_start = time.time()
                    draw_reconnect_overlay(display,
                                           time.time() - reconnect_start)
                else:
                    reconnect_start = 0.0

                cv2.imshow("Survil v1", display)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    stop.set()
            else:
                # Headless: just sleep to avoid busy loop
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[pipeline] KeyboardInterrupt — shutting down …")

    finally:
        stop.set()
        t_cap.join(timeout=3)
        t_proc.join(timeout=3)
        if not headless:
            cv2.destroyAllWindows()
        gallery.stop()
        logger.close()
        _embed_pool.shutdown(wait=False)
        print("[pipeline] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = _merge_args_into_config(args, cfg)

    if not cfg.get("rtsp") and not cfg.get("video"):
        print("[pipeline] ERROR: Provide --rtsp or --video (or set in config.json)")
        sys.exit(1)

    run(cfg)
