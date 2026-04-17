import cv2
import time
import queue
import os
import numpy as np
import threading
from ultralytics import YOLO

from config import (
    YOLO_MODEL, YOLO_RUN_EVERY_N_FRAMES, TRACK_SEND_COOLDOWN,
    TRACK_EXPIRY_SECONDS, CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_CONF,
    ENHANCE_LOW_LIGHT, ENHANCE_MIN_BRIGHTNESS,
    VOTE_WINDOW_SECONDS, HIGH_CONF_LOCK_THRESH
)


# ──────────────────────────────────────────────
# 🧵 DEDICATED CAPTURE THREAD
# Grabs frames as fast as possible so the RTSP
# buffer never overflows — completely decoupled
# from YOLO / AI processing.
# ──────────────────────────────────────────────
class RTSPCaptureThread:
    """
    Runs cap.grab() in a tight background thread.
    Main thread calls get_frame() to retrieve the
    latest decoded frame on demand — no blocking.
    """
    def __init__(self, url):
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stop = threading.Event()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

        self.cap = self._open()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open(self):
        # Check if the url is an integer (like 0, 1) or a numeric string
        if isinstance(self.url, int) or str(self.url).isdigit():
            # For local webcams, use DirectShow (Windows) or the default backend
            cap = cv2.VideoCapture(int(self.url), cv2.CAP_DSHOW)
        else:
            # For RTSP IP cameras, force FFmpeg
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        return cap

    def _capture_loop(self):
        """
        Tight loop: grab → decode → store latest frame.
        Never sleeps. If connection drops, reconnects after 3 s.
        """
        while not self._stop.is_set():
            ret = self.cap.grab()
            if not ret:
                print("[CAPTURE] ⚠️ grab() failed — reconnecting in 3 s...")
                time.sleep(3)
                self.cap.release()
                self.cap = self._open()
                continue

            # Decode only the grabbed frame
            ret, frame = self.cap.retrieve()
            with self._lock:
                self._ret = ret
                self._frame = frame
            # Giving gil a tiny break
            time.sleep(0.005)

    def get_frame(self):
        """Returns (ret, frame) — always the freshest available."""
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._stop.set()
        self.cap.release()


# ──────────────────────────────────────────────
# 💡 LOW-LIGHT ENHANCEMENT
# ──────────────────────────────────────────────
def enhance_low_light_crop(img):
    """Conservative CLAHE + Gamma only for truly dark faces."""
    if img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness > ENHANCE_MIN_BRIGHTNESS:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clip = 1.5 if mean_brightness < 40 else 1.2
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16, 16))
    l = clahe.apply(l)

    img_enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if mean_brightness < 50:
        gamma = 1.5
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)]).astype("uint8")
        img_enhanced = cv2.LUT(img_enhanced, table)

    return img_enhanced


# ──────────────────────────────────────────────
# 👀 MAIN CAMERA WORKER
# ──────────────────────────────────────────────
def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting YOLOv8 Tracker on {cam_name}...")

    model = YOLO(YOLO_MODEL)

    # 🧵 Start dedicated capture thread — decoupled from processing
    capture = RTSPCaptureThread(url)
    print(f"[EYES] 🧵 Capture thread started for {cam_name}")

    frame_count = 0
    track_states = {}    # {track_id: {"votes", "display", "last_seen", "box", "lock_until"}}
    track_last_sent = {} # {track_id: last_sent_timestamp}

    while True:

        # ─────────────────────────────────────────
        # 🗳️ 1. Drain result queue & update votes
        # ─────────────────────────────────────────
        while not result_queue.empty():
            try:
                res_cam, res_track_id, matched_name, conf = result_queue.get_nowait()
                if res_cam != cam_name or res_track_id not in track_states:
                    continue

                state = track_states[res_track_id]
                current_time = time.time()

                # Always record valid votes (NEVER wipe the history array)
                if matched_name != "Unknown":
                    state["votes"].append({"name": matched_name, "conf": conf, "time": current_time})

                # Prune old votes based on the time window
                cutoff = current_time - VOTE_WINDOW_SECONDS
                state["votes"] = [v for v in state["votes"] if v["time"] > cutoff]

                # Calculate weighted scores and count total votes per person
                scores = {}
                vote_counts = {}
                for v in state["votes"]:
                    time_weight = max(0.2, 1.0 - (current_time - v["time"]) / VOTE_WINDOW_SECONDS)
                    scores[v["name"]] = scores.get(v["name"], 0.0) + (v["conf"] * time_weight)
                    vote_counts[v["name"]] = vote_counts.get(v["name"], 0) + 1

                current_display = state.get("display", "Unknown")

                if scores:
                    # HYSTERESIS LOGIC: Give the currently displayed name a 50% score boost.
                    # This means a flicker/false-positive has to violently outscore the incumbent to take over.
                    if current_display in scores:
                        scores[current_display] *= 1.5

                    best_name = max(scores, key=scores.get)
                    
                    # Revert multiplier to check absolute raw score against thresholds
                    raw_best_score = scores[best_name] / 1.5 if best_name == current_display else scores[best_name]

                    if best_name == current_display:
                        # Drop to Unknown if the track score decays too much (e.g., person turns away)
                        if raw_best_score < 0.3:
                            state["display"] = "Unknown"
                    else:
                        # CHALLENGER LOGIC (Unknown -> A, or A -> B)
                        # Require at least 2 votes OR one extremely high-confidence frame to switch displays
                        if vote_counts[best_name] >= 2 or raw_best_score >= HIGH_CONF_LOCK_THRESH:
                            state["display"] = best_name
                else:
                    state["display"] = "Unknown"

            except queue.Empty:
                break

        # ─────────────────────────────────────────
        # 📷 2. Get latest frame from capture thread
        # ─────────────────────────────────────────
        ret, frame = capture.get_frame()

        if not ret or frame is None:
            time.sleep(0.01)  # brief yield — capture thread handles reconnection
            continue

        frame_count += 1
        h_orig, w_orig = frame.shape[:2]
        current_time = time.time()

        # ─────────────────────────────────────────
        # 🧹 3. Expire stale tracks
        # ─────────────────────────────────────────
        expired_ids = [
            tid for tid, state in track_states.items()
            if current_time - state["last_seen"] > TRACK_EXPIRY_SECONDS
        ]
        for tid in expired_ids:
            print(f"[EYES] 🚶 Track {tid} exited frame. Clearing votes.")
            del track_states[tid]
            track_last_sent.pop(tid, None)

        # ─────────────────────────────────────────
        # 🔍 4. YOLO ByteTrack (every N frames)
        # ─────────────────────────────────────────
        if frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
            results = model.track(
                source=frame, classes=[0], verbose=False,
                conf=YOLO_CONF, persist=True
            )

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else None
                if track_id is None:
                    continue

                if track_id not in track_states:
                    track_states[track_id] = {
                        "votes": [], "display": "Unknown",
                        "last_seen": current_time,
                        "box": (x1, y1, x2, y2),
                        "lock_until": 0
                    }

                track_states[track_id]["last_seen"] = current_time
                track_states[track_id]["box"] = (x1, y1, x2, y2)

                # ⏱️ Throttle: max 1 crop sent per track per TRACK_SEND_COOLDOWN seconds
                last_sent = track_last_sent.get(track_id, 0)
                if current_time - last_sent < TRACK_SEND_COOLDOWN:
                    continue

                # Crop face region with padding
                pad = 20
                raw_crop = frame[
                    max(0, y1 - pad):min(h_orig, y2 + pad),
                    max(0, x1 - pad):min(w_orig, x2 + pad)
                ]
                if raw_crop.size == 0 or raw_crop.shape[0] < 30 or raw_crop.shape[1] < 30:
                    continue

                person_crop = enhance_low_light_crop(raw_crop) if ENHANCE_LOW_LIGHT else raw_crop
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[EYES] 📸 Track {track_id} detected. Sending to AI...")

                try:
                    face_queue.put_nowait((cam_name, person_crop, track_id, timestamp))
                    track_last_sent[track_id] = current_time
                    with cache_counter.get_lock():
                        cache_counter.value += 1
                except queue.Full:
                    print(f"[EYES] ⚠️ Queue full. Dropping Track {track_id} frame.")

        # ─────────────────────────────────────────
        # 🖼️ 5. Draw overlays & display
        # ─────────────────────────────────────────
        for tid, state in track_states.items():
            x1, y1, x2, y2 = state["box"]
            display_name = state["display"]
            vote_count = len(state["votes"])
            color = (0, 255, 0) if display_name != "Unknown" else (0, 165, 255)
            label = f"ID:{tid} | {display_name} ({vote_count} votes)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # HUD
        cv2.rectangle(frame, (0, 0), (500, 90), (0, 0, 0), -1)
        cv2.putText(frame, cam_name, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(track_states)} | Cache: {cache_counter.value}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow(f"Live Tracker: {cam_name}", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.stop()
    cv2.destroyAllWindows()