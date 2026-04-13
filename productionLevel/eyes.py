import cv2
import time
import queue
import numpy as np
from collections import Counter
from ultralytics import YOLO

# Centralized config imports
from config import (
    YOLO_MODEL, YOLO_RUN_EVERY_N_FRAMES, TRACK_SEND_COOLDOWN,
    TRACK_EXPIRY_SECONDS, CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_CONF,
    ENHANCE_LOW_LIGHT, ENHANCE_MIN_BRIGHTNESS,
    VOTE_WINDOW_SECONDS, HIGH_CONF_LOCK_THRESH
)

def enhance_low_light_crop(img):
    """Conservative CLAHE + Gamma only for truly dark faces"""
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
    
    lab_enhanced = cv2.merge((l, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    if mean_brightness < 50:
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img_enhanced = cv2.LUT(img_enhanced, table)
        
    return img_enhanced

def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting YOLOv8 Tracker on {cam_name}...")
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    frame_count = 0
    track_states = {}      # {track_id: {"votes": [], "display": "Unknown", "last_seen": time, "box": tuple, "lock_until": 0}}
    track_last_sent = {}   # {track_id: last_sent_timestamp}

    while True:
        # 🗳️ 1. Process AI Results & Update Votes (Confidence + Recency Weighted)
        while not result_queue.empty():
            try:
                res_cam, res_track_id, matched_name, conf = result_queue.get_nowait()
                if res_cam == cam_name and res_track_id in track_states:
                    state = track_states[res_track_id]
                    current_time = time.time()

                    # Rule 1: High-confidence lock (instant override for clear shots)
                    if matched_name != "Unknown" and conf >= HIGH_CONF_LOCK_THRESH:
                        state["display"] = matched_name
                        state["lock_until"] = current_time + 2.0
                        state["votes"] = []  # Clear history to prevent drift
                        
                    # Rule 2: Keep locked name if timer hasn't expired
                    elif "lock_until" in state and current_time < state["lock_until"]:
                        pass
                        
                    # Rule 3: Recency-weighted scoring (ignores "Unknown")
                    else:
                        if matched_name != "Unknown":
                            state["votes"].append({"name": matched_name, "conf": conf, "time": current_time})
                            
                            # Sliding window: drop old votes
                            cutoff = current_time - VOTE_WINDOW_SECONDS
                            state["votes"] = [v for v in state["votes"] if v["time"] > cutoff]
                            
                            # Calculate time-decayed weighted scores
                            scores = {}
                            for v in state["votes"]:
                                time_weight = max(0.2, 1.0 - (current_time - v["time"]) / VOTE_WINDOW_SECONDS)
                                scores[v["name"]] = scores.get(v["name"], 0.0) + (v["conf"] * time_weight)
                                
                            state["display"] = max(scores, key=scores.get) if scores else "Unknown"
                        else:
                            # "Unknown" votes don't count, just fallback
                            state["display"] = "Unknown"
                            
            except queue.Empty:
                break

        # 📷 2. Capture Frame
        ret, frame = cap.read()
        if not ret:
            time.sleep(3)
            cap = cv2.VideoCapture(url)
            continue
            
        frame_count += 1
        h_orig, w_orig = frame.shape[:2]
        current_time = time.time()

        # 🧹 3. Cleanup expired tracks
        expired_ids = [tid for tid, state in track_states.items() if current_time - state["last_seen"] > TRACK_EXPIRY_SECONDS]
        for tid in expired_ids:
            print(f"[EYES] 🚶‍♂️ Track {tid} exited frame. Clearing votes.")
            del track_states[tid]
            if tid in track_last_sent: del track_last_sent[tid]

        # 🔍 4. YOLO ByteTrack (Persistent IDs)
        if frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
            results = model.track(source=frame, classes=[0], verbose=False, conf=YOLO_CONF, persist=True)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else None
                if track_id is None: continue

                if track_id not in track_states:
                    track_states[track_id] = {"votes": [], "display": "Unknown", "last_seen": current_time, "box": (x1, y1, x2, y2), "lock_until": 0}
                
                track_states[track_id]["last_seen"] = current_time
                track_states[track_id]["box"] = (x1, y1, x2, y2)

                # ⏱️ THROTTLE: Prevent queue spam
                if track_id in track_last_sent and current_time - track_last_sent[track_id] < TRACK_SEND_COOLDOWN:
                    continue

                # Crop & Enhance
                pad = 20
                raw_crop = frame[max(0, y1-pad):min(h_orig, y2+pad), max(0, x1-pad):min(w_orig, x2+pad)]
                if raw_crop.size == 0 or raw_crop.shape[0] < 30 or raw_crop.shape[1] < 30: continue
                
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

        # 🖼️ 5. Draw Boxes & Live Voting Names
        for tid, state in track_states.items():
            x1, y1, x2, y2 = state["box"]
            display_name = state["display"]
            vote_count = len(state["votes"])
            
            color = (0, 255, 0) if display_name != "Unknown" else (0, 165, 255)
            label = f"ID:{tid} | {display_name} ({vote_count} votes)"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # HUD
        cv2.rectangle(frame, (0, 0), (500, 90), (0,0,0), -1)
        cv2.putText(frame, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(track_states)} | Cache: {cache_counter.value}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow(f"Live Tracker: {cam_name}", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()