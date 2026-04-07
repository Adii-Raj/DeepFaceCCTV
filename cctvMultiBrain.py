import cv2
import threading
import queue
import time
import os
import numpy as np
import requests
from deepface import DeepFace

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH           = "my_database"       
MODEL_NAME        = "Facenet512"        
DETECTOR_BACKEND  = "mtcnn"            
DISTANCE_THRESH   = 0.30                
NUM_WORKERS       = 2                   # Bumped to 2 workers for 4 cameras
AI_EVERY_N_FRAMES = 10                  # Check AI every 10 frames to save CPU
GRID_CELL         = 200                 
MAX_TRACK_FAILURES= 5                   

# ─── YOUR 4 CAMERAS GO HERE ───────────────────────────────────────────────────
# Replace these with your actual IP Webcam and VLC links!
# You can use 0 for your laptop's built-in webcam.
CAMERAS = {
    "CAM_1_PHONE_A": "rtsp://192.168.1.15:8080/h264_pcm.sdp",
    "CAM_2_PHONE_B": "rtsp://192.168.1.16:8080/h264_pcm.sdp",
    "CAM_3_LAPTOP" : "rtsp://192.168.1.20:8554/stream",
    "CAM_4_LOCAL"  : 0  
}

NOTIFY_URL        = "http://your-server.com/api/alert"
NOTIFY_COOLDOWN   = 30

os.makedirs(DB_PATH, exist_ok=True)

# ── Queues & Shared State (Now tracking per-camera) ───────────────────────────
frame_queue      = queue.Queue(maxsize=4)
latest_frames    = {cam: None for cam in CAMERAS}
recognized_faces = {cam: [] for cam in CAMERAS}
trackers         = {cam: [] for cam in CAMERAS}

lock             = threading.Lock()
tracker_lock     = threading.Lock()
notify_lock      = threading.Lock()
cooldown_map     = {}

# ─── IN-MEMORY VECTOR DATABASE ────────────────────────────────────────────────
KNOWN_FACES = []
db_lock = threading.Lock()       
processed_files = set()          

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_new_faces():
    global KNOWN_FACES, processed_files
    for person_folder in os.listdir(DB_PATH):
        person_path = os.path.join(DB_PATH, person_folder)
        if not os.path.isdir(person_path): continue 
            
        for file in os.listdir(person_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                unique_file_id = f"{person_folder}/{file}"
                if unique_file_id in processed_files: continue

                full_img_path = os.path.join(person_path, file)
                try:
                    res = DeepFace.represent(
                        img_path=full_img_path, model_name=MODEL_NAME, 
                        detector_backend=DETECTOR_BACKEND, enforce_detection=True
                    )
                    if len(res) > 0:
                        with db_lock:
                            KNOWN_FACES.append({"name": person_folder, "embedding": res[0]["embedding"]})
                        processed_files.add(unique_file_id)
                        print(f"  ✅ Hot-Loaded: {person_folder} ({file})")
                except Exception: pass

def dynamic_db_updater():
    print("⏳ Initializing AI Models...")
    DeepFace.build_model(MODEL_NAME) 
    process_new_faces() 
    print(f"✅ Database Ready! Loaded {len(KNOWN_FACES)} faces total.\n")
    while True:
        time.sleep(60) 
        process_new_faces() 

# ─── NOTIFICATIONS & RECOGNITION ──────────────────────────────────────────────
def send_http_notification(zone, timestamp, cam_name):
    payload = {"event": "UNKNOWN", "zone": str(zone), "timestamp": timestamp, "camera": cam_name}
    try: requests.post(NOTIFY_URL, json=payload, timeout=5)
    except Exception: pass 

def maybe_notify(x, y, cam_name):
    key = (cam_name, x // GRID_CELL, y // GRID_CELL)
    now = time.time()
    with notify_lock:
        if now - cooldown_map.get(key, 0) < NOTIFY_COOLDOWN: return 
        cooldown_map[key] = now
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    threading.Thread(target=send_http_notification, args=(key, timestamp, cam_name), daemon=True).start()

def recognize(face_crop, x, y, cam_name):
    try:
        res = DeepFace.represent(img_path=face_crop, model_name=MODEL_NAME, enforce_detection=False)
        target_emb = res[0]["embedding"]
        best_match, best_dist = "Unknown", float("inf")
        
        with db_lock:  
            for known in KNOWN_FACES:
                dist = cosine_distance(target_emb, known["embedding"])
                if dist < best_dist: best_dist, best_match = dist, known["name"] 
                
        if best_dist <= DISTANCE_THRESH: return best_match
    except Exception: pass 

    maybe_notify(x, y, cam_name)
    return "Unknown"

# ─── AI WORKER THREAD ─────────────────────────────────────────────────────────
def worker_brain(worker_id):
    global recognized_faces
    while True:
        cam_name, frame = frame_queue.get()
        try:
            h_orig, w_orig = frame.shape[:2]
            small = cv2.resize(frame, (640, 480))
            scale_x, scale_y = w_orig / 640, h_orig / 480

            try: faces = DeepFace.extract_faces(img_path=small, detector_backend=DETECTOR_BACKEND, enforce_detection=True, align=True)
            except ValueError: faces = []

            new_faces = []
            for face_obj in faces:
                area = face_obj["facial_area"]
                x, y = int(area["x"] * scale_x), int(area["y"] * scale_y)
                w, h = int(area["w"] * scale_x), int(area["h"] * scale_y)

                pad = 10
                y1, y2 = max(0, y-pad), min(h_orig, y+h+pad)
                x1, x2 = max(0, x-pad), min(w_orig, x+w+pad)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue

                name = recognize(face_crop, x, y, cam_name)
                new_faces.append({"name": name, "box": (x, y, w, h)})

            with lock: recognized_faces[cam_name] = new_faces

        except Exception: pass
        finally: frame_queue.task_done()

# ─── CAMERA STREAM READER THREADS ─────────────────────────────────────────────
def stream_reader(cam_name, rtsp_url):
    """Background thread that constantly pulls frames from the network camera."""
    cap = cv2.VideoCapture(rtsp_url)
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ {cam_name} disconnected. Reconnecting...")
            time.sleep(3)
            cap = cv2.VideoCapture(rtsp_url)
            continue
            
        # Standardize all camera sizes to 640x480 so they fit perfectly in the grid
        frame = cv2.resize(frame, (640, 480))
        latest_frames[cam_name] = frame.copy()
        frame_counter += 1

        if frame_counter % AI_EVERY_N_FRAMES == 0:
            try: frame_queue.put((cam_name, frame.copy()), block=False)
            except queue.Full: pass

# ─── TRACKER LOGIC (Multi-Camera) ─────────────────────────────────────────────
def get_csrt_tracker():
    try: return cv2.legacy.TrackerCSRT_create()
    except AttributeError: return cv2.TrackerCSRT_create()

def update_trackers(cam_name, frame, ai_results):
    updated = []
    with tracker_lock:
        # If AI gave us fresh results, rebuild trackers
        if ai_results is not None:
            trackers[cam_name] = []
            for face in ai_results:
                x, y, w, h = face["box"]
                t = get_csrt_tracker()
                t.init(frame, (x, y, w, h))
                trackers[cam_name].append({"tracker": t, "name": face["name"], "failures": 0, "box": (x, y, w, h)})
            return ai_results

        # Otherwise, update existing trackers
        active_trackers = []
        for t in trackers[cam_name]:
            success, box = t["tracker"].update(frame)
            if success:
                t["failures"] = 0   
                x, y, w, h = [int(v) for v in box]
                updated.append({"name": t["name"], "box": (x, y, w, h)})
                active_trackers.append(t)
            else:
                t["failures"] += 1  
                if t["failures"] < MAX_TRACK_FAILURES: active_trackers.append(t)
        trackers[cam_name] = active_trackers
    return updated

# ─── STARTUP ──────────────────────────────────────────────────────────────────
threading.Thread(target=dynamic_db_updater, daemon=True).start()
for i in range(NUM_WORKERS):
    threading.Thread(target=worker_brain, args=(i,), daemon=True).start()
for cam_name, url in CAMERAS.items():
    threading.Thread(target=stream_reader, args=(cam_name, url), daemon=True).start()

# ─── MAIN DISPLAY LOOP (The 2x2 Grid) ─────────────────────────────────────────
print("✅ CCTV NVR Active. Waiting for cameras to connect...")
last_ai_state = {cam: [] for cam in CAMERAS}

while True:
    time.sleep(0.03) # Limit UI refresh rate to ~30fps
    
    drawn_frames = []
    cam_names = list(CAMERAS.keys())

    for cam_name in cam_names:
        frame = latest_frames[cam_name]
        
        # If camera hasn't connected yet, show a black placeholder
        if frame is None:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, f"{cam_name} Connecting...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            drawn_frames.append(black_frame)
            continue

        # Check AI state safely
        with lock: current_ai = recognized_faces[cam_name].copy()
        
        # Decide whether to rebuild trackers or just update them
        if current_ai != last_ai_state[cam_name]:
            display_faces = update_trackers(cam_name, frame, current_ai)
            last_ai_state[cam_name] = current_ai
        else:
            display_faces = update_trackers(cam_name, frame, None)

        # Draw HUD for this specific camera
        for face in display_faces:
            x, y, w, h = face["box"]
            name = face["name"]
            is_unknown = (name == "Unknown")

            color = (0, 0, 255) if is_unknown else (0, 255, 0)
            label = "Unknown" if is_unknown else name

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), color, -1)
            cv2.putText(frame, label, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add Camera Name Tag to the top left
        cv2.rectangle(frame, (0, 0), (250, 40), (0,0,0), -1)
        cv2.putText(frame, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        drawn_frames.append(frame)

    # Stitch the 4 frames together (2x2 grid)
    top_row = np.hstack((drawn_frames[0], drawn_frames[1]))
    bot_row = np.hstack((drawn_frames[2], drawn_frames[3]))
    security_grid = np.vstack((top_row, bot_row))

    # Resize the final grid slightly so it fits on normal laptop screens
    security_grid = cv2.resize(security_grid, (1280, 960))
    
    cv2.imshow("Enterprise NVR Dashboard", security_grid)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()