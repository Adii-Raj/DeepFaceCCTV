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
MODEL_NAME        = "Facenet512"        # Upgraded from VGG-Face
DETECTOR_BACKEND  = "mtcnn"            # Upgraded from opencv
DISTANCE_THRESH   = 0.30                # Facenet512 cosine threshold is usually ~0.3
NUM_WORKERS       = 1
AI_EVERY_N_FRAMES = 5                   
GRID_CELL         = 200                 
MAX_TRACK_FAILURES= 5                   # Drop a tracker if it loses the face for 5 frames

# ─── NOTIFICATION CONFIG ──────────────────────────────────────────────────────
NOTIFY_URL        = "http://your-server.com/api/alert"
NOTIFY_COOLDOWN   = 30

os.makedirs(DB_PATH, exist_ok=True)

# ── Queues & Shared State ──────────────────────────────────────────────────────
frame_queue      = queue.Queue(maxsize=2)
recognized_faces = []
lock             = threading.Lock()
notify_lock      = threading.Lock()
cooldown_map     = {}

# ─── IN-MEMORY VECTOR DATABASE ────────────────────────────────────────────────
KNOWN_FACES = []
db_lock = threading.Lock()       # Prevents memory crashes when updating RAM
processed_files = set()          # Keeps track of images we already processed

def cosine_distance(a, b):
    """Calculates cosine distance between two arrays/vectors."""
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_new_faces():
    """Scans subfolders for new images and adds them to RAM safely."""
    global KNOWN_FACES, processed_files
    
    # Step 1: Loop through the subfolders (the Student Names)
    for person_folder in os.listdir(DB_PATH):
        person_path = os.path.join(DB_PATH, person_folder)
        
        # Skip if it's not a folder (like a hidden .DS_Store file on Mac)
        if not os.path.isdir(person_path): 
            continue 
            
        # Step 2: Loop through the images inside the student's folder
        for file in os.listdir(person_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                # We use the relative path as our unique tracker to avoid re-processing
                unique_file_id = f"{person_folder}/{file}"
                
                if unique_file_id in processed_files:
                    continue

                full_img_path = os.path.join(person_path, file)
                
                try:
                    res = DeepFace.represent(
                        img_path=full_img_path, 
                        model_name=MODEL_NAME, 
                        detector_backend=DETECTOR_BACKEND, 
                        enforce_detection=True
                    )
                    if len(res) > 0:
                        with db_lock:
                            # Save a dictionary containing the folder name and the mathematical face
                            KNOWN_FACES.append({
                                "name": person_folder, 
                                "embedding": res[0]["embedding"]
                            })
                        processed_files.add(unique_file_id)
                        print(f"✅ Hot-Loaded: {person_folder} ({file})")
                except Exception as e:
                    print(f"⚠️ Failed to hot-load {unique_file_id}: {e}")

def dynamic_db_updater():
    """Background thread that checks for new faces every 60 seconds."""
    print("⏳ Initializing database...")
    DeepFace.build_model(MODEL_NAME) 
    process_new_faces() # Run once instantly at startup
    print(f"✅ Database Ready! Loaded {len(KNOWN_FACES)} faces.\n")

    # Now, loop forever in the background
    while True:
        time.sleep(60) # Wait 60 seconds
        process_new_faces() # Check for new faces

# ─── NOTIFICATION SYSTEM ──────────────────────────────────────────────────────
def send_http_notification(zone, timestamp):
    payload = {"event": "UNKNOWN_FACE_DETECTED", "zone": str(zone), "timestamp": timestamp}
    try:
        requests.post(NOTIFY_URL, json=payload, timeout=5)
        print(f"🔔 Alert Sent | Zone {zone}")
    except Exception:
        pass # Silently fail to avoid console spam

def zone_key(x, y):
    return (x // GRID_CELL, y // GRID_CELL)

def maybe_notify(x, y):
    key = zone_key(x, y)
    now = time.time()
    with notify_lock:
        if now - cooldown_map.get(key, 0) < NOTIFY_COOLDOWN:
            return 
        cooldown_map[key] = now

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    threading.Thread(target=send_http_notification, args=(key, timestamp), daemon=True).start()

# ─── RECOGNIZE FUNCTION ───────────────────────────────────────────────────────
def recognize(face_crop, x, y):
    """Generates embedding for cropped face and matches via RAM dictionary."""
    try:
        # Get embedding of the unknown face (no need to enforce detection on a crop)
        res = DeepFace.represent(
            img_path=face_crop, 
            model_name=MODEL_NAME, 
            enforce_detection=False
        )
        target_emb = res[0]["embedding"]
        
        best_match = "Unknown"
        best_dist = float("inf")
        
       # Super-fast vector search (In RAM)
        with db_lock:  
            for known in KNOWN_FACES:
                # Compare the live camera face to every known embedding
                dist = cosine_distance(target_emb, known["embedding"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = known["name"] # Grabs the folder name!
                
        if best_dist <= DISTANCE_THRESH:
            return best_match
            
    except Exception:
        pass # Return unknown if extraction fails

    maybe_notify(x, y)
    return "Unknown"

# ─── AI WORKER THREAD ─────────────────────────────────────────────────────────
def worker_brain(worker_id):
    global recognized_faces
    print(f"Worker {worker_id} online.")

    while True:
        frame = frame_queue.get()
        try:
            h_orig, w_orig = frame.shape[:2]
            small = cv2.resize(frame, (640, 480))
            scale_x, scale_y = w_orig / 640, h_orig / 480

            # 1. Detect faces using YOLOv8
            try:
                faces = DeepFace.extract_faces(
                    img_path=small,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True,
                    align=True
                )
            except ValueError:
                faces = []

            # 2. Recognize
            new_faces = []
            for face_obj in faces:
                area = face_obj["facial_area"]
                x, y = int(area["x"] * scale_x), int(area["y"] * scale_y)
                w, h = int(area["w"] * scale_x), int(area["h"] * scale_y)

                # Expand crop slightly for better embeddings
                pad = 10
                y1, y2 = max(0, y-pad), min(h_orig, y+h+pad)
                x1, x2 = max(0, x-pad), min(w_orig, x+w+pad)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue

                name = recognize(face_crop, x, y)
                new_faces.append({"name": name, "box": (x, y, w, h)})

            with lock:
                recognized_faces = new_faces

        except Exception as e:
            print(f"[W{worker_id}] Error: {e}")
        finally:
            frame_queue.task_done()

# ─── STARTUP ──────────────────────────────────────────────────────────────────
# Start the background database updater
threading.Thread(target=dynamic_db_updater, daemon=True).start()

for i in range(NUM_WORKERS):
    threading.Thread(target=worker_brain, args=(i,), daemon=True).start()

# ─── TRACKER LOGIC ────────────────────────────────────────────────────────────
trackers = []
tracker_lock = threading.Lock()

def get_csrt_tracker():
    """Handles API changes between OpenCV versions."""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        return cv2.TrackerCSRT_create()

def rebuild_trackers(frame, ai_results):
    new_trackers = []
    for face in ai_results:
        x, y, w, h = face["box"]
        tracker = get_csrt_tracker()
        tracker.init(frame, (x, y, w, h))
        new_trackers.append({
            "tracker": tracker,
            "name": face["name"],
            "failures": 0,          # Added failure counter
            "box": (x, y, w, h)
        })
    return new_trackers

def update_trackers(frame):
    updated = []
    with tracker_lock:
        active_trackers = []
        for t in trackers:
            success, box = t["tracker"].update(frame)
            if success:
                t["failures"] = 0   # Reset on success
                x, y, w, h = [int(v) for v in box]
                updated.append({"name": t["name"], "box": (x, y, w, h)})
                active_trackers.append(t)
            else:
                t["failures"] += 1  # Increment on failure
                if t["failures"] < MAX_TRACK_FAILURES:
                    active_trackers.append(t)
        
        # Prune dead trackers to fix memory leak
        trackers.clear()
        trackers.extend(active_trackers)
    return updated

# ─── MAIN CAMERA LOOP ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ CCTV Active. Press 'q' to quit.\n")

frame_counter = 0
last_ai_results = []
display_faces = []

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_counter += 1

    # Non-blocking AI dispatch
    if frame_counter % AI_EVERY_N_FRAMES == 0:
        try:
            frame_queue.put(frame.copy(), block=False)
        except queue.Full:
            pass # Drop frame to prevent camera lag

    with lock:
        current_ai = recognized_faces.copy()

    # Rebuild if new AI results, otherwise update tracker
    if current_ai != last_ai_results:
        last_ai_results = current_ai
        with tracker_lock:
            trackers = rebuild_trackers(frame, current_ai)
        display_faces = current_ai
    else:
        display_faces = update_trackers(frame)

    # ── Draw HUD ──────────────────────────────────────────────────────────────
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

    cv2.imshow("CCTV Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()