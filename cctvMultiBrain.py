import cv2
import threading
import queue
import time
import os
import numpy as np
from deepface import DeepFace

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH          = "my_database"
MODEL_NAME       = "ArcFace"
DETECTOR_BACKEND = "opencv"      # "retinaface" for better accuracy, slower
DISTANCE_THRESH  = 0.55
SAVE_COOLDOWN    = 30            # seconds before re-saving same zone
NUM_WORKERS      = 1
AI_EVERY_N_FRAMES = 5           # AI runs every 5 frames; tracker handles rest
GRID_CELL        = 200
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(DB_PATH, exist_ok=True)

# Queues
frame_queue = queue.Queue(maxsize=2)

# Shared state
recognized_faces = []   # Latest AI results: [{"name", "box"}]
lock             = threading.Lock()
save_lock        = threading.Lock()
cooldown_map     = {}   # zone_key -> last_save_time

# ── Pre-load model ONCE at startup ────────────────────────────────────────────
print("⏳ Loading ArcFace model...")
DeepFace.build_model(MODEL_NAME)
print("✅ ArcFace ready!\n")

# ─── HELPER: Zone-based dedup save ────────────────────────────────────────────
def zone_key(x, y):
    return (x // GRID_CELL, y // GRID_CELL)

def maybe_save_unknown(face_crop, x, y, worker_id):
    key = zone_key(x, y)
    now = time.time()
    with save_lock:
        if now - cooldown_map.get(key, 0) < SAVE_COOLDOWN:
            return "Unknown"
        cooldown_map[key] = now
        ts   = int(now * 1000)
        name = f"Unknown_{ts}_W{worker_id}"
        cv2.imwrite(os.path.join(DB_PATH, f"{name}.jpg"), face_crop)
        print(f"[W{worker_id}] ⚠️  New face saved → {name}.jpg")
        return name

# ─── HELPER: Recognize one face crop ──────────────────────────────────────────
def recognize(face_crop, x, y, worker_id):
    try:
        dfs = DeepFace.find(
            img_path          = face_crop,
            db_path           = DB_PATH,
            model_name        = MODEL_NAME,
            distance_metric   = "cosine",
            enforce_detection = False,
            silent            = True,
        )
        df = dfs[0] if dfs else None
        if df is not None and not df.empty:
            row      = df.iloc[0]
            distance = row.get("distance", 1.0)
            if distance <= DISTANCE_THRESH:
                raw = os.path.basename(row["identity"]).split(".")[0]
                if not raw.startswith("Unknown_"):
                    return raw   # ✅ Known person
    except Exception as e:
        print(f"[W{worker_id}] recognize error: {e}")

    return maybe_save_unknown(face_crop, x, y, worker_id)

# ─── WORKER: Runs continuously forever ────────────────────────────────────────
def worker_brain(worker_id):
    global recognized_faces
    print(f"Worker {worker_id} online.")

    while True:   # ← Infinite loop — keeps running as long as CCTV is on
        frame = frame_queue.get()

        try:
            # ── 1. Resize for faster AI processing ────────────────────────────
            h_orig, w_orig = frame.shape[:2]
            small   = cv2.resize(frame, (640, 480))
            scale_x = w_orig / 640
            scale_y = h_orig / 480

            # ── 2. Detect ALL faces in current frame ──────────────────────────
            try:
                faces = DeepFace.extract_faces(
                    img_path          = small,
                    detector_backend  = DETECTOR_BACKEND,
                    enforce_detection = True,
                    align             = True,
                )
            except ValueError:
                faces = []

            # ── 3. Recognize each detected face ───────────────────────────────
            new_faces = []
            for face_obj in faces:
                area = face_obj["facial_area"]

                # Scale coords back to original resolution
                x = int(area["x"] * scale_x)
                y = int(area["y"] * scale_y)
                w = int(area["w"] * scale_x)
                h = int(area["h"] * scale_y)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                name = recognize(face_crop, x, y, worker_id)
                new_faces.append({"name": name, "box": (x, y, w, h)})

            # ── 4. Push results to main thread ────────────────────────────────
            with lock:
                recognized_faces = new_faces

        except Exception as e:
            print(f"[W{worker_id}] Unexpected error: {e}")
        finally:
            frame_queue.task_done()

# ── Spawn workers ──────────────────────────────────────────────────────────────
for i in range(NUM_WORKERS):
    t = threading.Thread(target=worker_brain, args=(i,), daemon=True)
    t.start()

# ─── TRACKER POOL: Smooths boxes between AI frames ────────────────────────────
# When AI isn't running, OpenCV trackers keep the boxes moving with the person
trackers     = []   # List of {"tracker", "name", "box"}
tracker_lock = threading.Lock()

def rebuild_trackers(frame, ai_results):
    """Called every time AI gives us new results — reset all trackers."""
    new_trackers = []
    for face in ai_results:
        x, y, w, h = face["box"]
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, (x, y, w, h))
        new_trackers.append({
            "tracker": tracker,
            "name"   : face["name"],
            "box"    : (x, y, w, h)
        })
    return new_trackers

def update_trackers(frame):
    """Move existing tracker boxes to follow faces frame-by-frame."""
    updated = []
    with tracker_lock:
        for t in trackers:
            success, box = t["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                updated.append({"name": t["name"], "box": (x, y, w, h)})
    return updated

# ─── MAIN CAMERA LOOP ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ CCTV Active. Press 'q' to quit.\n")

frame_counter    = 0
last_ai_results  = []
display_faces    = []   # What actually gets drawn — either AI or tracker results

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # ── Every N frames: send to AI workers ────────────────────────────────────
    if frame_counter % AI_EVERY_N_FRAMES == 0:
        if not frame_queue.full():
            frame_queue.put(frame.copy())

    # ── Check if AI workers returned new results ───────────────────────────────
    with lock:
        current_ai = recognized_faces.copy()

    if current_ai != last_ai_results:
        # AI gave us fresh data → rebuild trackers to follow these new positions
        last_ai_results = current_ai
        with tracker_lock:
            trackers = rebuild_trackers(frame, current_ai)
        display_faces = current_ai
    else:
        # No new AI data yet → use tracker to keep boxes moving smoothly
        display_faces = update_trackers(frame)

    # ── Draw boxes ────────────────────────────────────────────────────────────
    for face in display_faces:
        x, y, w, h = face["box"]
        name        = face["name"]
        is_unknown  = "Unknown" in name

        color       = (0, 0, 255) if is_unknown else (0, 255, 0)
        label       = "Unknown" if is_unknown else name

        # Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+6, y), color, -1)
        cv2.putText(frame, label, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ── HUD: frame counter + face count ───────────────────────────────────────
    cv2.putText(frame, f"Faces: {len(display_faces)}  Frame: {frame_counter}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("CCTV Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()