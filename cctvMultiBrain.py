import cv2
import threading
import queue
import time
import os
import numpy as np
import requests                         # pip install requests
from deepface import DeepFace

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH           = "my_database"       # Folder with KNOWN faces only
MODEL_NAME        = "VGG-Face"
DETECTOR_BACKEND  = "opencv"            # "retinaface" for better accuracy
DISTANCE_THRESH   = 0.55
NUM_WORKERS       = 1
AI_EVERY_N_FRAMES = 5                   # AI runs every N frames; tracker handles rest
GRID_CELL         = 200                 # Grid size for dedup zones

# ─── NOTIFICATION CONFIG ──────────────────────────────────────────────────────
# Option A: Your own Kotlin/any backend endpoint
NOTIFY_URL        = "http://your-server.com/api/alert"   # ← change this

# Option B: Firebase Cloud Messaging (uncomment + fill to use FCM instead)
# FCM_SERVER_KEY  = "YOUR_FCM_SERVER_KEY"
# FCM_DEVICE_TOKEN= "DEVICE_FCM_TOKEN"

# How many seconds to wait before sending another alert for the same zone
NOTIFY_COOLDOWN   = 30                  # seconds
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(DB_PATH, exist_ok=True)

# ── Queues & Shared State ──────────────────────────────────────────────────────
frame_queue      = queue.Queue(maxsize=2)
recognized_faces = []
lock             = threading.Lock()
notify_lock      = threading.Lock()
cooldown_map     = {}                   # zone_key → last_notify_time

# ── Pre-load model ONCE at startup ────────────────────────────────────────────
print("⏳ Loading model — please wait...")
DeepFace.build_model(MODEL_NAME)
print("✅ Model ready!\n")


# ─── NOTIFICATION: Send HTTP POST to your Kotlin/any backend ──────────────────
def send_http_notification(zone, timestamp):
    """
    Sends a POST request to your backend (Kotlin server or any REST API).
    Your Kotlin server receives this and can trigger a push notification,
    sound alarm, log to DB, etc.
    """
    payload = {
        "event"    : "UNKNOWN_FACE_DETECTED",
        "zone"     : f"{zone}",
        "timestamp": timestamp,
        "camera"   : "CAM_01"           # useful if you have multiple cameras
    }
    try:
        response = requests.post(NOTIFY_URL, json=payload, timeout=5)
        print(f"🔔 Notification sent → HTTP {response.status_code} | Zone {zone}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Notification failed (server unreachable) | Zone {zone}")
    except requests.exceptions.Timeout:
        print(f"⚠️  Notification timed out | Zone {zone}")
    except Exception as e:
        print(f"⚠️  Notification error: {e}")


# ─── NOTIFICATION: Firebase FCM alternative (uncomment to use) ────────────────
# def send_fcm_notification(zone, timestamp):
#     headers = {
#         "Authorization": f"key={FCM_SERVER_KEY}",
#         "Content-Type" : "application/json"
#     }
#     payload = {
#         "to": FCM_DEVICE_TOKEN,
#         "notification": {
#             "title": "⚠️ Unknown Person Detected",
#             "body" : f"Zone {zone} at {timestamp}"
#         },
#         "data": {"zone": str(zone), "camera": "CAM_01"}
#     }
#     try:
#         r = requests.post("https://fcm.googleapis.com/fcm/send",
#                           headers=headers, json=payload, timeout=5)
#         print(f"🔔 FCM sent → {r.status_code}")
#     except Exception as e:
#         print(f"⚠️  FCM error: {e}")


# ─── ZONE KEY: Prevents spam by grouping nearby positions ─────────────────────
def zone_key(x, y):
    return (x // GRID_CELL, y // GRID_CELL)


# ─── MAYBE NOTIFY: Cooldown-gated notification trigger ────────────────────────
def maybe_notify(x, y):
    """
    Fires a notification for an unknown face, but only once per zone
    per NOTIFY_COOLDOWN seconds — avoids spamming your backend.
    Runs in a background thread so it never blocks the camera loop.
    """
    key = zone_key(x, y)
    now = time.time()

    with notify_lock:
        if now - cooldown_map.get(key, 0) < NOTIFY_COOLDOWN:
            return                          # Still in cooldown, skip
        cooldown_map[key] = now

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

    # Non-blocking: send notification in background thread
    threading.Thread(
        target=send_http_notification,
        args=(key, timestamp),
        daemon=True
    ).start()


# ─── RECOGNIZE: Check if face belongs to a known person ──────────────────────
def recognize(face_crop, x, y, worker_id):
    """
    Returns the person's name if found in DB, else "Unknown".
    Triggers a notification for unknown faces.
    """
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
                name = os.path.basename(row["identity"]).split(".")[0]
                return name                 # ✅ Known person — no alert

    except Exception as e:
        print(f"[W{worker_id}] recognize error: {e}")

    # ── Unknown face: trigger notification (cooldown-gated) ───────────────────
    maybe_notify(x, y)
    return "Unknown"


# ─── WORKER: Background thread that runs AI processing ────────────────────────
def worker_brain(worker_id):
    global recognized_faces
    print(f"Worker {worker_id} online.")

    while True:
        frame = frame_queue.get()
        try:
            h_orig, w_orig = frame.shape[:2]
            small   = cv2.resize(frame, (640, 480))
            scale_x = w_orig / 640
            scale_y = h_orig / 480

            # ── Detect all faces ───────────────────────────────────────────────
            try:
                faces = DeepFace.extract_faces(
                    img_path          = small,
                    detector_backend  = DETECTOR_BACKEND,
                    enforce_detection = True,
                    align             = True,
                )
            except ValueError:
                faces = []

            # ── Recognize each face ────────────────────────────────────────────
            new_faces = []
            for face_obj in faces:
                area = face_obj["facial_area"]
                x = int(area["x"] * scale_x)
                y = int(area["y"] * scale_y)
                w = int(area["w"] * scale_x)
                h = int(area["h"] * scale_y)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                name = recognize(face_crop, x, y, worker_id)
                new_faces.append({"name": name, "box": (x, y, w, h)})

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


# ─── TRACKER: Smooths bounding boxes between AI frames ────────────────────────
trackers     = []
tracker_lock = threading.Lock()

def rebuild_trackers(frame, ai_results):
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

frame_counter   = 0
last_ai_results = []
display_faces   = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # ── Send frame to AI worker every N frames ─────────────────────────────────
    if frame_counter % AI_EVERY_N_FRAMES == 0:
        if not frame_queue.full():
            frame_queue.put(frame.copy())

    # ── Check for fresh AI results ─────────────────────────────────────────────
    with lock:
        current_ai = recognized_faces.copy()

    if current_ai != last_ai_results:
        last_ai_results = current_ai
        with tracker_lock:
            trackers = rebuild_trackers(frame, current_ai)
        display_faces = current_ai
    else:
        display_faces = update_trackers(frame)

    # ── Draw bounding boxes ────────────────────────────────────────────────────
    for face in display_faces:
        x, y, w, h = face["box"]
        name        = face["name"]
        is_unknown  = name == "Unknown"

        color = (0, 0, 255) if is_unknown else (0, 255, 0)
        label = "Unknown ⚠️" if is_unknown else f"✅ {name}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x, y-th-12), (x+tw+8, y), color, -1)
        cv2.putText(frame, label, (x+4, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── HUD overlay ───────────────────────────────────────────────────────────
    known_count   = sum(1 for f in display_faces if f["name"] != "Unknown")
    unknown_count = sum(1 for f in display_faces if f["name"] == "Unknown")

    cv2.putText(frame,
                f"Frame: {frame_counter}  |  Known: {known_count}  |  Unknown: {unknown_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.imshow("CCTV Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()