import time
import os
import threading
import numpy as np
from deepface import DeepFace
from config import DB_PATH, MODEL_NAME, DISTANCE_THRESH, WAITING_TIME

# Global Matrix Variables
KNOWN_NAMES = []
KNOWN_EMBEDDINGS_MATRIX = None
processed_files = set()
db_lock = threading.Lock()

def process_new_faces():
    global KNOWN_NAMES, KNOWN_EMBEDDINGS_MATRIX, processed_files
    os.makedirs(DB_PATH, exist_ok=True)
    new_faces_added = False
    temp_embeddings = []

    if KNOWN_EMBEDDINGS_MATRIX is not None:
        temp_embeddings = KNOWN_EMBEDDINGS_MATRIX.tolist()

    for person_folder in os.listdir(DB_PATH):
        person_path = os.path.join(DB_PATH, person_folder)
        if not os.path.isdir(person_path): continue 
            
        for file in os.listdir(person_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                unique_file_id = f"{person_folder}/{file}"
                if unique_file_id in processed_files: continue

                try:
                    res = DeepFace.represent(
                        img_path=os.path.join(person_path, file), 
                        model_name=MODEL_NAME, 
                        enforce_detection=False 
                    )
                    with db_lock:
                        KNOWN_NAMES.append(person_folder)
                        temp_embeddings.append(res[0]["embedding"])
                        processed_files.add(unique_file_id)
                        new_faces_added = True
                        print(f"[BRAIN] ✅ Loaded/Hot-Loaded: {person_folder}")
                except Exception as e:
                    print(f"[BRAIN] ❌ Failed to load {file}: {e}")
                    
    if new_faces_added:
        with db_lock:
            KNOWN_EMBEDDINGS_MATRIX = np.array(temp_embeddings)

def dynamic_db_updater():
    while True:
        time.sleep(60)
        process_new_faces()

def start_ai_worker(face_queue, result_queue, cache_counter):
    print("[BRAIN] ⏳ Booting AI and loading Matrix Database...")
    DeepFace.build_model(MODEL_NAME)
    process_new_faces()
    threading.Thread(target=dynamic_db_updater, daemon=True).start()
    print("[BRAIN] 🧠 AI Worker active. Waiting for faces...")

    recent_logs = {}

    while True:
        cam_name, face_crop, track_id, timestamp = face_queue.get()
        
        with cache_counter.get_lock():
            cache_counter.value -= 1
            current_cache = cache_counter.value

        print(f"⚙️ [BRAIN] Processing Track {track_id} from {cam_name}. (Queue left: {current_cache})")

        try:
            # ✅ CRITICAL: OpenCV + Align restores angle/tilt accuracy
            res = DeepFace.represent(
                img_path=face_crop, 
                model_name=MODEL_NAME, 
                detector_backend="opencv",
                enforce_detection=False, 
                align=True
            )
                
            target_emb = np.array(res[0]["embedding"])
            best_match, best_dist = "Unknown", float("inf")
            
            with db_lock:
                if KNOWN_EMBEDDINGS_MATRIX is not None and len(KNOWN_NAMES) > 0:
                    dot_products = np.dot(KNOWN_EMBEDDINGS_MATRIX, target_emb)
                    norms = np.linalg.norm(KNOWN_EMBEDDINGS_MATRIX, axis=1) * np.linalg.norm(target_emb)
                    distances = 1 - (dot_products / norms)
                    
                    best_index = np.argmin(distances)
                    best_dist = distances[best_index]
                    
                    if best_dist <= DISTANCE_THRESH:
                        best_match = KNOWN_NAMES[best_index]

            # Calculate confidence for voting (0.0 to 1.0)
            confidence = max(0.0, 1.0 - best_dist) if best_match != "Unknown" else 0.0

            # 🖥️ Terminal logging for EVERY attempt (match or reject)
            if best_match != "Unknown":
                print(f"[BRAIN] ✅ Track {track_id} MATCHED: {best_match} (Dist: {best_dist:.4f} | Conf: {confidence:.2f})")
                try: result_queue.put_nowait((cam_name, track_id, best_match, confidence))
                except Exception: pass

                now = time.time()
                if now - recent_logs.get(best_match, 0) > WAITING_TIME: 
                    recent_logs[best_match] = now
                    print(f"\n📝 [ATTENDANCE LOG] {best_match} | Track {track_id} | {cam_name} | {timestamp}")
            else:
                print(f"[BRAIN] ❌ Track {track_id} REJECTED/UNKNOWN (Dist: {best_dist:.4f} > {DISTANCE_THRESH})")
                try: result_queue.put_nowait((cam_name, track_id, "Unknown", 0.0))
                except Exception: pass
                
        except ValueError:
            print(f"[BRAIN] ⚠️ Track {track_id}: No face detected in crop (skipped)")
        except Exception as e:
            print(f"[BRAIN] ❌ Track {track_id} Error: {e}")