# brain.py
import time
import os
import cv2
import threading
import numpy as np
from deepface import DeepFace
from config import DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_THRESH, WAITING_TIME

# Global Matrix Variables
KNOWN_NAMES = []
KNOWN_EMBEDDINGS_MATRIX = None
processed_files = set()
db_lock = threading.Lock() 

def process_new_faces():
    """Scans for new faces, generates embeddings, and rebuilds the Math Matrix."""
    global KNOWN_NAMES, KNOWN_EMBEDDINGS_MATRIX, processed_files
    os.makedirs(DB_PATH, exist_ok=True)
    
    new_faces_added = False
    temp_embeddings = []
    
    # If we already have a matrix, pull its data so we don't lose it
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
                    
    # Rebuild the ultra-fast NumPy Matrix if new faces were added
    if new_faces_added:
        with db_lock:
            KNOWN_EMBEDDINGS_MATRIX = np.array(temp_embeddings)

def dynamic_db_updater():
    """Background heartbeat to hot-load new students."""
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
        cam_name, face_crop, timestamp = face_queue.get()
        
        with cache_counter.get_lock():
            cache_counter.value -= 1
            current_cache = cache_counter.value
            
        print(f"⚙️ [PROCESSING] AI grabbing crop from {cam_name}. (Cache: {current_cache})")

        try:
            # We keep align=True for CCTV angles, but anti_spoofing=False for speed
            res = DeepFace.represent(
                img_path=face_crop, 
                model_name=MODEL_NAME, 
                detector_backend="skip", #earlier: DETECTOR_BACKEND
                enforce_detection=False, 
                align=False 
            )
                
            target_emb = np.array(res[0]["embedding"])
            best_match, best_dist = "Unknown", float("inf")
            
            with db_lock:
                if KNOWN_EMBEDDINGS_MATRIX is not None and len(KNOWN_NAMES) > 0:
                    # --- THE ENTERPRISE MATH UPGRADE ---
                    # Checks 1,500 faces simultaneously in 0.002 seconds
                    dot_products = np.dot(KNOWN_EMBEDDINGS_MATRIX, target_emb)
                    norms = np.linalg.norm(KNOWN_EMBEDDINGS_MATRIX, axis=1) * np.linalg.norm(target_emb)
                    distances = 1 - (dot_products / norms)
                    
                    best_index = np.argmin(distances)
                    best_dist = distances[best_index]
                    
                    if best_dist <= DISTANCE_THRESH:
                        best_match = KNOWN_NAMES[best_index]
            
            if best_match != "Unknown":
                try: result_queue.put_nowait((cam_name, best_match))
                except Exception: pass

                now = time.time()
                if now - recent_logs.get(best_match, 0) > WAITING_TIME: 
                    recent_logs[best_match] = now
                    print(f"\n✅ [ATTENDANCE LOG] {best_match} walked past {cam_name} at {timestamp} (Accuracy: {1-best_dist:.2f})")
            else:
                try: result_queue.put_nowait((cam_name, "Unknown"))
                except Exception: pass
                
        except ValueError:
            # DeepFace couldn't find a face in the YOLO crop (e.g. they turned backwards)
            pass
        except Exception as e:
            pass