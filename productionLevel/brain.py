# brain.py
import time
import os
import numpy as np
from deepface import DeepFace
from RealDeepFace.productionLevel.config import DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_THRESH

KNOWN_FACES = []

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_database():
    print("[BRAIN] ⏳ Booting AI and loading database...")
    DeepFace.build_model(MODEL_NAME)
    
    os.makedirs(DB_PATH, exist_ok=True)
    for person_folder in os.listdir(DB_PATH):
        person_path = os.path.join(DB_PATH, person_folder)
        if not os.path.isdir(person_path): continue 
            
        for file in os.listdir(person_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    res = DeepFace.represent(
                        img_path=os.path.join(person_path, file), 
                        model_name=MODEL_NAME, 
                        enforce_detection=False 
                    )
                    KNOWN_FACES.append({"name": person_folder, "embedding": res[0]["embedding"]})
                    print(f"[BRAIN] ✅ Loaded: {person_folder}")
                except Exception as e:
                    pass

def start_ai_worker(face_queue, result_queue, cache_counter): # <-- Added result_queue here
    load_database()
    print("[BRAIN] 🧠 AI Worker active. Waiting for faces...")
    
    recent_logs = {}
    
    while True:
        cam_name, face_crop, timestamp = face_queue.get()
        
        # --- ADD THIS: Safely subtract -1 and print the status ---
        with cache_counter.get_lock():
            cache_counter.value -= 1
            current_cache = cache_counter.value
            
        print(f"⚙️ [PROCESSING] AI grabbing face from {cam_name}. (Faces waiting in cache: {current_cache})")
        # ---------------------------------------------------------

        try:
            res = DeepFace.represent(img_path=face_crop, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
            target_emb = res[0]["embedding"]
            best_match, best_dist = "Unknown", float("inf")
            
            for known in KNOWN_FACES:
                dist = cosine_distance(target_emb, known["embedding"])
                if dist < best_dist: 
                    best_dist, best_match = dist, known["name"] 
            
            if best_dist <= DISTANCE_THRESH:
                
                # --- TO REMOVE LATER: Delete or comment out these next 2 lines ---
                print(f"🖥️ [TEST NOTIFICATION]: Face matched with '{best_match}' on {cam_name}!")
                # ------------------------------------------------------------------

                # Send the name back to the live video feed!
                try: result_queue.put_nowait((cam_name, best_match))
                except Exception: pass

                # Ensure we only log attendance once every 30 seconds
                now = time.time()
                if now - recent_logs.get(best_match, 0) > 30: 
                    recent_logs[best_match] = now
                    print(f"\n✅ [ATTENDANCE LOG] {best_match} walked past {cam_name} at {timestamp}")
            else:
                try: result_queue.put_nowait((cam_name, "Unknown"))
                except Exception: pass
                
        except Exception as e:
            pass