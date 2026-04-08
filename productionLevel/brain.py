# brain.py
import time
import os
import numpy as np
from deepface import DeepFace
from config import DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_THRESH, WAITING_TIME

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

def start_ai_worker(face_queue, result_queue, cache_counter):
    load_database()
    print("[BRAIN] 🧠 AI Worker active. Waiting for faces...")
    
    recent_logs = {}
    
    while True:
        cam_name, face_crop, timestamp = face_queue.get()
        
        with cache_counter.get_lock():
            cache_counter.value -= 1
            current_cache = cache_counter.value
            
        print(f"⚙️ [PROCESSING] AI grabbing face from {cam_name}. (Faces waiting in cache: {current_cache})")

        try:
            # Run DeepFace with Anti-Spoofing on the LIVE face
            res = DeepFace.represent(
                img_path=face_crop, 
                model_name=MODEL_NAME, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False,
                anti_spoofing=True,
                align=True  #You can remove this line(it is used to scan 3d view)
            )
            
            # Check if DeepFace thinks this is a painting or a photo
            is_real = res[0].get("is_real", True) 
            
            if not is_real:
                print(f"⚠️ [SECURITY] Detected a fake face / painting on {cam_name} at {timestamp}")
                try: result_queue.put_nowait((cam_name, "SPOOF_DETECTED"))
                except Exception: pass
                continue # Skip the rest of the matching logic
                
            target_emb = res[0]["embedding"]
            best_match, best_dist = "Unknown", float("inf")
            
            for known in KNOWN_FACES:
                dist = cosine_distance(target_emb, known["embedding"])
                if dist < best_dist: 
                    best_dist, best_match = dist, known["name"] 
            
            if best_dist <= DISTANCE_THRESH:
                
                print(f"🖥️ [TEST NOTIFICATION]: Face matched with '{best_match}' on {cam_name}!")

                try: result_queue.put_nowait((cam_name, best_match))
                except Exception: pass

                # Ensure we only log attendance once every 30 seconds
                now = time.time()
                if now - recent_logs.get(best_match, 0) > WAITING_TIME: 
                    recent_logs[best_match] = now
                    print(f"\n✅ [ATTENDANCE LOG] {best_match} walked past {cam_name} at {timestamp}")
            else:
                try: result_queue.put_nowait((cam_name, "Unknown"))
                except Exception: pass
                
        except Exception as e:
            pass