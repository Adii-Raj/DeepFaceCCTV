# main.py
import multiprocessing as mp
# ADD THIS IMPORT:
import ctypes 
from config import CAMERAS,NUM_BRAINS
from eyes import start_camera_worker
from brain import start_ai_worker

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("🚀 Starting Enterprise Server...")
    
    face_queue = mp.Queue(maxsize=300)
    result_queue = mp.Queue(maxsize=300) 
    
    # --- ADD THIS: The Shared Cache Counter ---
    # 'i' stands for integer. It starts at 0.
    cache_counter = mp.Value('i', 0) 
    
    brain_processes = []
    
    print(f"🧠 Booting {NUM_BRAINS} AI Workers...")
    for i in range(NUM_BRAINS):
        # Pass the cache_counter to the brain
        p = mp.Process(target=start_ai_worker, args=(face_queue, result_queue, cache_counter), daemon=True)
        p.start()
        brain_processes.append(p)
    
    eye_processes = []
    for cam_name, url in CAMERAS.items():
        # Pass the cache_counter to the eyes
        p = mp.Process(target=start_camera_worker, args=(cam_name, url, face_queue, result_queue, cache_counter), daemon=True)
        p.start()
        eye_processes.append(p)
        
    try:
        for p in brain_processes:
            p.join()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")