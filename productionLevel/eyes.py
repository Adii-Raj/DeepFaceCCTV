# eyes.py
import cv2
import time
from config import HAAR_CASCADE_PATH

def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting vision on {cam_name}...")
    last_queued_time = 0
    
    cap = cv2.VideoCapture(url)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    frame_count = 0
    
    # These variables hold the data sent back from the AI Brain
    last_known_name = None
    last_match_time = 0
    
    while True:
        # 1. Listen for messages from the AI Brain
        try:
            while not result_queue.empty():
                res_cam, matched_name = result_queue.get_nowait()
                # If the AI result belongs to THIS camera, update the screen text
                if res_cam == cam_name:
                    last_known_name = matched_name
                    last_match_time = time.time()
        except Exception:
            pass

        # 2. Process the camera feed
        ret, frame = cap.read()
        if not ret:
            time.sleep(3)
            cap = cv2.VideoCapture(url)
            continue
            
        frame_count += 1
        display_frame = cv2.resize(frame, (640, 480))
        
        # Run fast detection
        # Run fast detection
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                current_time = time.time()
                # Check if 1 full second has passed since we last sent photos to the AI
                should_queue = (current_time - last_queued_time > 1.0)
                
                for (x, y, w, h) in faces:
                    pad = 20
                    h_orig, w_orig = display_frame.shape[:2]
                    y1, y2 = max(0, y - pad), min(h_orig, y + h + pad)
                    x1, x2 = max(0, x - pad), min(w_orig, x + w + pad)
                    
                    face_crop = display_frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        if should_queue:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            try:
                                face_queue.put_nowait((cam_name, face_crop, timestamp))
                                
                                # --- ADD THIS: Safely add +1 to the counter ---
                                with cache_counter.get_lock():
                                    cache_counter.value += 1
                                # ----------------------------------------------
                                
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(display_frame, "Analyzing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except Exception:
                                pass 
                        else:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            
                if should_queue:
                    last_queued_time = current_time


        # 3. DRAW THE UI DASHBOARD
        # Draw Camera Name
        cv2.rectangle(display_frame, (0, 0), (250, 40), (0,0,0), -1)
        cv2.putText(display_frame, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If the Brain found a match in the last 5 seconds, flash their name on screen!
        if last_known_name and (time.time() - last_match_time < 5):
            color = (0, 255, 0) if last_known_name != "Unknown" else (0, 0, 255)
            cv2.putText(display_frame, f"AI MATCH: {last_known_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow(f"Live Monitor: {cam_name}", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()