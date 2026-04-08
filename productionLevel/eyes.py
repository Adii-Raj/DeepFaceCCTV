# eyes.py
import cv2
import time
import queue
import math
from ultralytics import YOLO
from config import YOLO_MODEL, AI_EVERY_N_FRAMES, AI_COOLDOWN_SECONDS

def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting YOLOv8 vision on {cam_name}...")
    
    # This will automatically download yolov8n.pt the first time it runs
    model = YOLO(YOLO_MODEL)
    
    cap = cv2.VideoCapture(url)
    
    frame_count = 0
    last_known_name = None
    last_match_time = 0
    recent_locations = [] 
    
    while True:
        try:
            while not result_queue.empty():
                res_cam, matched_name = result_queue.get_nowait()
                if res_cam == cam_name:
                    last_known_name = matched_name
                    last_match_time = time.time()
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret:
            time.sleep(3)
            cap = cv2.VideoCapture(url)
            continue
            
        frame_count += 1
        display_frame = cv2.resize(frame, (640, 480))
        h_orig, w_orig = display_frame.shape[:2]
        
        if frame_count % AI_EVERY_N_FRAMES == 0:
            current_time = time.time()
            recent_locations = [loc for loc in recent_locations if current_time - loc['time'] < AI_COOLDOWN_SECONDS]
            
            # Run YOLOv8 specifically looking for "Class 0" (People)
            results = model.predict(source=display_frame, classes=[0], verbose=False, conf=0.4)
            
            for box in results[0].boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                cx, cy = x1 + (w / 2), y1 + (h / 2)
                
                is_new_person = True
                for loc in recent_locations:
                    prev_cx, prev_cy = loc['center']
                    if math.hypot(cx - prev_cx, cy - prev_cy) < 100: 
                        is_new_person = False
                        break
                
                # YOLO detects the whole body. Let's crop the upper half (head/shoulders) 
                # so DeepFace doesn't have to search their shoes for a face.
                head_y2 = int(y1 + (h * 0.5)) 
                
                pad = 20
                crop_y1, crop_y2 = max(0, y1 - pad), min(h_orig, head_y2 + pad)
                crop_x1, crop_x2 = max(0, x1 - pad), min(w_orig, x2 + pad)
                
                person_crop = display_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if person_crop.size > 0:
                    if is_new_person:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            face_queue.put_nowait((cam_name, person_crop, timestamp))
                            with cache_counter.get_lock():
                                cache_counter.value += 1
                            recent_locations.append({'center': (cx, cy), 'time': current_time})
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, "To AI...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except queue.Full:
                            pass
                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 4. DRAW THE UI DASHBOARD
        cv2.rectangle(display_frame, (0, 0), (250, 40), (0,0,0), -1)
        cv2.putText(display_frame, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if last_known_name and (time.time() - last_match_time < 5):
            color = (0, 255, 0) if last_known_name != "Unknown" else (0, 0, 255)
            cv2.putText(display_frame, f"AI MATCH: {last_known_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow(f"Live Monitor: {cam_name}", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()