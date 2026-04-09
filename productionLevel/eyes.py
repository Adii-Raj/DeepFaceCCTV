# eyes.py
import cv2
import time
import queue
import math
from ultralytics import YOLO
from config import YOLO_MODEL, AI_EVERY_N_FRAMES, AI_COOLDOWN_SECONDS, CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_CONF

def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting YOLOv8 vision on {cam_name}...")
    
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(url)
    
    # 1. Force the camera hardware to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
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
        h_orig, w_orig = frame.shape[:2]
        
        if frame_count % AI_EVERY_N_FRAMES == 0:
            current_time = time.time()
            recent_locations = [loc for loc in recent_locations if current_time - loc['time'] < AI_COOLDOWN_SECONDS]
            
            # 2. Run YOLO directly on the 1080p frame
            results = model.predict(source=frame, classes=[0], verbose=False, conf=YOLO_CONF)
            
            for box in results[0].boxes:
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
                
                # 3. Crop Logic & Padding for 1080p
                head_y2 = int(y1 + (h * 0.35)) # Upper half of the body
                pad = 30 # Increased padding slightly for higher resolution
                
                crop_y1, crop_y2 = max(0, y1 - pad), min(h_orig, head_y2 + pad)
                crop_x1, crop_x2 = max(0, x1 - pad), min(w_orig, x2 + pad)
                
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Filter out tiny crops that are too blurry anyway
                if person_crop.size > 0 and person_crop.shape[0] > 30 and person_crop.shape[1] > 30:
                    if is_new_person:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            # ✅ Upscale small/distant face crops before sending to Brain
                            target_h = 160
                            if person_crop.shape[0] < target_h:
                                scale = target_h / person_crop.shape[0]
                                new_w = int(person_crop.shape[1] * scale)
                                person_crop = cv2.resize(person_crop, (new_w, target_h),
                                                        interpolation=cv2.INTER_CUBIC)

                            # Send the upscaled crop to the Brain
                            face_queue.put_nowait((cam_name, person_crop, timestamp))
                            with cache_counter.get_lock():
                                cache_counter.value += 1
                            recent_locations.append({'center': (cx, cy), 'time': current_time})

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, "To AI...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        except queue.Full:
                            pass
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                        

        # 4. Draw UI on the 1080p frame (using thicker fonts/lines so it scales well)
        cv2.rectangle(frame, (0, 0), (400, 100), (0,0,0), -1)
        cv2.putText(frame, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        if last_known_name and (time.time() - last_match_time < 5):
            color = (0, 255, 0) if last_known_name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"AI MATCH: {last_known_name}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 5. Shrink JUST the display monitor so it fits on your screen (e.g., 50% scale)
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow(f"Live Monitor: {cam_name}", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()