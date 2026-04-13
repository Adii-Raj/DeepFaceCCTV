# eyes.py
import cv2
import time
import queue
import math
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, AI_EVERY_N_FRAMES, AI_COOLDOWN_SECONDS, CAMERA_WIDTH, CAMERA_HEIGHT, YOLO_CONF,ENHANCE_LOW_LIGHT


def enhance_low_light_crop(img):
    """Conservative CLAHE + Gamma only for truly dark faces"""
    if img.size == 0:
        return img
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # ✅ SKIP if already well-lit (prevents VGG-Face degradation)
    if mean_brightness > 90:
        return img
        
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ✅ Conservative CLAHE settings
    clip = 1.5 if mean_brightness < 40 else 1.2
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16, 16))
    l = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # ✅ Mild gamma ONLY for very dark faces
    if mean_brightness < 50:
        gamma = 1.5  # Softer than 2.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img_enhanced = cv2.LUT(img_enhanced, table)
        
    return img_enhanced


def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting YOLOv8 vision on {cam_name}...")
    
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(url)
    
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
                
                # yolov8n-face.pt draws box directly on face, no head guessing needed
                pad = 20
                crop_y1, crop_y2 = max(0, y1 - pad), min(h_orig, y2 + pad)
                crop_x1, crop_x2 = max(0, x1 - pad), min(w_orig, x2 + pad)
                
                raw_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                # ✅ Only enhance if enabled AND image is dark enough
                if globals().get('ENHANCE_LOW_LIGHT', False) and np.mean(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)) < 90:
                    person_crop = enhance_low_light_crop(raw_crop)
                else:
                    person_crop = raw_crop  # Keep original for well-lit faces
                
                if person_crop.size > 0 and person_crop.shape[0] > 30 and person_crop.shape[1] > 30:
                    if is_new_person:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        try:
                            # ✅ FIXED: Removed cv2.resize upscaling trap. Send sharp native pixels.
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

        cv2.rectangle(frame, (0, 0), (400, 100), (0,0,0), -1)
        cv2.putText(frame, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        if last_known_name and (time.time() - last_match_time < 5):
            color = (0, 255, 0) if last_known_name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"AI MATCH: {last_known_name}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow(f"Live Monitor: {cam_name}", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()