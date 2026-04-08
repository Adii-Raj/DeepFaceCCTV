# eyes.py
import cv2
import time
import queue
import math
import mediapipe as mp
from config import AI_EVERY_N_FRAMES, AI_COOLDOWN_SECONDS

def start_camera_worker(cam_name, url, face_queue, result_queue, cache_counter):
    print(f"[EYES] 👀 Starting MediaPipe vision on {cam_name}...")
    
    cap = cv2.VideoCapture(url)
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    # model_selection=1 is optimized for FAR faces (up to 5 meters). 0 is for close faces (like webcams).
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    frame_count = 0
    last_known_name = None
    last_match_time = 0
    recent_locations = [] 
    
    while True:
        # 1. Listen for messages from the AI Brain
        try:
            while not result_queue.empty():
                res_cam, matched_name = result_queue.get_nowait()
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
        h_orig, w_orig = display_frame.shape[:2]
        
        # 3. Run fast detection
        if frame_count % AI_EVERY_N_FRAMES == 0:
            current_time = time.time()
            recent_locations = [loc for loc in recent_locations if current_time - loc['time'] < AI_COOLDOWN_SECONDS]
            
            # MediaPipe requires RGB images
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    # MediaPipe returns relative coordinates (percentages). Convert to pixels.
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w_orig)
                    y = int(bboxC.ymin * h_orig)
                    w = int(bboxC.width * w_orig)
                    h = int(bboxC.height * h_orig)
                    
                    cx, cy = x + (w / 2), y + (h / 2)
                    
                    # Check if someone is already standing in this exact spot
                    is_new_face = True
                    for loc in recent_locations:
                        prev_cx, prev_cy = loc['center']
                        distance = math.hypot(cx - prev_cx, cy - prev_cy)
                        if distance < 100: 
                            is_new_face = False
                            break
                    
                    # Setup the crop boundaries with padding
                    pad = 20
                    y1, y2 = max(0, y - pad), min(h_orig, y + h + pad)
                    x1, x2 = max(0, x - pad), min(w_orig, x + w + pad)
                    face_crop = display_frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        if is_new_face:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            try:
                                face_queue.put_nowait((cam_name, face_crop, timestamp))
                                with cache_counter.get_lock():
                                    cache_counter.value += 1
                                recent_locations.append({'center': (cx, cy), 'time': current_time})
                                
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(display_frame, "Analyzing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except queue.Full:
                                pass
                        else:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # 4. DRAW THE UI DASHBOARD
        cv2.rectangle(display_frame, (0, 0), (250, 40), (0,0,0), -1)
        cv2.putText(display_frame, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if last_known_name and (time.time() - last_match_time < 5):
            color = (0, 255, 0) if last_known_name != "Unknown" else (0, 0, 255)
            # Added "Spoof" indicator support for the brain.py changes below
            if last_known_name == "SPOOF_DETECTED":
                cv2.putText(display_frame, "WARNING: FAKE FACE/PAINTING", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"AI MATCH: {last_known_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow(f"Live Monitor: {cam_name}", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()