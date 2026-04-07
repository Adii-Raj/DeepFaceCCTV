import cv2
import threading
import time
import os
from deepface import DeepFace

# Ensure the database folder exists
if not os.path.exists("my_database"):
    os.makedirs("my_database")

# Global variables to share data between the camera and the AI
current_frame = None
recognized_faces = []
lock = threading.Lock()

def ai_brain():
    """Background thread that analyzes frames and auto-saves unknown people."""
    global current_frame, recognized_faces
    
    while True:
        if current_frame is not None:
            with lock:
                frame_to_process = current_frame.copy()
            
            try:
                # 1. Strictly detect all faces in the frame
                try:
                    faces = DeepFace.extract_faces(img_path=frame_to_process, enforce_detection=True)
                except ValueError:
                    # If no faces are found, it throws a ValueError. We catch it and move on.
                    faces = [] 
                
                new_faces = []
                
                for face_obj in faces:
                    # Get the exact coordinates of the face
                    area = face_obj['facial_area']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    
                    # Crop just the face out of the larger camera frame
                    face_crop = frame_to_process[y:y+h, x:x+w]
                    
                    if face_crop.size == 0:
                        continue
                        
                    # 2. Check if this specific cropped face matches anyone in the database
                    dfs = DeepFace.find(
                        img_path=face_crop, 
                        db_path="my_database", 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    df = dfs[0] if len(dfs) > 0 else None
                    
                    if df is not None and not df.empty:
                        # MATCH FOUND: Extract their name from the file path
                        file_path = df.iloc[0]['identity']
                        name = os.path.basename(file_path).split('.')[0]
                    else:
                        # NO MATCH: This is an unknown person!
                        # Create a unique name using a timestamp so files don't overwrite
                        timestamp = int(time.time() * 100) 
                        name = f"Unknown_{timestamp}"
                        
                        # Save the cropped face into the database folder
                        new_path = os.path.join("my_database", f"{name}.jpg")
                        cv2.imwrite(new_path, face_crop)
                        print(f"⚠️ New face detected! Auto-saved as {name}.jpg")
                        
                    # Add to the list to be drawn on screen
                    new_faces.append({"name": name, "box": (x, y, w, h)})
                
                # Safely update the global list for the camera thread
                with lock:
                    recognized_faces = new_faces

            except Exception as e:
                pass # Silently skip frames with weird errors to keep the CCTV running

# --- MAIN CAMERA LOOP ---

# Start the AI brain in a background thread
ai_thread = threading.Thread(target=ai_brain, daemon=True)
ai_thread.start()

# Turn on the webcam
cap = cv2.VideoCapture(0)
print("CCTV Camera Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Send the newest frame to the AI brain
    with lock:
        current_frame = frame.copy()
        boxes_to_draw = recognized_faces.copy()

    # Draw boxes for anyone the AI found
    for face in boxes_to_draw:
        x, y, w, h = face["box"]
        name = face["name"]
        
        # Color coding: Red for Unknowns, Green for Knowns
        if "Unknown" in name:
            color = (0, 0, 255) # Red in BGR
        else:
            color = (0, 255, 0) # Green in BGR
            
        # Draw the rectangle and the name text
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the smooth, continuous video feed
    cv2.imshow("CCTV Feed", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()