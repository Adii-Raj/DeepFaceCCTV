# config.py

# Cameras
CAMERAS = {
    "CAM_LOCAL": 0 
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "Facenet512"      
DETECTOR_BACKEND = "opencv"    
DISTANCE_THRESH = 0.35

# System Settings
NUM_BRAINS = 3
WAITING_TIME = 5             

# YOLO Camera Settings
YOLO_MODEL = "yolov8n.pt"    
AI_EVERY_N_FRAMES = 2        
AI_COOLDOWN_SECONDS = 1.0    

# --- NEW: Camera Hardware Settings ---
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080