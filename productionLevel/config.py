# config.py

# Cameras
CAMERAS = {
    # "CAM_1_DOOR": "http://10.148.100.223:8080/video", 
    "CAM_LOCAL": 0 
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "Facenet512"      
# Changed from 'skip' to 'opencv' because YOLO sends a body/head crop, 
# so DeepFace needs to quickly grab the exact face from that crop.
DETECTOR_BACKEND = "opencv"    
DISTANCE_THRESH = 0.35

# System Settings
NUM_BRAINS = 3
WAITING_TIME = 5             # Cooldown before logging the same student again

# YOLO Camera Settings
YOLO_MODEL = "yolov8n.pt"    # The ultra-fast YOLOv8 Nano model
AI_EVERY_N_FRAMES = 2        # Check almost every frame because YOLO is fast!
AI_COOLDOWN_SECONDS = 1.0    # Wait 1 sec before tracking the same spot again