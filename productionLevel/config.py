# config.py

# Cameras
CAMERAS = {
    "CAM_LOCAL": 0
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "ArcFace"      
DETECTOR_BACKEND = "skip"   
DISTANCE_THRESH = 0.50
YOLO_CONF = 0.25


# System Settings
NUM_BRAINS = 1
WAITING_TIME = 3             

# YOLO Camera Settings
YOLO_MODEL = "yolo8n_face.pt" #This is yolo face detecion model which reduce headache of croping face
AI_EVERY_N_FRAMES = 5        
AI_COOLDOWN_SECONDS = 1.0    

# --- NEW: Camera Hardware Settings ---
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080