# config.py

# Cameras
CAMERAS = {
    #"CAM_LOCAL": "http://10.165.200.223:8080/video"
    #"Cam_Local":0
    "CCTV1":"rtsp://user:CCTV%2Auser@10.13.10.104:554/11"
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "Facenet512"      
DETECTOR_BACKEND = "skip"   
DISTANCE_THRESH = 0.40
YOLO_CONF = 0.25


# System Settings
NUM_BRAINS = 2
WAITING_TIME = 3             

# YOLO Camera Settings
YOLO_MODEL = "yolo8n_face.pt" #This is yolo face detecion model which reduce headache of croping face
AI_EVERY_N_FRAMES = 5        
AI_COOLDOWN_SECONDS = 1.0    

# --- NEW: Camera Hardware Settings ---
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Low-Light Enhancement (Conservative Mode)
ENHANCE_LOW_LIGHT = True
ENHANCE_MIN_BRIGHTNESS = 90  # Skip enhancement above this value (0-255)

# ⏱️ Timing & Throttling Controls
YOLO_RUN_EVERY_N_FRAMES = 1       # Run YOLO tracker every N frames (1 = every frame, 3 = ~10 FPS, saves CPU)
TRACK_SEND_COOLDOWN = 1.0         # Max 1 face crop sent to AI per track_id every X seconds (prevents queue spam)
TRACK_EXPIRY_SECONDS = 3.0        # Remove track from memory if not seen for X seconds (clean up exited people)
MAX_QUEUE_SIZE       = 1100   # max items in face_queue before dropping


# 🗳️ Smart Voting Settings
VOTE_WINDOW_SECONDS = 5.0     # Only consider votes from the last X seconds
HIGH_CONF_LOCK_THRESH = 0.75  # Confidence >= this instantly locks the name for the track