# config.py
import cv2

# Cameras
CAMERAS = {
    #"CAM_2_HALLWAY": "http://10.148.100.223:8080/video", 
    "CAM_LOCAL": 0 
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "Facenet512"      # VGG-Face is lighter and faster for CPU than Facenet
# You can use "skip" if you want fast detection, retina face is used for cctv angle
# Also if you remove retinface then also remove align=true from brain.py
DETECTOR_BACKEND = "retinaface"    # Set to 'skip' because 'eyes.py' already cropped the face!
DISTANCE_THRESH = 0.30
AI_EVERY_N_FRAMES = 3        # CPU Power Saver: How often the camera scans for a face
AI_COOLDOWN_SECONDS = 1.0    # Queue Protector: Wait 1 second before snapping the same person again
NUM_BRAINS = 2
WAITING_TIME = 5      # It add delay in brain.py from marking same student multiple time.

# System Settings
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'