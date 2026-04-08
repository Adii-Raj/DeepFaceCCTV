# config.py
import cv2

# Cameras
CAMERAS = {
    "CAM_1_ENTRANCE": "http://10.196.215.69:8080/video",
    #"CAM_2_HALLWAY": "http://10.148.100.223:8080/video", 
    "CAM_LOCAL": 0 
}

# AI Settings
DB_PATH = "my_database"
MODEL_NAME = "Facenet512"      # VGG-Face is lighter and faster for CPU than Facenet
DETECTOR_BACKEND = "skip"    # Set to 'skip' because 'eyes.py' already cropped the face!
DISTANCE_THRESH = 0.30

# System Settings
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'