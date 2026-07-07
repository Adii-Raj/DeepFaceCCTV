cd "/mnt/c/Users/Amrita/OneDrive/my drive/OneDrive/Desktop/CCTVproject/DeepFaceCCTV"

python3 -c "
import cv2
import numpy as np
from core.detector import YuNetDetector

# Load detector with low threshold
detector = YuNetDetector.from_path('models/face_detection_yunet_2023mar.onnx', score_threshold=0.1)

# Open webcam
cap = cv2.VideoCapture(0)
print(f'Webcam opened: {cap.isOpened()}')

if cap.isOpened():
    # Read a few frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f'Frame {i}: failed to read')
            continue
        
        print(f'Frame {i}: shape={frame.shape}, mean={frame.mean():.1f}')
        
        # Detect faces
        faces = detector.detect(frame)
        print(f'  Raw detections: {len(faces)}')
        
        if len(faces) > 0:
            for j, face in enumerate(faces):
                print(f'  Face {j}: conf={face[14]:.3f}, bbox=[{face[0]:.1f}, {face[1]:.1f}, {face[2]:.1f}, {face[3]:.1f}]')
        print()
        
cap.release()
"