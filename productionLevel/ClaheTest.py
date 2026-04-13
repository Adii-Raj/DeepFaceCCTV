import cv2
import numpy as np
import os

def enhance_low_light_crop(img):
    """CLAHE + Adaptive Gamma for low-light face crops"""
    if img.size == 0:
        return img
        
    # Convert to LAB and apply CLAHE to Luminance channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab_enhanced = cv2.merge((l, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply gamma correction only if still too dark
    mean_brightness = np.mean(img_enhanced)
    if mean_brightness < 60:
        inv_gamma = 1.0 / 2.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img_enhanced = cv2.LUT(img_enhanced, table)
        
    return img_enhanced

def calculate_metrics(original, enhanced):
    """Calculate improvement metrics"""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Brightness improvement
    orig_mean = np.mean(orig_gray)
    enh_mean = np.mean(enh_gray)
    brightness_gain = enh_mean - orig_mean
    
    # Contrast improvement (standard deviation)
    orig_contrast = np.std(orig_gray)
    enh_contrast = np.std(enh_gray)
    contrast_gain = enh_contrast - orig_contrast
    
    # Sharpness (Laplacian variance)
    orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    enh_sharpness = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
    sharpness_gain = enh_sharpness - orig_sharpness
    
    return {
        'brightness_gain': brightness_gain,
        'contrast_gain': contrast_gain,
        'sharpness_gain': sharpness_gain,
        'orig_brightness': orig_mean,
        'enh_brightness': enh_mean
    }

# Test with sample images
test_dir = "test_faces"
os.makedirs(test_dir, exist_ok=True)

print("📸 Press 's' to save current frame from camera for testing")
print("📂 Or place test images in 'test_faces' folder")
print("❌ Press 'q' to quit\n")

cap = cv2.VideoCapture(0)  # Use your camera URL

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop center region (simulate face crop)
    h, w = frame.shape[:2]
    crop = frame[h//4:3*h//4, w//4:3*w//4]
    
    # Apply CLAHE
    enhanced = enhance_low_light_crop(crop)
    
    # Calculate metrics
    metrics = calculate_metrics(crop, enhanced)
    
    # Create comparison display
    comparison = np.hstack([crop, enhanced])
    
    # Add metrics text
    cv2.putText(comparison, f"Original - Brightness: {metrics['orig_brightness']:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(comparison, f"CLAHE - Brightness: {metrics['enh_brightness']:.1f}", 
                (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, f"Brightness Gain: +{metrics['brightness_gain']:.1f}", 
                (w//2 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(comparison, f"Contrast Gain: +{metrics['contrast_gain']:.1f}", 
                (w//2 + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("LEFT: Original | RIGHT: CLAHE Enhanced", comparison)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = int(time.time())
        cv2.imwrite(f"{test_dir}/original_{timestamp}.jpg", crop)
        cv2.imwrite(f"{test_dir}/enhanced_{timestamp}.jpg", enhanced)
        print(f"✅ Saved test images at {timestamp}")

cap.release()
cv2.destroyAllWindows()