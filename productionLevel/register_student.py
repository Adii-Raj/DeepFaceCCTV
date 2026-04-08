# register_student.py
import cv2
import os
import time
from config import DB_PATH

def register():
    student_name = input("Enter Student Name to register: ").strip().replace(" ", "_")
    if not student_name:
        print("Invalid Name!")
        return

    save_path = os.path.join(DB_PATH, student_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"\n📸 Registering {student_name}...")
    print("Instructions: Press 'S' to Save Photo | Press 'Q' to Quit")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        height, width, _ = frame.shape
        cv2.rectangle(display_frame, (width//2-150, height//2-150), (width//2+150, height//2+150), (255, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {count}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Student Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            count += 1
            full_path = os.path.join(save_path, f"{student_name}_{int(time.time())}.jpg")
            cv2.imwrite(full_path, frame)
            print(f"✅ Saved image {count} to {full_path}")
            
            if count >= 3:
                print(f"\n🎉 Registration Complete for {student_name}!")
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register()