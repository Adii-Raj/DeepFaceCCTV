import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Settings
DB_PATH = "my_database"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAM_INDEX = 0 # Change if using an IP cam or external webcam

class FaceEnrollerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Registration System")
        self.root.geometry("950x600")
        
        os.makedirs(DB_PATH, exist_ok=True)

        # --- Layout ---
        # Left Panel for Video
        self.left_frame = tk.Frame(self.root, width=500, bg='#2c3e50')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Right Panel for Captured Faces (Scrollable)
        self.right_frame = tk.Frame(self.root, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(pady=10, padx=10)

        self.capture_btn = tk.Button(self.left_frame, text="📸 Capture Faces in Frame", 
                                     command=self.capture_faces, font=("Arial", 14, "bold"), 
                                     bg="#27ae60", fg="white", cursor="hand2", pady=10)
        self.capture_btn.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(self.right_frame, text="Pending Unknown Faces", font=("Arial", 16, "bold")).pack(pady=5)

        # Scrollable Canvas Setup
        self.canvas = tk.Canvas(self.right_frame)
        self.scrollbar = tk.Scrollbar(self.right_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # --- OpenCV Setup ---
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.current_frame = None
        self.current_faces = []

        # Start the video loop
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Mirror the frame so it feels natural to the user
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Detect Faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            self.current_faces = faces

            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert to Tkinter compatible image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize for the UI
            img = img.resize((480, 360), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Re-run this function every 30 milliseconds
        self.root.after(30, self.update_video)

    def capture_faces(self):
        if self.current_frame is None or len(self.current_faces) == 0:
            return

        for (x, y, w, h) in self.current_faces:
            # Add some padding to the face crop
            pad = 20
            h_orig, w_orig = self.current_frame.shape[:2]
            y1, y2 = max(0, y - pad), min(h_orig, y + h + pad)
            x1, x2 = max(0, x - pad), min(w_orig, x + w + pad)

            face_crop = self.current_frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                self.add_face_to_ui(face_crop)

    def add_face_to_ui(self, face_img):
        # Convert crop to Tkinter image
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_face)
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        # Create a container row for this face
        row = tk.Frame(self.scrollable_frame, bd=1, relief=tk.SOLID, pady=5, padx=5)
        row.pack(fill=tk.X, pady=5, padx=5)

        lbl_img = tk.Label(row, image=imgtk)
        lbl_img.image = imgtk # Keep reference to prevent garbage collection
        lbl_img.pack(side=tk.LEFT, padx=5)

        # Entry for the name
        entry_name = tk.Entry(row, font=("Arial", 14), width=10)
        entry_name.pack(side=tk.LEFT, padx=5, fill=tk.Y, pady=30)
        entry_name.insert(0, "Name...")
        
        # Clear placeholder text on click
        entry_name.bind("<FocusIn>", lambda args: entry_name.delete('0', 'end') if entry_name.get() == "Name..." else None)

        # --- NEW: Delete Button ---
        # row.destroy instantly removes this specific entry from the UI without saving
        delete_btn = tk.Button(row, text="❌", bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), cursor="hand2",
                               command=row.destroy)
        delete_btn.pack(side=tk.RIGHT, padx=5, fill=tk.Y, pady=30)

        # Save Button
        save_btn = tk.Button(row, text="💾 Save", bg="#2980b9", fg="white", font=("Arial", 11, "bold"), cursor="hand2",
                             command=lambda: self.save_to_db(face_img, entry_name.get().strip(), row))
        save_btn.pack(side=tk.RIGHT, padx=5, fill=tk.Y, pady=30)


    def save_to_db(self, face_img, name, row_widget):
        # Validate name
        if not name or name.lower() == "name...":
            messagebox.showwarning("Invalid Input", "Please type a student's name before saving.")
            return

        # Format Name (e.g. "aditya" -> "Aditya")
        name = name.title()

        # Create folder if it doesn't exist
        person_dir = os.path.join(DB_PATH, name)
        os.makedirs(person_dir, exist_ok=True)

        # Figure out the next image number (img1.jpg, img2.jpg)
        existing_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_num = len(existing_files) + 1
        filename = os.path.join(person_dir, f"img{img_num}.jpg")

        # Save to disk
        cv2.imwrite(filename, face_img)
        print(f"✅ Saved new face for {name}: {filename}")

        # Remove this row from the UI
        row_widget.destroy()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEnrollerApp(root)
    root.mainloop()