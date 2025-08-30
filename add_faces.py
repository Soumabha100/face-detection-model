import cv2
import os
import time
import math
import numpy as np

# --- Configuration ---
DB_PATH = "face_db"
os.makedirs(DB_PATH, exist_ok=True)

# --- Quality & Pose Configuration ---
BLUR_THRESHOLD = 60         # Adjusted for better usability
BRIGHTNESS_MIN = 50
BRIGHTNESS_MAX = 210
FACE_CENTER_BUFFER = 0.20   # % of screen width/height for "center" zone
FACE_SIDE_BUFFER = 0.35     # % of screen width for "left/right" zones
FACE_VERTICAL_BUFFER = 0.30 # % of screen height for "up/down" zones
CLOSER_THRESHOLD_FACTOR = 1.3 # Face must be 30% larger than average to be "closer"

# --- Pose Sequence Definition ---
# Defines the sequence of poses and how many pictures to take for each.
POSE_SEQUENCE = {
    "Center": 5,
    "Look Left": 3,
    "Look Right": 3,
    "Look Up": 3,
    "Look Down": 3,
    "Move Closer": 3
}
TARGET_SAMPLES = sum(POSE_SEQUENCE.values())

# --- Helper Functions for Quality Checks ---
def is_blurry(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def check_brightness(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < BRIGHTNESS_MIN: return "Too Dark"
    if mean_brightness > BRIGHTNESS_MAX: return "Too Bright"
    return "OK"

def save_faces_smartly():
    """Guides the user through a sequence of poses to capture high-quality, diverse face images."""
    name = input("Enter the person's name (no spaces, e.g., 'John_Doe'): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(DB_PATH, name)
    os.makedirs(person_dir, exist_ok=True)

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return
        
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    sample_count = 0
    pose_sequence_list = list(POSE_SEQUENCE.items())
    current_pose_index = 0
    current_pose_count = 0
    avg_center_face_width = 0
    center_widths = []

    while sample_count < TARGET_SAMPLES:
        ret, frame = video.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        current_pose, required_samples = pose_sequence_list[current_pose_index]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        status_text = "Searching..."
        box_color = (0, 0, 255) # Red for issues

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # --- Perform Checks ---
            brightness = check_brightness(face_img)
            blurry = is_blurry(face_img)
            
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            pose_ok = False
            # Check if current face position matches the required pose
            if current_pose == "Center":
                if (width * (0.5 - FACE_CENTER_BUFFER) < face_center_x < width * (0.5 + FACE_CENTER_BUFFER)) and \
                   (height * (0.5 - FACE_CENTER_BUFFER) < face_center_y < height * (0.5 + FACE_CENTER_BUFFER)):
                    pose_ok = True
                    center_widths.append(w) # Collect width for "closer" check
            elif current_pose == "Look Left":
                if face_center_x > width * (1 - FACE_SIDE_BUFFER): pose_ok = True
            elif current_pose == "Look Right":
                if face_center_x < width * FACE_SIDE_BUFFER: pose_ok = True
            elif current_pose == "Look Up":
                if face_center_y < height * FACE_VERTICAL_BUFFER: pose_ok = True
            elif current_pose == "Look Down":
                if face_center_y > height * (1 - FACE_VERTICAL_BUFFER): pose_ok = True
            elif current_pose == "Move Closer":
                if avg_center_face_width and w > avg_center_face_width * CLOSER_THRESHOLD_FACTOR:
                    pose_ok = True
            
            # --- Provide Feedback ---
            if not pose_ok:
                status_text = f"Please {current_pose}"
                box_color = (0, 165, 255) # Orange for wrong pose
            elif blurry:
                status_text = "BLURRY - Hold Still"
            elif brightness != "OK":
                status_text = brightness
            else:
                # All checks passed! Save the image.
                status_text = "OK! Capturing..."
                box_color = (0, 255, 0) # Green for good
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Smart Face Enrollment", frame)
                cv2.waitKey(500) # Give user a moment to see the feedback

                img_path = os.path.join(person_dir, f"{name}_{sample_count + 1}.jpg")
                cv2.imwrite(img_path, face_img)
                
                sample_count += 1
                current_pose_count += 1
                
                # Check if we are done with the current pose
                if current_pose_count >= required_samples:
                    if current_pose == "Center" and center_widths:
                        avg_center_face_width = np.mean(center_widths)
                    current_pose_index += 1
                    current_pose_count = 0
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif len(faces) > 1:
            status_text = "Multiple faces detected!"
        
        # --- Display UI on screen ---
        progress_text = f"Progress: {sample_count}/{TARGET_SAMPLES}"
        goal_text = f"CURRENT GOAL: {current_pose} ({current_pose_count}/{required_samples})"
        
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, goal_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Smart Face Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {sample_count} high-quality images for {name}.")

if __name__ == "__main__":
    save_faces_smartly()