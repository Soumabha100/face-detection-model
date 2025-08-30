import cv2
import os
import numpy as np
import math
import time

# --- Configuration ---
DB_PATH = "face_db"
TARGET_SAMPLES = 20 # Total images to collect
os.makedirs(DB_PATH, exist_ok=True)

# --- Quality Thresholds (Can be adjusted) ---
BLUR_THRESHOLD = 75  # Lower value -> more tolerant to blur
BRIGHTNESS_MIN = 60  # Reject images that are too dark
BRIGHTNESS_MAX = 200 # Reject images that are too bright

# --- Function to check for blur ---
def is_blurry(face_roi):
    """Calculates the variance of the Laplacian to detect blur."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < BLUR_THRESHOLD

# --- Function to check brightness ---
def check_brightness(face_roi):
    """Checks if the image brightness is within an acceptable range."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < BRIGHTNESS_MIN:
        return "Too Dark"
    if mean_brightness > BRIGHTNESS_MAX:
        return "Too Bright"
    return "OK"

def save_faces_smartly():
    """Captures and saves face images intelligently by checking quality and pose."""
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
        
    # Get frame dimensions for grid calculation
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # --- Pose Diversity Logic ---
    # Divide the screen into a 3x3 grid
    grid_zones = {(i, j): 0 for i in range(3) for j in range(3)}
    max_per_zone = math.ceil(TARGET_SAMPLES / len(grid_zones))
    
    zone_prompts = {
        (0, 0): "Look Top-Left", (0, 1): "Look Top-Center", (0, 2): "Look Top-Right",
        (1, 0): "Look Left",     (1, 1): "Look Center",     (1, 2): "Look Right",
        (2, 0): "Look Bottom-Left",(2, 1): "Look Bottom-Center",(2, 2): "Look Bottom-Right"
    }

    sample_count = 0
    while sample_count < TARGET_SAMPLES:
        ret, frame = video.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        status_text = "Searching for face..."
        
        # We only proceed if exactly one face is detected
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # --- Quality and Pose Checks ---
            brightness_status = check_brightness(face_img)
            blur_status = is_blurry(face_img)
            
            # Determine which grid zone the center of the face is in
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            zone_x = min(int(face_center_x / (width / 3)), 2)
            zone_y = min(int(face_center_y / (height / 3)), 2)
            current_zone = (zone_y, zone_x)

            if grid_zones[current_zone] >= max_per_zone:
                status_text = "Please change position"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2) # Orange rectangle
            elif blur_status:
                status_text = "BLURRY - Hold Still"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rectangle
            elif brightness_status != "OK":
                status_text = brightness_status
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rectangle
            else:
                # All checks passed, save the image
                status_text = "Looks Good! Saving..."
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle
                
                img_path = os.path.join(person_dir, f"{name}_{sample_count + 1}.jpg")
                cv2.imwrite(img_path, face_img)
                
                sample_count += 1
                grid_zones[current_zone] += 1
                time.sleep(0.5) # Pause briefly after saving an image
        
        elif len(faces) > 1:
            status_text = "Multiple faces detected!"
        
        # --- Display UI/Feedback on the screen ---
        # Display progress and status
        progress_text = f"Progress: {sample_count}/{TARGET_SAMPLES}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display grid prompts
        for zone, count in grid_zones.items():
            if count < max_per_zone:
                prompt = zone_prompts[zone]
                cv2.putText(frame, prompt, (10, 90 + (len(zone_prompts) - list(zone_prompts.keys()).index(zone) - 1) * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Smart Face Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {sample_count} high-quality images for {name}.")

if __name__ == "__main__":
    save_faces_smartly()