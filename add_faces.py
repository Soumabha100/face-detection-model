import cv2
import os
import numpy as np

# --- Configuration ---
DATA_DIR = 'data/raw'
NUMBER_OF_SAMPLES = 100  # Number of images to collect per person
IMAGE_SIZE = (100, 100) # Standardize image size

# Ensure the main data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_image_quality(image):
    """
    Analyzes an image to determine its quality based on brightness and blurriness.
    Returns a tuple: (is_good_quality, brightness, focus)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Brightness check (mean of pixel intensities)
    brightness = np.mean(gray)
    
    # Focus check (Laplacian variance)
    focus = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Define acceptable thresholds
    min_brightness = 60
    max_brightness = 200
    min_focus = 100  # Adjust this threshold based on camera quality
    
    is_good = min_brightness < brightness < max_brightness and focus > min_focus
    return is_good, brightness, focus

def collect_faces():
    """
    Main function to guide the user through the face sample collection process.
    """
    name = input("Enter the person's name (no spaces, e.g., 'John_Doe'): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    
    sample_count = 0
    while sample_count < NUMBER_OF_SAMPLES:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a more natural mirror-like view
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fine-tuned face detection
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # --- Image Quality and Centering Checks ---
            is_good, brightness, focus = get_image_quality(face_img)
            
            # Center check
            frame_center_x = frame.shape[1] / 2
            face_center_x = x + w / 2
            is_centered = abs(frame_center_x - face_center_x) < 50 # Allow 50 pixels deviation

            if is_good and is_centered:
                # Save the high-quality, centered face
                img_path = os.path.join(person_dir, f"{sample_count + 1}.jpg")
                cv2.imwrite(img_path, cv2.resize(face_img, IMAGE_SIZE))
                sample_count += 1
                
                # Visual feedback for successful capture
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) # Green box
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) # Red box

            # Display quality stats
            cv2.putText(frame, f"Brightness: {brightness:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Focus: {focus:.0f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Display instructions and progress ---
        progress_text = f"Progress: {sample_count}/{NUMBER_OF_SAMPLES}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if sample_count < 25:
            instruction = "Look FORWARD and hold still."
        elif sample_count < 50:
            instruction = "Slowly turn your head LEFT."
        elif sample_count < 75:
            instruction = "Slowly turn your head RIGHT."
        else:
            instruction = "Tilt head UP and DOWN slowly."
        
        cv2.putText(frame, instruction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Collecting Faces...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {sample_count} samples for {name}.")

if __name__ == "__main__":
    collect_faces()

