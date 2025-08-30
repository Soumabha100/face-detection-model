import cv2
import os
import numpy as np

# --- Configuration ---
DATA_DIR = 'data/raw'
NUMBER_OF_SAMPLES = 100
IMAGE_SIZE = (100, 100)

os.makedirs(DATA_DIR, exist_ok=True)

def get_image_quality(image_gray):
    brightness = np.mean(image_gray)
    focus = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    min_brightness, max_brightness, min_focus = 60, 200, 100
    is_good = min_brightness < brightness < max_brightness and focus > min_focus
    return is_good, brightness, focus

def collect_faces():
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
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img_gray = gray[y:y+h, x:x+w]
            
            is_good, brightness, focus = get_image_quality(face_img_gray)
            
            frame_center_x = frame.shape[1] / 2
            face_center_x = x + w / 2
            is_centered = abs(frame_center_x - face_center_x) < 50

            if is_good and is_centered:
                img_path = os.path.join(person_dir, f"{sample_count + 1}.jpg")
                cv2.imwrite(img_path, cv2.resize(face_img_gray, IMAGE_SIZE))
                sample_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

            cv2.putText(frame, f"Brightness: {brightness:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Focus: {focus:.0f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        progress_text = f"Progress: {sample_count}/{NUMBER_OF_SAMPLES}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if sample_count < 25: instruction = "Look FORWARD and hold still."
        elif sample_count < 50: instruction = "Slowly turn your head LEFT."
        elif sample_count < 75: instruction = "Slowly turn your head RIGHT."
        else: instruction = "Tilt head UP and DOWN slowly."
        cv2.putText(frame, instruction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Collecting Faces...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {sample_count} samples for {name}.")

if __name__ == "__main__":
    collect_faces()