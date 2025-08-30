import cv2
import os

# --- Configuration ---
DB_PATH = "face_db"  # Directory to store the face images
SAMPLES_PER_PERSON = 20 # We need fewer samples for deepface

os.makedirs(DB_PATH, exist_ok=True)

def save_faces():
    """Captures and saves face images for a new person."""
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

    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    sample_count = 0
    while sample_count < SAMPLES_PER_PERSON:
        ret, frame = video.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            
            # Save the detected face
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{name}_{sample_count + 1}.jpg")
            cv2.imwrite(img_path, face_img)
            
            sample_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display progress
        progress_text = f"Progress: {sample_count}/{SAMPLES_PER_PERSON}"
        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Adding Faces...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"\nCollected {sample_count} images for {name} in '{DB_PATH}/{name}/'")

if __name__ == "__main__":
    save_faces()