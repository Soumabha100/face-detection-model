import cv2
import os
from deepface import DeepFace
import time

# --- Configuration ---
DB_PATH = "face_db"
# --- OPTIMIZATION 1: Use a much lighter and faster model ---
MODEL_NAME = "SFace" 
DETECTOR_BACKEND = 'opencv'

def run_realtime_recognition():
    """
    Performs optimized real-time face recognition.
    """
    print("Starting optimized real-time recognition...")
    print(f"Using model '{MODEL_NAME}' and database path '{DB_PATH}'")
    
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        print(f"Error: The database path '{DB_PATH}' is empty.")
        print("Please run add_faces.py to add images first.")
        return

    # Pre-load the face detection model
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Pre-load the DeepFace model to prevent lag on the first detection
    print("Loading DeepFace model...")
    DeepFace.build_model(MODEL_NAME)
    print("Model loaded.")

    video = cv2.VideoCapture(0)
    prev_frame_time = 0
    
    # --- OPTIMIZATION 2: Variables for periodic recognition ---
    recognition_counter = 0
    RECOGNITION_INTERVAL = 10 # Recognize face every 10 frames
    last_known_name = "Unknown"

    print("System started. Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret: 
            break

        # --- OPTIMIZATION 3: Resize frame for faster processing ---
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Resize by 50%
        
        # We use a grayscale version for the fast cascade detector
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        recognition_counter += 1

        for (x, y, w, h) in faces:
            # Scale bounding box back to original frame size
            x, y, w, h = x*2, y*2, w*2, h*2
            
            # --- OPTIMIZATION 2 (Continued): Recognize only periodically ---
            if recognition_counter % RECOGNITION_INTERVAL == 0:
                try:
                    # Run the heavy recognition task
                    dfs = DeepFace.find(
                        img_path=frame[y:y+h, x:x+w], # Pass only the face region
                        db_path=DB_PATH,
                        model_name=MODEL_NAME,
                        detector_backend='skip', # We already detected the face
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if dfs and not dfs[0].empty:
                        identity = dfs[0]['identity'].iloc[0]
                        name = os.path.basename(os.path.dirname(identity))
                        last_known_name = name # Update the name
                    else:
                        last_known_name = "Unknown"
                        
                except Exception as e:
                    last_known_name = "Unknown"
            
            # Draw the rectangle and the last known name on EVERY frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, last_known_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Display FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Optimized Real-time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_recognition()