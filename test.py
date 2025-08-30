import cv2
import os
from deepface import DeepFace
import time

# --- Configuration ---
DB_PATH = "face_db"
MODEL_NAME = "VGG-Face" # Other options: "Facenet", "ArcFace", "OpenFace"
DETECTOR_BACKEND = 'opencv' # Other options: 'ssd', 'dlib', 'mtcnn'

def run_realtime_recognition():
    """
    Performs real-time face recognition using the DeepFace library.
    """
    print("Starting real-time recognition...")
    print(f"Using model '{MODEL_NAME}' and database path '{DB_PATH}'")
    print("Press 'q' to quit.")
    
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        print(f"Error: The database path '{DB_PATH}' is empty.")
        print("Please run add_faces.py to add images first.")
        return

    video = cv2.VideoCapture(0)
    prev_frame_time = 0

    while True:
        ret, frame = video.read()
        if not ret: break

        try:
            # The core of DeepFace: find faces in the frame and recognize them
            # against the images in the DB_PATH folder.
            dfs = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False, # Continue even if no face is found
                silent=True # Suppress console output
            )

            if dfs and not dfs[0].empty:
                for _, row in dfs[0].iterrows():
                    x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
                    identity = row['identity']
                    
                    # Extract the name from the identity path
                    name = os.path.basename(os.path.dirname(identity))
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            # Handle cases where no face is detected
            pass

        # --- Display FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("DeepFace Real-time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_recognition()