import cv2
import pickle
import numpy as np
import os
import time

# --- Configuration ---
MODEL_PATH = 'data/face_recognition_model.pkl'
CONFIDENCE_THRESHOLD = 0.75  # Must be same as in app.py
IMAGE_SIZE = (100, 100)

def run_model_test():
    """
    Opens a webcam feed to perform real-time face recognition for testing purposes.
    """
    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please run 'train_model.py' to create the model.")
        return

    with open(MODEL_PATH, 'rb') as f:
        knn = pickle.load(f)
        
    # Initialize webcam and face detector
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return
        
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # For calculating FPS
    prev_frame_time = 0
    new_frame_time = 0

    print("\nStarting real-time model test...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.flip(frame, 1) # Mirror view
        
        # --- FPS Calculation ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # --- Face Detection and Recognition ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(face_img, IMAGE_SIZE).flatten().reshape(1, -1)
            
            # --- Prediction with Confidence ---
            probabilities = knn.predict_proba(resized_img)[0]
            max_prob = np.max(probabilities)
            
            if max_prob > CONFIDENCE_THRESHOLD:
                name = knn.predict(resized_img)[0]
                color = (0, 255, 0) # Green for confident match
            else:
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown
            
            # Display name and confidence on the frame
            display_text = f"{name} ({max_prob*100:.2f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Model Test - Press 'q' to exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    run_model_test()
