import cv2
import pickle
import numpy as np
import os
import time

# --- Configuration ---
MODEL_PATH = 'data/face_recognizer.yml'
LABELS_PATH = 'data/labels.pkl'
CONFIDENCE_THRESHOLD = 60 # Lower is better. Max ~150. Start with 60.

def run_model_test():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Please run 'train_model.py'.")
        return

    # Load the recognizer and labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        id_to_label = pickle.load(f)

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return
        
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    prev_frame_time = 0

    print("\nStarting real-time model test... Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # FPS Calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img_gray = gray[y:y+h, x:x+w]
            
            # --- Prediction and Confidence ---
            label_id, confidence = recognizer.predict(face_img_gray)
            
            # Confidence in LBPH is a distance. Lower is better.
            if confidence < CONFIDENCE_THRESHOLD:
                name = id_to_label.get(label_id, "Unknown")
                color = (0, 255, 0) # Green for confident match
            else:
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown
            
            display_text = f"{name} (Conf: {confidence:.2f})"
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