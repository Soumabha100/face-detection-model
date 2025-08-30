import cv2
import pickle
import numpy as np
import os
import time

# --- Configuration ---
# The target number of face samples to collect.
# Increasing this number will improve model accuracy but take longer.
NUMBER_OF_FACES_TO_COLLECT = 100

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

# --- User Interface and Instructions ---

def create_ui_frame():
    """Creates the base UI frame with instructions."""
    frame = np.zeros((700, 1000, 3), dtype=np.uint8)
    frame[:, :] = (50, 50, 150)  # Dark blue background

    cv2.putText(frame, "Collecting Face Samples", (280, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.rectangle(frame, (250, 150), (750, 550), (255, 255, 255), 2)
    return frame

# --- Main Loop ---

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with optimized parameters
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Create the main UI frame for display
    ui_frame = create_ui_frame()

    # Get the current number of collected faces
    collected_count = len(faces_data)

    # --- Instructions based on progress ---
    if collected_count < 25:
        instruction = "Look straight at the camera."
    elif collected_count < 50:
        instruction = "Slowly turn your head to the LEFT."
    elif collected_count < 75:
        instruction = "Slowly turn your head to the RIGHT."
    else:
        instruction = "Slowly tilt your head UP and DOWN."
    
    cv2.putText(ui_frame, instruction, (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(faces) > 0:
        # Process the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Capture an image every 2nd frame to get slight variations
        if i % 2 == 0:
            faces_data.append(resized_img)
        i += 1
        
        # Draw face bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
    
    # --- Display Progress ---
    progress_text = f"Progress: {collected_count}/{NUMBER_OF_FACES_TO_COLLECT}"
    cv2.putText(ui_frame, progress_text, (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Place the live camera feed into the designated rectangle on the UI frame
    frame_resized = cv2.resize(frame, (500, 400))
    ui_frame[150:550, 250:750] = frame_resized
    
    cv2.imshow("Adding New Face", ui_frame)
    
    # Exit condition: Press 'q' or when enough faces are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= NUMBER_OF_FACES_TO_COLLECT:
        break

video.release()
cv2.destroyAllWindows()

# --- Save Data ---
if len(faces_data) >= NUMBER_OF_FACES_TO_COLLECT:
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(NUMBER_OF_FACES_TO_COLLECT, -1)

    # Load existing data or create new files
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * NUMBER_OF_FACES_TO_COLLECT
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + ([name] * NUMBER_OF_FACES_TO_COLLECT)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
    
    print(f"Successfully collected and saved {len(faces_data)} images for {name}.")
else:
    print("Face collection was interrupted. No data was saved.")
