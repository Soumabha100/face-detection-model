from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

try:
    from win32com.client import Dispatch
    def speak(str1):
        speak = Dispatch(("SAPI.SpVoice"))
        speak.Speak(str1)
except:
    def speak(str1):
        print(str1)

# Create directories if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Check if training data exists
if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
    print("No training data found. Please run add_faces.py first.")
    exit()

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Number of labels --> ', len(LABELS))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
try:
    background = cv2.imread('background.png')
    background = cv2.resize(background, (1000, 700))
except:
    background = np.zeros((700, 1000, 3), dtype=np.uint8)
    cv2.putText(background, "FACE RECOGNITION ATTENDANCE", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(background, "Press 'o' to take attendance", (100, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(background, "Press 'q' to quit", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

COL_NAMES = ['NAME', 'TIME']
current_face = None
current_prediction = None

def is_attendance_today(name):
    """Check if attendance for this person has already been recorded today"""
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    filename = "Attendance/Attendance_" + date + ".csv"
    
    if not os.path.exists(filename):
        return False
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0] == name:
                return True
    return False

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Only process the first face detected
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        current_prediction = output[0]
        current_face = (x, y, w, h)
        
        # Draw rectangles and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    # Create a composite image with background and webcam feed
    composite = background.copy()
    # Place webcam feed on the right side
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        frame_resized = cv2.resize(frame, (500, 400))
        composite[150:550, 450:950] = frame_resized
    
    # Display the composite image
    cv2.imshow("Face Recognition Attendance", composite)
    
    k = cv2.waitKey(1)
    if k == ord('o'):
        if current_prediction is not None:
            # Check if attendance already recorded today
            if is_attendance_today(current_prediction):
                print(f"Attendance already recorded for {current_prediction} today")
                # Show already recorded message
                already_bg = np.zeros((200, 500, 3), dtype=np.uint8)
                cv2.putText(already_bg, f"Attendance already recorded", (50, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(already_bg, f"for {current_prediction}", (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Already Recorded", already_bg)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyWindow("Already Recorded")
                
                # Exit after showing message
                break
            else:
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
                exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
                
                speak(f"Attendance Taken for {current_prediction}")
                attendance = [str(current_prediction), str(timestamp)]
                
                if exist:
                    with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                else:
                    with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                
                print(f"Attendance recorded for {current_prediction} at {timestamp}")
                
                # Show confirmation message on screen
                confirmation_bg = np.zeros((200, 500, 3), dtype=np.uint8)
                cv2.putText(confirmation_bg, f"Attendance recorded for", (50, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(confirmation_bg, f"{current_prediction}", (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Confirmation", confirmation_bg)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyWindow("Confirmation")
                
                # Exit after taking attendance
                break
        else:
            print("No face detected for attendance")
            # Show no face detected message
            no_face_bg = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(no_face_bg, "No face detected", (50, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(no_face_bg, "Please try again", (50, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("No Face", no_face_bg)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow("No Face")
            
            # Exit after showing message
            break
   

video.release()
cv2.destroyAllWindows()