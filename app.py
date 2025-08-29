import streamlit as st
import pandas as pd
import time 
from datetime import datetime
import os
import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tempfile
import shutil
import csv

st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
st.title("Face Recognition Attendance System")

# Create directories if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Check if training data exists
if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
    st.error("No training data found. Please run add_faces.py first.")
    st.stop()

# Load trained model
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except Exception as e:
    st.error(f"Error loading training data: {str(e)}")
    st.stop()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize face detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    # Fallback to the path in your code
    cascade_path = 'data/haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        st.error("Haar cascade file not found. Please ensure it's in the data directory.")
        st.stop()

facedetect = cv2.CascadeClassifier(cascade_path)

# Function to check if attendance already recorded for a person today
def is_attendance_recorded_today(name):
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    attendance_file = f"Attendance/Attendance_{date}.csv"
    
    if not os.path.exists(attendance_file):
        return False
        
    try:
        df = pd.read_csv(attendance_file)
        return name in df['NAME'].values
    except:
        return False

# Function to create background image
def create_background():
    try:
        background = cv2.imread('background.png')
        background = cv2.resize(background, (1000, 700))
    except:
        # Create a more visually appealing background
        background = np.zeros((700, 1000, 3), dtype=np.uint8)
        # Add gradient background
        for i in range(700):
            color = int(i / 700 * 100)
            background[i, :] = (color, color, 100 + color//2)
        
        # Add text with better styling
        cv2.putText(background, "FACE RECOGNITION ATTENDANCE", (120, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(background, "Automatically detecting faces...", (120, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(background, "Attendance will be taken automatically", (120, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add a rectangle for the webcam feed
        cv2.rectangle(background, (450, 150), (950, 550), (255, 255, 255), 2)
    return background

# Function to take attendance automatically
def take_attendance_auto():
    # Create a placeholder for the webcam feed
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Create background
    background = create_background()
    
    # Start webcam
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        status_placeholder.error("Cannot access webcam. Please check your camera settings.")
        return False
    
    current_prediction = None
    attendance_taken = False
    
    # Instructions
    status_placeholder.info("Looking for a face. Attendance will be taken automatically when a face is detected.")
    
    # Create a temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            ret, frame = video.read()
            if not ret:
                status_placeholder.error("Failed to capture image from webcam.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Only process the first face detected
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = knn.predict(resized_img)
                current_prediction = output[0]
                
                # Check if attendance already recorded today
                already_recorded = is_attendance_recorded_today(current_prediction)
                status_text = "Already recorded today" if already_recorded else "Ready to record"
                status_color = (0, 0, 255) if already_recorded else (0, 255, 0)
                
                # Draw rectangles and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(frame, status_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Automatically take attendance (record every attempt)
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
                
                COL_NAMES = ['NAME', 'TIME']
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
                
                if already_recorded:
                    status_placeholder.warning(f"Attendance recorded again for {current_prediction} at {timestamp}")
                else:
                    status_placeholder.success(f"Attendance recorded for {current_prediction} at {timestamp}")
                
                attendance_taken = True
                time.sleep(2)
                return True
            
            # Create composite image with background and webcam feed
            composite = background.copy()
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                frame_resized = cv2.resize(frame, (500, 400))
                composite[150:550, 450:950] = frame_resized
            
            # Convert to RGB for Streamlit display
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            
            # Display the composite image
            webcam_placeholder.image(composite_rgb, channels="RGB", use_container_width=True)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
        
        if not attendance_taken:
            status_placeholder.warning("No face detected within time limit. Please try again.")
    
    finally:
        video.release()
        cv2.destroyAllWindows()
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return attendance_taken

# Main interface
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Display today's attendance if available - Show only unique attendees
attendance_file = f"Attendance/Attendance_{date}.csv"

if os.path.exists(attendance_file):
    try:
        df = pd.read_csv(attendance_file)
        st.subheader(f"Today's Attendance ({date})")
        
        # Get unique attendees with their first recorded time
        unique_attendees = df.groupby('NAME').first().reset_index()
        
        # Count unique attendees
        unique_count = len(unique_attendees)
        
        st.metric("Unique Attendees Today", unique_count)
        
        # Display unique attendees in table format
        st.dataframe(unique_attendees)
        
    except Exception as e:
        st.error(f"Error reading attendance file: {str(e)}")
else:
    st.warning(f"No attendance recorded for {date}")

# Add a button to start automatic attendance
if st.button("Start Attendance Session", key="start_attendance_btn_unique"):
    attendance_taken = take_attendance_auto()
    
    # Refresh the page if attendance was taken
    if attendance_taken:
        st.rerun()

# Display all attendance files with summary (unchanged)
st.subheader("All Attendance Records")

if os.path.exists('Attendance'):
    files = os.listdir("Attendance")
    csv_files = [f for f in files if f.startswith("Attendance_") and f.endswith(".csv")]
    csv_files.sort(reverse=True)  # Show most recent first
    
    if csv_files:
        for file in csv_files:
            try:
                date_str = file.replace("Attendance_", "").replace(".csv", "")
                df = pd.read_csv(f"Attendance/{file}")
                
                # Calculate statistics
                unique_attendees = df['NAME'].nunique()
                total_records = len(df)
                
                # Display with expander
                with st.expander(f"{date_str} - {unique_attendees} unique attendees, {total_records} total records"):
                    st.write(f"**Date:** {date_str}")
                    st.write(f"**Unique Attendees:** {unique_attendees}")
                    st.write(f"**Total Records:** {total_records}")
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"Could not read {file}: {str(e)}")
    else:
        st.info("No attendance records found")
else:
    st.info("No attendance directory found")