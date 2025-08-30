import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import cv2
import pickle
import numpy as np
import csv

# --- Configuration ---
MODEL_PATH = 'data/face_recognition_model.pkl'
ATTENDANCE_DIR = 'Attendance'
CONFIDENCE_THRESHOLD = 0.75  # 75% confidence required for a match

# --- Page Setup ---
st.set_page_config(page_title="Advanced Face Recognition Attendance", layout="wide")
st.title("Advanced Face Recognition Attendance System")

# --- Load Model ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please run 'train_model.py' first.")
        st.stop()
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

knn = load_model()
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# --- Helper Functions ---
def is_attendance_recorded_today(name, date):
    """Checks if a person's attendance has already been logged for the given date."""
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")
    if not os.path.exists(attendance_file):
        return False
    try:
        df = pd.read_csv(attendance_file)
        return name in df['NAME'].values
    except pd.errors.EmptyDataError:
        return False

def mark_attendance(name, date, timestamp):
    """Records a new attendance entry."""
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")
    file_exists = os.path.exists(attendance_file)
    
    with open(attendance_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['NAME', 'TIME'])
        writer.writerow([name, timestamp])

# --- Main Application Logic ---
def run_attendance_system():
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        status_placeholder.error("Cannot access webcam. Please check settings.")
        return

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                status_placeholder.error("Failed to capture image from webcam.")
                break
            
            frame = cv2.flip(frame, 1) # Mirror view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                resized_img = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
                
                # --- Prediction with Confidence ---
                distances, indices = knn.kneighbors(resized_img, n_neighbors=5)
                probabilities = knn.predict_proba(resized_img)[0]
                max_prob = np.max(probabilities)
                
                predicted_name = knn.predict(resized_img)[0]

                if max_prob > CONFIDENCE_THRESHOLD:
                    name = predicted_name
                    color = (0, 255, 0) # Green for confident match
                    
                    ts = time.time()
                    date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    
                    if not is_attendance_recorded_today(name, date):
                        mark_attendance(name, date, timestamp)
                        status_placeholder.success(f"Welcome {name}! Attendance marked at {timestamp}.")
                        time.sleep(3) # Pause to show the message
                else:
                    name = "Unknown"
                    color = (0, 0, 255) # Red for unknown
                
                # Display name and confidence
                display_text = f"{name} ({max_prob*100:.2f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    finally:
        video.release()

def display_attendance():
    """Shows the attendance records in the Streamlit app."""
    st.header("Today's Attendance")
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    
    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
            st.dataframe(df)
        except pd.errors.EmptyDataError:
            st.info("No attendance has been recorded yet today.")
    else:
        st.info("No attendance has been recorded yet today.")
        
    st.header("Historical Attendance Records")
    files = sorted(os.listdir(ATTENDANCE_DIR), reverse=True)
    for file in files:
        date_str = file.replace("Attendance_", "").replace(".csv", "")
        with st.expander(f"Records for {date_str}"):
            try:
                df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
                st.dataframe(df)
            except pd.errors.EmptyDataError:
                st.write("No records for this day.")

# --- Streamlit UI ---
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Live Camera Feed")
    if st.button("Start Attendance System"):
        run_attendance_system()

with col2:
    display_attendance()
