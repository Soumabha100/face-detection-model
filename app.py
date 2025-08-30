import streamlit as st
import pandas as pd
from datetime import datetime
import os
import cv2
from deepface import DeepFace
import time
import csv

# --- Configuration ---
DB_PATH = "face_db"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = 'opencv'
ATTENDANCE_DIR = 'Attendance'
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# --- Page Setup ---
st.set_page_config(page_title="DeepFace Attendance System", layout="wide")
st.title("Smart Attendance System")

# --- Helper Functions ---
def mark_attendance(name):
    """Marks attendance for a recognized person, ensuring one entry per day."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    
    # Check if the file exists and has content to avoid errors
    file_exists = os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0

    if file_exists:
        df = pd.read_csv(attendance_file)
        if name in df['NAME'].values:
            return f"{name}'s attendance already marked today."

    with open(attendance_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['NAME', 'TIME'])
        writer.writerow([name, timestamp])
    
    return f"Welcome, {name}! Attendance marked."

# --- Main Application Logic ---
def run_attendance_system():
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        status_placeholder.error("Cannot access webcam. Please check settings.")
        return

    # Check if the database is empty
    if not os.listdir(DB_PATH):
        status_placeholder.warning("The face database is empty. Please run add_faces.py to add people.")
        video.release()
        return

    status_placeholder.info("System started. Looking for faces...")
    
    # We use a session state to stop the loop
    if 'stop' not in st.session_state:
        st.session_state.stop = False

    while not st.session_state.stop:
        ret, frame = video.read()
        if not ret:
            status_placeholder.error("Failed to capture image from webcam.")
            break
        
        try:
            # DeepFace recognition
            dfs = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )

            # Draw rectangles on the original frame
            display_frame = frame.copy()

            if dfs and not dfs[0].empty:
                for _, row in dfs[0].iterrows():
                    x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
                    identity = row['identity']
                    name = os.path.basename(os.path.dirname(identity))
                    
                    # Mark attendance and get status message
                    status_message = mark_attendance(name)
                    status_placeholder.success(status_message)
                    
                    # Draw visuals
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            webcam_placeholder.image(display_frame, channels="BGR", use_container_width=True)

        except Exception as e:
            pass # Suppress errors when no face is detected
            
        time.sleep(0.1) # Small delay to keep Streamlit responsive

    video.release()
    webcam_placeholder.empty()
    status_placeholder.info("System stopped.")
    st.session_state.stop = False # Reset for next run

def display_attendance():
    """Shows the attendance records in the Streamlit app."""
    st.header("Today's Attendance")
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    
    if os.path.exists(attendance_file):
        try:
            df = pd.read_csv(attendance_file)
            st.dataframe(df, use_container_width=True)
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
                st.dataframe(df, use_container_width=True)
            except pd.errors.EmptyDataError:
                st.write("No records for this day.")

# --- Streamlit UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Camera Feed")
    
    start_button = st.button("Start Attendance System")
    stop_button = st.button("Stop System")

    if start_button:
        st.session_state.stop = False
        run_attendance_system()
        st.rerun()

    if stop_button:
        st.session_state.stop = True
        st.rerun()

with col2:
    display_attendance()