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
MODEL_NAME = "SFace"  # Lightweight model for performance
DETECTOR_BACKEND = 'opencv'
ATTENDANCE_DIR = 'Attendance'
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# --- Page Setup ---
st.set_page_config(page_title="Smart Attendance System", layout="wide")
st.title("Smart Attendance System")

# --- Helper Functions ---
@st.cache_data
def get_todays_attendance():
    """Reads today's attendance file and returns a set of names."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    if not os.path.exists(attendance_file):
        return set()
    try:
        df = pd.read_csv(attendance_file)
        return set(df['NAME'].tolist())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return set()

def mark_attendance(name, todays_attendance_set):
    """Marks attendance in the CSV and updates the session's attendance set."""
    if name in ["Unknown", None, "..."]:
        return "Cannot mark attendance for 'Unknown'.", False

    if name in todays_attendance_set:
        return f"{name}'s attendance is already marked.", False

    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    
    file_exists_and_not_empty = os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0

    with open(attendance_file, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists_and_not_empty:
            writer.writerow(['NAME', 'TIME'])
        writer.writerow([name, timestamp])
    
    todays_attendance_set.add(name)
    return f"Welcome, {name}! Attendance marked.", True

# --- Main Application Logic ---
@st.cache_resource
def load_models():
    """Loads facial detection and recognition models into memory."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    DeepFace.build_model(MODEL_NAME)
    return detector

def run_attendance_system():
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()
    
    facedetect = load_models()

    if not any(os.scandir(DB_PATH)):
        status_placeholder.warning("Face database is empty. Please run add_faces.py to enroll users.")
        return

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        status_placeholder.error("Cannot access webcam. Please check camera permissions.")
        return

    status_placeholder.info("System activated. Please look at the camera.")
    
    if 'stop' not in st.session_state: st.session_state.stop = False
    if 'todays_attendance' not in st.session_state: st.session_state.todays_attendance = get_todays_attendance()
    
    # Use a local dictionary for tracking faces in the current run, not session_state
    face_tracking_data = {} 
    RECOGNITION_INTERVAL = 15

    while not st.session_state.stop:
        ret, frame = video.read()
        if not ret:
            status_placeholder.error("Failed to capture image from webcam.")
            break
        
        display_frame = frame.copy()
        
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR_GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            x, y, w, h = x*2, y*2, w*2, h*2
            face_roi = frame[y:y+h, x:x+w]
            
            face_id = i
            
            if face_id not in face_tracking_data:
                face_tracking_data[face_id] = {'name': '...', 'timer': 0}
            
            face_tracking_data[face_id]['timer'] += 1

            if face_tracking_data[face_id]['timer'] % RECOGNITION_INTERVAL == 0:
                try:
                    dfs = DeepFace.find(
                        img_path=face_roi,
                        db_path=DB_PATH,
                        model_name=MODEL_NAME,
                        detector_backend='skip',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if dfs and not dfs[0].empty:
                        identity = dfs[0]['identity'].iloc[0]
                        name = os.path.basename(os.path.dirname(identity))
                        face_tracking_data[face_id]['name'] = name
                    else:
                        face_tracking_data[face_id]['name'] = "Unknown"
                except:
                    face_tracking_data[face_id]['name'] = "Searching..."

            name_to_display = face_tracking_data[face_id]['name']
            
            status_message, marked = mark_attendance(name_to_display, st.session_state.todays_attendance)
            if marked:
                status_placeholder.success(status_message)
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, name_to_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        webcam_placeholder.image(display_frame, channels="BGR", use_container_width=True)
            
        time.sleep(0.01) 

    video.release()
    webcam_placeholder.empty()
    status_placeholder.info("System stopped.")
    st.session_state.stop = False

def display_attendance():
    st.header("Today's Attendance")
    if st.button("Refresh"):
        st.session_state.todays_attendance = get_todays_attendance()
        st.rerun()

    attendance_list = sorted(list(st.session_state.get('todays_attendance', get_todays_attendance())))
    if attendance_list:
        df = pd.DataFrame(attendance_list, columns=["NAME"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance has been recorded yet today.")
        
    st.header("Historical Attendance Records")
    files = sorted([f for f in os.listdir(ATTENDANCE_DIR) if f.endswith('.csv')], reverse=True)
    for file in files:
        date_str = file.replace("Attendance_", "").replace(".csv", "")
        with st.expander(f"Records for {date_str}"):
            try:
                df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
                st.dataframe(df, use_container_width=True, hide_index=True)
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
    
    if stop_button:
        st.session_state.stop = True
        st.rerun()

with col2:
    display_attendance()