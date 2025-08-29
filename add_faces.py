import cv2
import pickle
import numpy as np
import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

# Create a custom background instead of loading back.jpg
background = np.zeros((700, 1000, 3), dtype=np.uint8)
# Set a nice blue background color
background[:, :] = (50, 50, 150)  # Dark blue background

# Add title and instructions
cv2.putText(background, "PLEASE BE PATIENT", (200, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(background, "WE ARE UPLOADING YOUR FACE TO OUR SYSTEM", (350, 130), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Add some decorative elements
cv2.rectangle(background, (50, 300), (950, 650), (255, 255, 255), 2)
cv2.putText(background, "Camera Feed Will Appear Here", (300, 330), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Create a named window and position it to ensure it's visible
cv2.namedWindow("Face Recognition System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition System", 1000, 700)

# Get screen dimensions and center the window
screen_width = 1920  # Default screen width, will be updated if possible
screen_height = 1080  # Default screen height, will be updated if possible

try:
    import screeninfo
    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height
except:
    pass  # If screeninfo is not available, use default values

# Calculate center position
window_x = (screen_width - 1000) // 2
window_y = (screen_height - 700) // 2
cv2.moveWindow("Face Recognition System", window_x, window_y)

# Show the background immediately
cv2.imshow("Face Recognition System", background)
cv2.waitKey(100)  # Small delay to ensure window is displayed

# Force window to be on top (Windows-specific solution)
try:
    import win32gui
    import win32con
    hwnd = win32gui.FindWindow(None, "Face Recognition System")
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, window_x, window_y, 1000, 700, 0)
except:
    pass  # If win32gui is not available, continue without setting window to top

# Adjust face detection parameters to be more sensitive
# This will help detect faces even when not too close to the camera
face_params = {
    'scaleFactor': 1.1,    # Lower value = more sensitive but slower
    'minNeighbors': 3,     # Lower value = more detections but more false positives
    'minSize': (30, 30)    # Smaller minimum size to detect faces from farther away
}

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use adjusted parameters for face detection
    faces = facedetect.detectMultiScale(gray, 
                                       scaleFactor=face_params['scaleFactor'],
                                       minNeighbors=face_params['minNeighbors'],
                                       minSize=face_params['minSize'])
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        if len(faces_data) < 20 and i % 5 == 0:  # Capture every 5th frame until we have 20 images
            faces_data.append(resized_img)
            
        i += 1
        
        # Draw face bounding box and info on the frame
        cv2.putText(frame, f"Collected: {len(faces_data)}/20", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        
        # Add distance indicator
        face_size_percentage = (w * h) / (frame.shape[0] * frame.shape[1]) * 100
        distance_text = f"Face size: {face_size_percentage:.1f}%"
        cv2.putText(frame, distance_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Create a composite image with background and webcam feed
    composite = background.copy()
    
    # Place webcam feed on the right side (centered in the white rectangle)
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        # Resize frame to fit in the designated area
        frame_resized = cv2.resize(frame, (400, 300))
        composite[350:650, 300:700] = frame_resized
    
    # Update collection progress on the composite image
    progress_text = f"Progress: {len(faces_data)}/20 images collected"
    cv2.putText(composite, progress_text, (300, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Face Recognition System", composite)
    
    # Keep window on top
    try:
        import win32gui
        import win32con
        hwnd = win32gui.FindWindow(None, "Face Recognition System")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, window_x, window_y, 1000, 700, 0)
    except:
        pass
    
    k = cv2.waitKey(1)
    
    if k == ord('q') or len(faces_data) >= 20:
        break

video.release()
cv2.destroyAllWindows()

# Process and save the faces data
if len(faces_data) > 0:
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(len(faces_data), -1)

    # Save names
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save faces data
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

    print(f"Successfully added {len(faces_data)} faces for {name}")
else:
    print("No faces were captured. Please try again.")