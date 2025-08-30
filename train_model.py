import cv2
import os
import numpy as np
import pickle

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
MODEL_SAVE_PATH = 'data/face_recognizer.yml'
LABELS_SAVE_PATH = 'data/labels.pkl'

def train_lbph_model():
    print("Preparing data for training...")
    faces = []
    labels = []
    label_map = {}
    current_label_id = 0

    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print(f"Error: The '{RAW_DATA_DIR}' directory is empty. Please run 'add_faces.py' first.")
        return

    for person_name in os.listdir(RAW_DATA_DIR):
        person_dir = os.path.join(RAW_DATA_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        if person_name not in label_map:
            label_map[person_name] = current_label_id
            current_label_id += 1
        
        label_id = label_map[person_name]
        
        print(f"- Loading images for {person_name} (ID: {label_id})")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(image)
                labels.append(label_id)

    if not faces:
        print("No face data found. Aborting training.")
        return

    # --- Train the LBPH Face Recognizer ---
    print("\nTraining the LBPH face recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # --- Save the Model and Labels ---
    recognizer.save(MODEL_SAVE_PATH)
    
    # Invert the label map for easy lookup later
    id_to_label = {v: k for k, v in label_map.items()}
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(id_to_label, f)
        
    print(f"\nTraining complete.")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Labels saved to: {LABELS_SAVE_PATH}")

if __name__ == "__main__":
    train_lbph_model()