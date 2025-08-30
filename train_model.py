import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import imutils

# --- Configuration ---
RAW_DATA_DIR = 'data/raw'
MODEL_SAVE_PATH = 'data/face_recognition_model.pkl'
IMAGE_SIZE = (100, 100)

def augment_image(image):
    """
    Applies a set of augmentations to an image to create variations.
    """
    augmented_images = [image]
    # Add horizontal flip
    augmented_images.append(cv2.flip(image, 1))
    
    # Add slight rotations
    for angle in [-10, 10, -5, 5]:
        augmented_images.append(imutils.rotate(image, angle))
        
    return augmented_images

def train_model():
    """
    Loads raw images, applies augmentation, trains the KNN model,
    and saves it to a file.
    """
    faces = []
    labels = []

    print("Loading images and augmenting data...")

    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print(f"Error: The '{RAW_DATA_DIR}' directory is empty or does not exist.")
        print("Please run 'add_faces.py' to collect face samples first.")
        return

    # Loop through each person in the raw data directory
    for person_name in os.listdir(RAW_DATA_DIR):
        person_dir = os.path.join(RAW_DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"- Processing images for {person_name}...")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            
            # Augment the image
            augmented_images = augment_image(image)
            
            for aug_image in augmented_images:
                # Convert to grayscale and flatten
                gray_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_image, IMAGE_SIZE)
                faces.append(resized_face.flatten())
                labels.append(person_name)

    if not faces:
        print("No face data found to train the model. Aborting.")
        return

    faces = np.array(faces)
    labels = np.array(labels)

    print(f"\nTotal training samples after augmentation: {len(faces)}")

    # --- Train the K-Nearest Neighbors Classifier ---
    print("Training the face recognition model...")
    knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    
    # Split data for accuracy testing
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    
    knn_classifier.fit(X_train, y_train)

    # --- Evaluate Model Accuracy ---
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel training complete.")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # --- Save the Trained Model ---
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(knn_classifier, f)
    
    print(f"\nModel successfully saved to '{MODEL_SAVE_PATH}'")
    print("You can now run 'app.py' to start the attendance system.")

if __name__ == "__main__":
    train_model()
