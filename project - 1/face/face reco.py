import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# Initialize models
detector = MTCNN()
embedder = FaceNet()

dataset_path = "C:/Projects/project - 1/dataset"
embeddings = []
labels = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(img_rgb)
        if len(faces) == 0:
            print(f"❌ No face detected in {img_name} of {person}")
            continue
        
        # Extract face bounding box
        x, y, w, h = faces[0]['box']
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # Resize for FaceNet
        
        # Extract embedding
        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)
        labels.append(person)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save embeddings & labels
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("✅ Embeddings extracted and saved!")
