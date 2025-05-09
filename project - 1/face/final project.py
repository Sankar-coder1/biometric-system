import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Load models
detector = MTCNN()
embedder = FaceNet()
with open("face_recognizer.pkl", "rb") as f:
    classifier, encoder = pickle.load(f)

cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_crop = img_rgb[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))
        
        embedding = embedder.embeddings([face_crop])[0]

        probabilities = classifier.predict_proba([embedding])[0]
        max_index = np.argmax(probabilities)
        name = encoder.classes_[max_index]
        confidence = probabilities[max_index]

        threshold = 0.5 + (np.mean(probabilities) * 0.2)
        if confidence > threshold:
            display_name = f"{name} ({confidence:.2f})"
        else:
            display_name = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
