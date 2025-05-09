from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

# Load embeddings
with open("face_embeddings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Train SVM classifier
classifier = SVC(kernel="linear", probability=True)
classifier.fit(embeddings, encoded_labels)

# Save model & encoder
with open("face_recognizer.pkl", "wb") as f:
    pickle.dump((classifier, encoder), f)

print("✅ Face recognition model trained and saved!")
