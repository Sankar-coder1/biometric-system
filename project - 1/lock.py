import os
import cv2
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
import whisper
import tkinter as tk
from tkinter import messagebox
from collections import Counter

# Load Pre-trained Emotion Model
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

time.sleep(5)
emotion_model.load_weights('face_emotion/emotion_model.weights.h5')
print("Loaded emotion model")

# Capture Face and Train Face Recognition Model

def capture_faces(name, num_images=5):
    os.makedirs(f'dataset/{name}', exist_ok=True)
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    embedder = FaceNet()
    face_embeddings, emotions = [], []
    count = 0
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
            embedding = embedder.embeddings([face])[0]
            face_embeddings.append(embedding)
            
            # Emotion Detection
            gray_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = img_to_array(gray_face) / 255.0
            gray_face = np.expand_dims(gray_face, axis=0)
            emotion_prediction = emotion_model.predict(gray_face)
            emotion_label = emotion_dict[np.argmax(emotion_prediction)]
            emotions.append(np.argmax(emotion_prediction))
            
            img_path = f'dataset/{name}/img{count}_{emotion_label}.jpg'
            cv2.imwrite(img_path, face)
            
            cv2.imshow("Capturing Faces", frame)
            count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    avg_emotion = max(Counter(emotions), key=Counter(emotions).get)
    print(f"✅ Average detected emotion: {emotion_dict[avg_emotion]}")
    return np.array(face_embeddings), avg_emotion


def train_face_recognition(name, embeddings):
    labels = [name] * len(embeddings)
    unknown_embeddings, unknown_labels = [], []
    
    if os.path.exists("unknown_embeddings.npy"):
        unknown_embeddings = np.load("unknown_embeddings.npy")
        unknown_labels = ["Unknown"] * len(unknown_embeddings)
    else:
        print("❌ No unknown face dataset found.")
        return
    
    all_embeddings = np.vstack((embeddings, unknown_embeddings))
    all_labels = np.array(labels + unknown_labels)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(all_labels)
    
    if len(np.unique(encoded_labels)) < 2:
        print("❌ Not enough classes to train. Add more user data.")
        return
    
    model = SVC(kernel='linear', probability=True)
    model.fit(all_embeddings, encoded_labels)
    with open("face_recognition.pkl", "wb") as f:
        pickle.dump((model, encoder), f)
    print("✅ Face recognition model trained successfully!")
    time.sleep(5)

# Define a CNN model for voice recognition
def create_voice_recognition_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the voice recognition model
def train_voice_recognition(name, features, labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(encoded_labels))
    one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)
    
    # Reshape features for CNN input
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
    
    # Create and train the model
    input_shape = (X_train.shape[1], 1)  # Input shape for CNN
    model = create_voice_recognition_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model and encoder
    model.save(f'voice_recognition_{name}.h5')
    with open(f'voice_recognition_encoder_{name}.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("✅ Voice recognition model trained successfully!")

# Predict speaker using the trained model
def predict_speaker(feature, model, encoder):
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    feature = np.expand_dims(feature, axis=-1)  # Add channel dimension
    prediction = model.predict(feature)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Capture voice samples
def capture_voice(name, num_samples=5, duration=3, fs=44100):
    os.makedirs(f'voice_dataset/{name}', exist_ok=True)
    features, labels = [], []
    
    for i in range(num_samples):
        print(f'Recording sample {i+1}...')
        time.sleep(2)  # Added time for user readiness
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        filename = f'voice_dataset/{name}/sample{i+1}.wav'
        sf.write(filename, audio, fs)
        feature = extract_features(filename)
        if feature is not None:
            features.append(feature)
            labels.append(name)
    
    return np.array(features), np.array(labels)

# Extract MFCC features from audio
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Record audio for password
def record_audio(duration=5, filename="password.wav"):
    print("Get ready to record your password in 3 seconds...")
    time.sleep(3)  # Giving time to prepare
    print("Recording started. Speak now!")
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, 44100)
    print("Recording saved as:", filename)
    return filename

# Set password using speech-to-text
def set_password():
    filename = record_audio(5, "password.wav")  # Increased duration to 5 seconds
    password_text = speech_to_text(filename)
    with open("password.txt", "w") as f:
        f.write(password_text)
    print("✅ Password saved successfully!")
    return password_text

# Speech-to-text using Whisper
def speech_to_text(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result["text"].strip().lower()

# GUI to display user information
def show_user_info(name, avg_emotion, password):
    root = tk.Tk()
    root.title("User Information")
    
    tk.Label(root, text=f"Name: {name}", font=("Arial", 14)).pack(pady=10)
    tk.Label(root, text=f"Average Emotion: {emotion_dict[avg_emotion]}", font=("Arial", 14)).pack(pady=10)
    tk.Label(root, text=f"Password: {password}", font=("Arial", 14)).pack(pady=10)
    
    root.mainloop()

# Main function
def main():
    name = input("Enter your name: ")
    
    # Face recognition
    face_embeddings, avg_emotion = capture_faces(name)
    train_face_recognition(name, face_embeddings)
    with open(f'emotion_{name}.txt', 'w') as f:
        f.write(str(avg_emotion))
    
    # Voice recognition
    voice_features, voice_labels = capture_voice(name)
    train_voice_recognition(name, voice_features, voice_labels)
    
    # Set password
    password_text = set_password()
    
    # Show user information in GUI
    show_user_info(name, avg_emotion, password_text)
    
    print("🔒 Locking System Setup Complete!")

if __name__ == '__main__':
    main()