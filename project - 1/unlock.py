import os
import cv2
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
import warnings
import whisper
import difflib
import noisereduce as nr
import tkinter as tk
from tkinter import messagebox, simpledialog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, model_from_json
import sys  # For restarting the script
import subprocess  # For running lock.py

warnings.filterwarnings("ignore", category=UserWarning)  # Remove FP32 warning

# Constants
FACE_MODEL_PATH = "face_recognition.pkl"
EMOTION_MODEL_JSON = "face_emotion\\emotion_model.json"
EMOTION_MODEL_WEIGHTS = "face_emotion\\emotion_model.weights.h5"
PASSWORD_FILE = "password.txt"
RECOVERY_ANSWER = "skyblue"  # Replace with a secure hashed answer in production

# Load trained models
print("⏳ Loading models...")
try:
    with open(FACE_MODEL_PATH, "rb") as f:
        face_model, face_encoder = pickle.load(f)

    whisper_model = whisper.load_model("small")

    with open(EMOTION_MODEL_JSON, "r") as json_file:
        emotion_model = model_from_json(json_file.read())
    emotion_model.load_weights(EMOTION_MODEL_WEIGHTS)

    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# Helper Functions
def load_password():
    if os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, "r") as f:
            return f.read().strip().lower()
    return None

def load_emotion(name):
    """Load the stored emotion for a given user."""
    emotion_file = f"emotion_{name}.txt"
    if os.path.exists(emotion_file):
        with open(emotion_file, "r") as f:
            return int(f.read().strip())  # Read the emotion as a number
    return None

def save_emotion(name, emotion):
    """Save the emotion for a given user."""
    emotion_file = f"emotion_{name}.txt"
    with open(emotion_file, "w") as f:
        f.write(str(emotion))  # Save the emotion as a number

def extract_features(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        y = nr.reduce_noise(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def record_audio(duration, filename, purpose="voice"):
    fs = 44100
    print(f"🎤 Recording audio for {purpose}...")
    print("Get ready to record in 3 seconds...")
    time.sleep(3)
    print("Recording started. Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, fs)
    print("✅ Recording complete.")
    return filename

def speech_to_text(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip().lower()
    except Exception as e:
        print(f"❌ Error in speech-to-text: {e}")
        return None

# Emotion Dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Main Functions
def recognize_face_and_emotion():
    print("📸 Capturing face for recognition...")
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    embedder = FaceNet()
    attempts = 0
    
    # Create a named window for the camera feed
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    
    while attempts < 3:
        ret, frame = cap.read()
        if not ret:
            attempts += 1
            time.sleep(0.5)
            continue
        
        # Display the camera feed
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)  # Refresh the window
        
        # Add a delay before capturing the image
        if attempts == 0:
            print("Get ready! Capturing image in 3 seconds...")
            time.sleep(3)
        
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
            embedding = embedder.embeddings([face])[0]
            prediction = face_model.predict([embedding])[0]
            name = face_encoder.inverse_transform([prediction])[0]
            
            face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (48, 48)) / 255.0
            face_gray = np.expand_dims(face_gray, axis=0)
            emotion_prediction = np.argmax(emotion_model.predict(face_gray)[0])
            detected_emotion = emotion_prediction  # Store the emotion as a number
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display the emotion label on the face
            emotion_label = emotion_dict[detected_emotion]
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Show the frame with the emotion label
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)  # Refresh the window
            
            # Release the camera and close the window
            cap.release()
            cv2.destroyAllWindows()
            return name, detected_emotion
        
        attempts += 1
        time.sleep(0.5)
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    return None, None

def recognize_voice(name):
    try:
        # Record audio for speech-to-text
        filename = record_audio(5, "voice_temp.wav", "speech recognition")
        
        # Extract text from the recorded audio
        extracted_text = speech_to_text(filename)
        
        if extracted_text:
            print(f"Extracted Text: {extracted_text}")
            
            # Load the stored password
            stored_password = load_password()
            if not stored_password:
                print("❌ No password found. Please set a password first.")
                return False
            
            # Compare the extracted text with the stored password
            similarity = difflib.SequenceMatcher(None, extracted_text.lower(), stored_password.lower()).ratio()
            
            # If the similarity is above a threshold, consider it a match
            similarity_threshold = 0.7  # Adjust this threshold as needed
            if similarity > similarity_threshold:
                print(f"✅ Voice (speech) matched! Similarity: {similarity:.2f}")
                return True
            else:
                print(f"❌ Voice (speech) not recognized! Similarity: {similarity:.2f}")
                return False
        else:
            print("❌ No text extracted from audio.")
            return False
    except Exception as e:
        print(f"❌ Error in voice (speech) recognition: {e}")
        return False

def reset_password():
    """Ask the recovery question and run lock.py if the answer is correct."""
    recovery_question = "What is your favorite color?"
    user_answer = simpledialog.askstring("Recovery Question", recovery_question)
    
    if user_answer and user_answer.strip().lower() == RECOVERY_ANSWER:
        try:
            # Close the current GUI
            root.destroy()
            # Run lock.py
            subprocess.run([sys.executable, "lock.py"])
        except Exception as e:
            print(f"❌ Error running lock.py: {e}")
    else:
        messagebox.showerror("Error", "❌ Incorrect answer. Please try again.")

def unlock_system():
    name, detected_emotion = recognize_face_and_emotion()
    if name and name != "Unknown":
        print("✅ Face matched!")
        print(f"Detected Emotion: {emotion_dict[detected_emotion]} (Number: {detected_emotion})")
        stored_emotion = load_emotion(name)
        if stored_emotion is None:
            # Save the detected emotion for the first time
            save_emotion(name, detected_emotion)
            stored_emotion = detected_emotion
        print(f"Stored Emotion: {emotion_dict[stored_emotion]} (Number: {stored_emotion})")
        if detected_emotion == stored_emotion:
            print("✅ Emotion matched!")
            if recognize_voice(name):  # Voice recognition includes password verification
                print("✅ Voice and password matched!")
                messagebox.showinfo("Access", "🔓 Access Granted!")
            else:
                print("❌ Voice or password not recognized!")
                if messagebox.askyesno("Error", "❌ Voice or password not recognized! Do you want to reset your password?"):
                    reset_password()
        else:
            print("❌ Emotion mismatch!")
            if messagebox.askyesno("Error", "❌ Emotion mismatch! Do you want to reset your password?"):
                reset_password()
    else:
        print("❌ Face not recognized!")
        if messagebox.askyesno("Error", "❌ Face not recognized! Do you want to reset your password?"):
            reset_password()

# GUI Implementation
root = tk.Tk()
root.title("Secure Access System")
root.geometry("300x200")

label = tk.Label(root, text="Facial Recognition Lock", font=("Arial", 12))
label.pack(pady=20)

unlock_button = tk.Button(root, text="Unlock", font=("Arial", 12), command=unlock_system)
unlock_button.pack(pady=10)

reset_button = tk.Button(root, text="Reset Password", font=("Arial", 12), command=reset_password)
reset_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit)
exit_button.pack(pady=10)

root.mainloop()