import subprocess
import speech_recognition as sr
import cv2
import numpy as np
from keras.models import model_from_json
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import tkinter as tk
import pyttsx3

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.weights.h5")
print("Loaded emotion detection model")

cap = cv2.VideoCapture(0)

detected_emotions = []
recognizing_flag = False
detecting_flag = False
full_text = ""

r = sr.Recognizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {
    "happiness": 0,
    "anger": 1,
    "fear": 2,
    "sadness": 3,
    "surprise": 4,
    "disgust": 5,
    "neutral": 6,
    "joy": 7,
    "anticipation": 8,
    "trust": 9,
    "sad": 10,
    "content": 11,
    "boredom": 12
}

model = AutoModelForSequenceClassification.from_pretrained("bert_tiny_emotion_model")
tokenizer = AutoTokenizer.from_pretrained("bert_tiny_emotion_model")
print("Model and Tokenizer loaded successfully.")

def predict_emotion(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    label_id = torch.argmax(logits).item()
    if label_id < len(label2id):
        return list(label2id.keys())[label_id]
    else:
        return "Unknown Emotion"

def predict_emotion_from_text(text):
    predicted_emotion = predict_emotion(text)
    return predicted_emotion

def detect_face_emotion():
    global detecting_flag
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
    while detecting_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]
            detected_emotions.append(emotion_label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calculate_average_emotion():
    if not detected_emotions:
        return "No emotions detected"

    unique_emotions, counts = np.unique(detected_emotions, return_counts=True)
    mode_emotion = unique_emotions[np.argmax(counts)]

    return mode_emotion

def record_text():
    global recognizing_flag, full_text
    full_text = ""
    while recognizing_flag:
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source)
                MyText = r.recognize_google(audio)
                full_text += " " + MyText  
        except sr.UnknownValueError:
            print()
        except sr.RequestError as e:
            print(f"Speech recognition API error: {e}")


# Initialize the TTS engine
engine = pyttsx3.init()

def stop_detection():
    global detecting_flag, recognizing_flag, cap
    recognizing_flag = False  
    detecting_flag = False   

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    time.sleep(1)  
    avg_face_emotion = calculate_average_emotion()

    if full_text.strip(): 
        predicted_text_emotion = predict_emotion_from_text(full_text)
    else:
        predicted_text_emotion = "No speech detected"

    prompt = f"If face emotion is {avg_face_emotion}, said {full_text.strip()} {predicted_text_emotion}ly. Provide your response within one line (please do not describe , i just want one line response like a human. please please don't describe)."

    print(f"You: {full_text.strip()} ")

    with open("output.txt", "a") as f:
        f.write(prompt + "\n")

    result = subprocess.run(['ollama', 'run', 'deepseek-r1:1.5b', prompt], capture_output=True, text=True, encoding='utf-8')

    if result.returncode == 0:
        output = result.stdout
        think_index = output.find('</think>') 
        if think_index != -1:
            response_after_think = output[think_index + len('</think>'):].strip() 
            print(f"Chatbot: {response_after_think}")
            
            engine.say(response_after_think)
            engine.runAndWait() 
        else:
            print("No </think> tag found in the response.")
    else:
        print("Error occurred while running the command.")



root = tk.Tk()
root.title("Emotion & Speech Recognition")

start_button = tk.Button(root, text="Start", command=lambda: start_detection())
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop", command=stop_detection)
stop_button.pack(pady=10)

cap = None

def start_detection():
    global detecting_flag, recognizing_flag, detected_emotions, cap
    detecting_flag = True
    recognizing_flag = True
    detected_emotions.clear()

    if cap is not None:
        cap.release()  
    cap = cv2.VideoCapture(0) 

    threading.Thread(target=detect_face_emotion, daemon=True).start()
    threading.Thread(target=record_text, daemon=True).start()

root.mainloop()

