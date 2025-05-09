import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PATH"] += os.pathsep + r"C:\Users\munag\Downloads\ffmpeg-2025-03-10-git-87e5da9067-full_build\bin"

# Load Whisper model
model = whisper.load_model("small")  # Change model size for better accuracy

# Function to record audio
def record_audio(filename="test_audio.wav", duration=3, fs=44100):
    input("Press ENTER and say the unlock phrase...")  # Wait for user to start
    print("Recording started... Speak now.")
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    
    sf.write(filename, recording, fs)
    print("Recording saved as", filename)

    input("Press ENTER to stop and verify...")  # Wait for user to stop
    print("Verifying speech...")

# Function to verify if the correct phrase was spoken
def verify_speech(required_phrase="I am Batman", filename="test_audio.wav"):
    # Transcribe the audio
    result = model.transcribe(filename)
    recognized_text = result["text"].strip().lower()

    print(f"Recognized Text: {recognized_text}")

    # Check if recognized text matches the required phrase
    if recognized_text == required_phrase.lower():
        print("✅ Access Granted: Unlocking the device")
    else:
        print("❌ Access Denied: Incorrect phrase. Please try again.")

# ---- Run Speech-Based Unlock ----
required_phrase = "I am Batman"  # Set your required unlock phrase
record_audio()  # Start recording after user input
verify_speech(required_phrase)  # Check if it matches

