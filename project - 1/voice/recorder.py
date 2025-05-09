import os
import sounddevice as sd
import soundfile as sf

# Configuration
fs = 44100  # Sample rate
duration = 3  # Duration in seconds
person_name = input("Enter person's name: ")  # Name of speaker
save_path = f"C:\\Projects\\project - 1\\voice\\voice_dataset\\{person_name}"
os.makedirs(save_path, exist_ok=True)

# Record multiple samples
num_samples = int(input("How many samples to record? "))
for i in range(num_samples):
    print(f"Recording sample {i+1}/{num_samples}... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    filename = os.path.join(save_path, f"sample{i+1}.wav")
    sf.write(filename, audio, fs)
    print(f"Saved: {filename}")

print("Recording complete! Your dataset is ready.")
