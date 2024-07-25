import tkinter as tk
from tkinter import filedialog
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import parselmouth
from parselmouth.praat import call
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

class DepressionDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Voice Depression Detection System")
        master.geometry("800x600")
        
        # Warna utama diinspirasi oleh Chelsea: biru gelap
        bg_color = "#034694"  # Dark blue
        text_color = "#FFFFFF"  # White for contrast
        
        master.configure(bg=bg_color)

        self.label = tk.Label(master, text="Select an audio file or record your voice:",
                              bg=bg_color, fg=text_color, font=("Arial", 14))
        self.label.pack(pady=20)

        self.open_button = tk.Button(master, text="Select Audio File", command=self.select_file,
                                     bg="#0A74DA", fg=text_color, font=("Arial", 12), height=2, width=20)
        self.open_button.pack(pady=10)

        self.record_button = tk.Button(master, text="Record Voice", command=self.start_recording_thread,
                                       bg="#0A74DA", fg=text_color, font=("Arial", 12), height=2, width=20)
        self.record_button.pack(pady=10)

        self.status_label = tk.Label(master, text="", bg=bg_color, fg=text_color, font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.result_label = tk.Label(master, text="", bg=bg_color, fg=text_color, font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.probability_label = tk.Label(master, text="", bg=bg_color, fg=text_color, font=("Arial", 14))
        self.probability_label.pack(pady=20)

    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.status_label.config(text="Predicting, please wait...")
            self.predict(filename)

    def start_recording_thread(self):
        self.status_label.config(text="Recording...")
        threading.Thread(target=self.record_voice, daemon=True).start()

    def record_voice(self):
        duration = 5  # seconds
        fs = 22050  # Sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        audio_path = 'temp_recording.wav'
        sf.write(audio_path, recording.flatten(), fs)
        self.status_label.config(text="Recording finished. Predicting, please wait...")
        self.predict(audio_path)

    def predict(self, audio_path):
        try:
            result, probability = self.load_and_predict(audio_path, 'best_model_skripsi_prosody_lebih_baru.h5', 'scaler_skripsi_prosody_lebih_beru.pkl')
            self.status_label.config(text=f"Prediction: {result} - Probability: {probability:.2f}%")
            self.result_label.config(text="")
            self.probability_label.config(text="")
        except Exception as e:
            self.status_label.config(text="An error occurred.")
            self.result_label.config(text=f"An error occurred: {str(e)}")

    def load_and_predict(self, audio_path, model_path, scaler_path):
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        prosody_features = self.extract_prosody_features(audio_path)
        features = np.array([prosody_features])
        features = scaler.transform(features)
        features = features.reshape(features.shape[0], features.shape[1], 1)
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)
        probability = np.max(predictions) * 100
        return ('Non-Depression' if predicted_class == 0 else 'Depression', probability)

    def extract_prosody_features(self, audio_path):
        snd = parselmouth.Sound(audio_path)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        stddev_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        mean_harmonicity = call(harmonicity, "Get mean", 0, 0)
        duration = snd.get_total_duration()
        return np.array([mean_pitch, stddev_pitch, mean_harmonicity, duration])

if __name__ == '__main__':
    root = tk.Tk()
    app = DepressionDetectionApp(root)
    root.mainloop()