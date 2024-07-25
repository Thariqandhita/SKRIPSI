import os
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Function to extract prosody features
def extract_features_without_intensity(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    stddev_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_harmonicity = call(harmonicity, "Get mean", 0, 0)

    intensity = call(snd, "To Intensity", 75, 0, "yes")
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    
    duration = snd.get_total_duration()
    
    return np.array([mean_pitch, stddev_pitch, mean_harmonicity, mean_intensity, duration])

# Function to load data and extract features
def load_and_predict(audio_paths, model_path, scaler_path):
    features = []
    
    # Load scaler and model
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    
    # Process each audio file
    for path in audio_paths:
        # Extract features
        prosody_features = extract_features_without_intensity(path)
        
        # Print prosody features
        print(f"Prosody Features for {os.path.basename(path)}:")
        print(f"Mean Pitch: {prosody_features[0]:.2f} Hz")
        print(f"Pitch Standard Deviation: {prosody_features[1]:.2f} Hz")
        print(f"Mean Harmonicity: {prosody_features[2]:.2f} dB")
        print(f"Duration: {prosody_features[3]:.2f} seconds")
        print()  # New line for separation
        
        features.append(prosody_features)
    
    # Convert list to numpy array
    features = np.array(features)
    
    # Normalize features
    features = scaler.transform(features)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    
    # Predict using the model
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert predicted classes to labels
    labels = ['Non-Depression' if pred == 0 else 'Depression' for pred in predicted_classes]
    return labels

# Example usage
if __name__ == '__main__':
    audio_paths = [
        "Recording (76).wav",
        "Recording (21).wav",
        "Recording (22).wav",
        "Recording (20).wav",
        "Test-suara.wav",
        "04-01Dataset313.wav",
        "02-05Dataset313.wav",
        "03-03Dataset313.wav",
        "05-02Dataset313.wav",
        "01-04Dataset313.wav"
    ]
    model_path = 'best_model_skripsi_prosody_lebih_baru.h5'
    scaler_path = 'scaler_skripsi_prosody_lebih_beru.pkl'
    
    predictions = load_and_predict(audio_paths, model_path, scaler_path)
    print(predictions)
