import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
from keras.utils import to_categorical
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, MaxPooling1D, SpatialDropout1D, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ProcessPoolExecutor

def extract_prosody_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)  # Ekstrak pitch
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    stddev_pitch = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_harmonicity = call(harmonicity, "Get mean", 0, 0)
    
    intensity = call(snd, "To Intensity", 75, 0, "yes")
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    
    duration = snd.get_total_duration()  # Dapatkan durasi suara
    
    return np.array([mean_pitch, stddev_pitch, mean_harmonicity, mean_intensity, duration])

def extract_features(audio_path):
    return extract_prosody_features(audio_path)

def load_data(non_depression_folder, depression_folder):
    non_depression_paths = [os.path.join(non_depression_folder, f) for f in os.listdir(non_depression_folder) if f.endswith('.wav')]
    depression_paths = [os.path.join(depression_folder, f) for f in os.listdir(depression_folder) if f.endswith('.wav')]
    audio_paths = non_depression_paths + depression_paths
    labels = [0] * len(non_depression_paths) + [1] * len(depression_paths)
    
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_features, audio_paths))
    
    features = np.array(features)
    return features, np.array(labels), non_depression_paths, depression_paths

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
        
        pitch_shift = np.random.randint(-3, 3)
        augmented_X.append(librosa.effects.pitch_shift(x.flatten(), sr=22050, n_steps=pitch_shift).reshape(x.shape))
        augmented_Y.append(y)
        
    return np.array(augmented_X), np.array(augmented_Y)

def build_enhanced_model(input_shape):
    model = Sequential([
        Conv1D(128, padding='same', kernel_size=3, input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1, padding='same'),
        Dropout(0.2),

        Conv1D(256, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1, padding='same'),
        Dropout(0.4),

        Conv1D(512, padding='same', kernel_size=3),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=1, padding='same'),
        Dropout(0.4),

        GlobalAveragePooling1D(),
        Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def process_and_train(X, Y, epochs=150):
    le = LabelEncoder()
    dataset_y_encoded = le.fit_transform(Y)
    dataset_y_onehot = to_categorical(dataset_y_encoded)
    X, scaler = normalize_data(X)
    
    # Simpan scaler
    joblib.dump(scaler, 'scaler_skripsi_prosody_lebih_beru.pkl')
    
    X_augmented, Y_augmented = augment_data(X, dataset_y_onehot)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_augmented, Y_augmented, test_size=0.2, random_state=42, stratify=Y_augmented)
    
    model = build_enhanced_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("best_model_skripsi_prosody_lebih_baru.h5", monitor='val_accuracy', save_best_only=True)
    
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    return model, history

if __name__ == '__main__':
    non_depression_folder = "C:/Users/THORIQ/Documents/Skripsi Alat/ravdes/NonDepression"
    depression_folder = "C:/Users/THORIQ/Documents/Skripsi Alat/Depression Sound"
    X, Y, non_depression_paths, depression_paths = load_data(non_depression_folder, depression_folder)
    trained_model, history = process_and_train(X, Y)

    Y_onehot = to_categorical(Y)
    X, _ = normalize_data(X)

    best_model = tf.keras.models.load_model("best_model_skripsi_prosody_lebih_baru.h5")
    best_evaluation = best_model.evaluate(X, Y_onehot)
    print(f"Best Model Final Loss: {best_evaluation[0]}")
    print(f"Best Model Final Accuracy: {best_evaluation[1]}")