
# Emotion Classifier CLI
# This script predicts the emotion in a speech audio file using a Random Forest model.

import argparse
import joblib
import librosa
import numpy as np
import os

def extract_features(file_path):
    """
    Extract MFCC, Chroma, and Mel Spectrogram features from an audio file.
    """
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = []
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    features.extend(mfccs)
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features.extend(chroma)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    features.extend(mel)
    return np.array(features).reshape(1, -1)

def predict_emotion(audio_path, model_path):
    """
    Predict emotion from audio using the trained model.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        print(f"Model file not found or empty: {model_path}\nTraining a new Random Forest model...")
        train_and_save_model()
    features = extract_features(audio_path)
    model = joblib.load(model_path)
    prediction = model.predict(features)
    print(f"\nPredicted Emotion: {prediction[0]}\n")

def train_and_save_model():
    """
    Train a Random Forest model using acoustic_features.joblib and labels.joblib, then save it.
    """
    from sklearn.ensemble import RandomForestClassifier
    features_path = 'acoustic_features.joblib'
    labels_path = 'labels.joblib'
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Feature or label file missing. Cannot train model.")
        return
    X = joblib.load(features_path)
    y = joblib.load(labels_path)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, 'random_forest_model.pkl')
    print("Model trained and saved as random_forest_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classifier CLI")
    parser.add_argument('--audio', type=str, required=True, help='Path to your .wav audio file')
    parser.add_argument('--model', type=str, default='random_forest_model.pkl', help='Path to trained model file')
    args = parser.parse_args()

    print("\nEmotion Classifier\n")
    predict_emotion(args.audio, args.model)
