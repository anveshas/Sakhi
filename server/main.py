r = sr.Recognizer()


# Emotion Classifier CLI — by aardhya
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
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    features = extract_features(audio_path)
    model = joblib.load(model_path)
    prediction = model.predict(features)
    print(f"\nPredicted Emotion: {prediction[0]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classifier CLI — by aardhya")
    parser.add_argument('--audio', type=str, required=True, help='Path to your .wav audio file')
    parser.add_argument('--model', type=str, default='random_forest_model.pkl', help='Path to trained model file')
    args = parser.parse_args()

    print("\nEmotion Classifier — aardhya edition\n")
    predict_emotion(args.audio, args.model)
