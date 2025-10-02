import os
import librosa
import numpy as np
import joblib
import re

# Path to your dataset folder
DATASET_PATH = 'input/ravdess-emotional-speech-audio/'

# Function to extract features from an audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = []
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    features.extend(mfccs)
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features.extend(chroma)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    features.extend(mel)
    return np.array(features)

# Function to parse emotion label from filename (RAVDESS convention)
def get_emotion_from_filename(filename):
    # RAVDESS: 03-01-01-01-01-01-01.wav
    # 3rd group is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    match = re.match(r'\d{2}-\d{2}-(\d{2})-.*', filename)
    if match:
        code = match.group(1)
        return emotion_map.get(code, 'unknown')
    return 'unknown'

features = []
labels = []

for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                try:
                    feat = extract_features(file_path)
                    label = get_emotion_from_filename(file)
                    features.append(feat)
                    labels.append(label)
                    print(f"Processed: {file_path} -> {label}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                # print(f"Total features: {len(features)}, Total labels: {len(labels)}")

# print(f"Total features: {len(features)}, Total labels: {len(labels)}")
# python3 extract_features_and_labels.py              
                
features = np.array(features)
labels = np.array(labels)

joblib.dump(features, 'acoustic_features.joblib')
joblib.dump(labels, 'labels.joblib')
print("Saved features to acoustic_features.joblib and labels to labels.joblib")
