import joblib
import librosa
import numpy as np
import os


def extract_features(file_path):
    """
    Extract MFCC, Chroma, and Mel Spectrogram features from an audio file.
    Returns a 2D numpy array shaped (1, features).
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


def train_and_save_model(features_path='acoustic_features.joblib', labels_path='labels.joblib', out_path='random_forest_model.pkl'):
    """
    Train a RandomForestClassifier using saved features and labels and persist the model.
    """
    from sklearn.ensemble import RandomForestClassifier
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError('Missing features or labels for training')
    X = joblib.load(features_path)
    y = joblib.load(labels_path)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    return out_path


if __name__ == '__main__':
    # simple CLI to train or extract
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--audio', type=str)
    args = parser.parse_args()
    if args.train:
        train_and_save_model()
    elif args.audio:
        print(extract_features(args.audio).shape)
    else:
        parser.print_help()
