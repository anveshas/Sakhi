# Sakhi

## Project Overview

This project detects emotions from speech audio files using machine learning. It uses the RAVDESS dataset and a Random Forest model to classify emotions based on acoustic features.

## Features
- Extracts MFCC, Chroma, and Mel Spectrogram features from audio
- Predicts emotion using a Random Forest classifier
- Simple command-line workflow

## Dataset
- [RAVDESS Emotional Speech Audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

## How It Works
1. Load an audio file from the RAVDESS dataset
2. Extract acoustic features
3. Use the trained Random Forest model to predict the emotion
4. Print the result

## Requirements
- Python 3.8+
- librosa
- scikit-learn
- numpy
- soundfile

## Usage

```bash
python main.py --audio path/to/audio.wav
```

## Customization
You can easily extend this project by:
- Training with your own audio data
- Adding new features or models
- Modifying the prediction logic in `main.py`

---
