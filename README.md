# 🌸 Sakhi – Women’s Safety & Speech Emotion Detection System

## Project Overview

Sakhi is an AI-powered safety system that detects emotional distress from speech using machine learning.
It analyzes audio using features like MFCC, Chroma, and Mel Spectrogram, and classifies emotions using a Random Forest model trained on the RAVDESS dataset.
This system can be extended to send emergency alerts to contacts, nearby police stations, or women’s helplines whenever a distress emotion is detected.

## 🚀 Features
- Extracts MFCC, Chroma, and Mel Spectrogram features from audio
- Predicts emotion using a Random Forest classifier
- Simple command-line workflow
- Extendable alert system for real-world women’s safety

## Dataset
- [RAVDESS Emotional Speech Audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

## Usage

```bash
python main.py --audio path/to/audio.wav
```

---
