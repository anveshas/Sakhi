# 🌸 Sakhi – Women’s Safety & Speech Emotion Detection System

## Project Overview

Sakhi is an AI-powered safety system that detects emotional distress from speech using machine learning.
It analyzes audio using features like MFCC, Chroma, and Mel Spectrogram, and classifies emotions using a Random Forest model trained on the RAVDESS dataset.
This system can be extended to send emergency alerts to contacts, nearby police stations, or women’s helplines whenever a distress emotion is detected.

## 🚀 Features
- Extracts MFCC, Chroma, and Mel Spectrogram features from audio
- Predicts emotion using a Random Forest classifier
- Extendable alert system for real-world women’s safety
- Supports multiple emotions (happy, sad, angry, fear, etc.)


## 🛠 Tech Stack
- Machine Learning & Audio Processing: librosa (MFCC, Chroma, Mel-Spectrogram extraction), scikit-learn (Random Forest classifier), numpy, soundfile, RAVDESS Emotional Speech Dataset
- Backend: FastAPI, Uvicorn ASGI server
- Frontend: HTML5, CSS, JavaScript, MediaRecorder API, Geolocation API
- Alerting & Communication: Twilio API
