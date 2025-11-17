# emotions

Minimal CLI tool to predict emotions from speech audio using a Random Forest model.

## Usage
```bash
python main.py --audio project.wav
```

## Web app (optional)
This repo now includes a simple web demo and API for integrating emotion detection into a safety app.

Run the API (from the `emotions` folder):

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000` in your browser to access the demo recorder UI. When a distressing emotion is detected the server will attempt to send alerts using configured contacts (see `.env.example`).

## Files added for the web demo
- `api.py` — FastAPI server exposing `/api/predict` and serving static UI
- `alerts.py` — Twilio-based alert helper (stubbed when credentials missing)
- `static/index.html` — simple recorder + geolocation UI
- `requirements.txt` — required packages for the web API
- `.env.example` — example environment variables for alerting
