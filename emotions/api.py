from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os
import joblib
from main_clean import extract_features, train_and_save_model
from alerts_clean import send_alert

app = FastAPI(title="Emotion Safety API")

# Allow local testing from file:// or other hosts during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static UI from /static and provide an index at /
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    return JSONResponse({"message": "Static UI not found"}, status_code=404)

MODEL_PATH = os.environ.get("MODEL_PATH", "random_forest_model.pkl")
EMERGENCY_LABELS = {"fearful", "angry", "disgust", "sad"}


@app.post("/api/predict")
async def predict(audio: UploadFile = File(...), lat: float = Form(None), lon: float = Form(None), user_id: str = Form(None), auto_alert: bool = Form(True)):
    """Accepts an audio file and optional location; returns predicted emotion and triggers alerts when needed."""
    # save to temp file
    try:
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name
    finally:
        audio.file.close()

    # ensure model exists (train if missing)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        train_and_save_model()

    # load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load model: {e}"}, status_code=500)

    # extract features and predict
    try:
        feats = extract_features(tmp_path)
        pred = model.predict(feats)[0]
        score = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(feats)
                # find index of predicted class
                classes = model.classes_
                idx = list(classes).index(pred)
                score = float(proba[0][idx])
            except Exception:
                score = None
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse({"error": f"Feature extraction or prediction error: {e}"}, status_code=500)

    os.remove(tmp_path)

    result = {"emotion": str(pred), "score": score, "alert_sent": False}

    # trigger alert when predicted label is in emergency set
    if pred in EMERGENCY_LABELS and auto_alert:
        # For now, send a simple alert message. The send_alert function reads from env vars.
        message = f"Emergency detected: {pred}. Location: {lat},{lon}."
        sent = send_alert(message, lat=lat, lon=lon)
        result["alert_sent"] = bool(sent)

    return JSONResponse(result)


@app.get("/api/health")
def health():
    return {"status": "ok"}
