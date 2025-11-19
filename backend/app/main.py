from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
import cv2
import numpy as np

from .video_stream import start_camera, stop_camera, get_frame_bytes, get_alerts_buffer
from .detection import run_detection_from_image

app = FastAPI(title="AI Traffic & Pedestrian Safety Assistant")

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running. Use /docs for API docs."}

@app.get("/start_camera")
def api_start_camera():
    start_camera()
    return {"status": "camera_started"}

@app.get("/stop_camera")
def api_stop_camera():
    stop_camera()
    return {"status": "camera_stopped"}

@app.get("/stream")
def api_stream_frame():
    frame_bytes = get_frame_bytes()
    if frame_bytes is None:
        return Response(status_code=204, content=b"")
    return Response(content=frame_bytes, media_type="image/jpeg")

@app.get("/alerts")
def api_alerts():
    return JSONResponse(get_alerts_buffer())

@app.post("/detect-frame")
async def api_detect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    detections = run_detection_from_image(img)
    return {"detections": detections}
