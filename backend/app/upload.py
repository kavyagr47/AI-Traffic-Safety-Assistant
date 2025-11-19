from fastapi import APIRouter, UploadFile, File
import os
import cv2
import numpy as np
from .detection import run_detection_from_image

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}
    detections = run_detection_from_image(img)
    return {"detections": detections}

@router.post("/video")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    uploads_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    outpath = os.path.join(uploads_dir, file.filename)
    with open(outpath, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(outpath)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"error": "could not read video"}
    detections = run_detection_from_image(frame)
    return {"filename": f"/static/uploads/{file.filename}", "detections_sample": detections}
