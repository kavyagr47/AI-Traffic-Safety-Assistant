# backend/app/video_stream.py
import cv2
import time
from threading import Thread, Lock
from .detection import run_detection_and_draw

# Globals
_camera = None
_camera_lock = Lock()
_alerts = []
_MAX_ALERTS = 500
_latest_frame = None
_camera_thread = None
_stop_thread = False

# Background camera loop
def _camera_loop():
    global _camera, _latest_frame, _alerts, _stop_thread
    while not _stop_thread:
        if _camera is None:
            time.sleep(0.05)
            continue
        ret, frame = _camera.read()
        if ret and frame is not None:
            # Run detection & draw bounding boxes
            frame_with_boxes, detections = run_detection_and_draw(frame)
            _latest_frame = frame_with_boxes

            # Update alerts buffer
            if detections:
                ts = time.strftime("%H:%M:%S")
                for d in detections:
                    _alerts.append({"time": ts, "label": d["label"], "confidence": d["confidence"]})
                if len(_alerts) > _MAX_ALERTS:
                    _alerts = _alerts[-_MAX_ALERTS:]

        time.sleep(0.03)  # ~30 FPS

def start_camera():
    global _camera, _camera_thread, _stop_thread
    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(0)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            time.sleep(0.5)
        if _camera_thread is None:
            _stop_thread = False
            _camera_thread = Thread(target=_camera_loop, daemon=True)
            _camera_thread.start()

def stop_camera():
    global _camera, _camera_thread, _stop_thread
    _stop_thread = True
    with _camera_lock:
        if _camera is not None:
            try:
                _camera.release()
            except Exception:
                pass
            _camera = None
    _camera_thread = None

def get_frame_bytes():
    global _latest_frame
    if _latest_frame is None:
        return None
    ret, jpeg = cv2.imencode(".jpg", _latest_frame)
    if not ret:
        return None
    return jpeg.tobytes()

def get_alerts_buffer():
    global _alerts
    return {"alerts": list(reversed(_alerts[-20:]))}
