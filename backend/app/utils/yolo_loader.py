from ultralytics import YOLO
import os

_MODEL = None

def load_yolo_model():
    global _MODEL
    if _MODEL is None:
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8n.pt")
        if os.path.exists(local_path):
            _MODEL = YOLO(local_path)
        else:
            _MODEL = YOLO("yolov8n.pt")
    return _MODEL
