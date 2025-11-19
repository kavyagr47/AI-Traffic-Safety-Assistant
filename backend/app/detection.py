import cv2
from .utils.yolo_loader import load_yolo_model

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = load_yolo_model()
    return _model

def run_detection_from_image(frame):
    model = _get_model()
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        try:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
        except:
            xyxy = box.xyxy[0].numpy()
            x1, y1, x2, y2 = map(int, xyxy.astype(int).tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
        label = results.names[cls] if hasattr(results, "names") else str(cls)
        detections.append({"label": label, "confidence": round(conf,3), "bbox":[x1,y1,x2,y2]})
    return detections

def run_detection_and_draw(frame):
    detections = run_detection_from_image(frame)
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d["label"]
        conf = d["confidence"]
        color = (0,200,0)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(out, text, (x1, max(16,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, detections
