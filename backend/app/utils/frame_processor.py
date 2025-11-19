import cv2

def resize_for_inference(frame, max_dim=1024):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w*scale), int(h*scale)))
