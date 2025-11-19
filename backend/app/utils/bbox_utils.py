def xyxy_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
