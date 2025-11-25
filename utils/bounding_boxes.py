import cv2

def draw_boxes(frame, boxes, color=(0,255,0)):
    """
    Draw bounding boxes on a frame.

    boxes = [(x1,y1,x2,y2), ...]
    """
    img = frame.copy()

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

    return img
