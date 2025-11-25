import numpy as np
import cv2

def mask_to_box(mask, min_area=150):
    """
    Convert a binary mask to a bounding box.
    mask: (H, W) float32 mask
    Returns (x1, y1, x2, y2)
    """
    # threshold mask
    m = (mask > 0.5).astype("uint8")

    # find contours
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # get largest contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x+w, y+h)



def masks_to_boxes(mask_list):
    """
    mask_list: list of masks for ONE frame
    return list of bounding boxes
    """
    boxes = []
    for mask in mask_list:
        box = mask_to_box(mask)
        if box is not None:
            boxes.append(box)
    return boxes



def extract_video_boxes(video_masks):
    """
    video_masks: list of frame → mask list
    Returns: list of frame → bounding box list
    """
    all_boxes = []
    for frame_masks in video_masks:
        boxes = masks_to_boxes(frame_masks)
        all_boxes.append(boxes)
    return all_boxes
