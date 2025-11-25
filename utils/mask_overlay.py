import cv2
import numpy as np

def overlay_masks(frame, masks, alpha=0.35):
    """
    Takes a BGR frame + list of masks.
    Each mask is a (H,W) boolean or 0/1 float array.
    """
    out = frame.copy()

    for m in masks:
        mask = (m * 255).astype("uint8")
        color = np.random.randint(0,255,(1,3),dtype="uint8")

        overlay = np.zeros_like(frame)
        overlay[:,:,0] = mask * color[0][0]
        overlay[:,:,1] = mask * color[0][1]
        overlay[:,:,2] = mask * color[0][2]

        out = cv2.addWeighted(out, 1, overlay, alpha, 0)

    return out
