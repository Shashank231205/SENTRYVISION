import numpy as np
import cv2

def draw_3d_skeleton(image, joints):
    if joints is None: return image
    for x, y, z in joints:
        px = int(x * image.shape[1])
        py = int(y * image.shape[0])
        cv2.circle(image, (px, py), 4, (255,0,0), -1)
    return image
