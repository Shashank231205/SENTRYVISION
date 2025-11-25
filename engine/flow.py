import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Basic Optical Flow (Farneback / GPU-friendly)
# ============================================================

def compute_optical_flow(prev_frame, next_frame):
    """
    Computes dense optical flow between two frames.
    Returns flow field in (dx, dy) format.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        0.5, 3, 15, 3, 5, 1.1, 0
    )
    return flow   # shape (H, W, 2)


# ============================================================
# Convert flow → colored visualization
# ============================================================

def flow_to_color(flow):
    """
    Converts optical flow to HSV+RGB visualization.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    hsv[..., 0] = ang * 180 / np.pi / 2      # Hue = direction
    hsv[..., 1] = 255                        # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ============================================================
# Track movement inside segmentation masks
# ============================================================

def extract_motion_vectors(flow, mask):
    """
    Average motion inside segmentation mask.
    Returns (dx, dy, speed)
    """
    m = mask > 0.5
    fx = flow[..., 0][m]
    fy = flow[..., 1][m]

    if len(fx) == 0:
        return (0.0, 0.0, 0.0)

    dx = float(np.mean(fx))
    dy = float(np.mean(fy))
    speed = float(np.sqrt(dx**2 + dy**2))

    return dx, dy, speed


# ============================================================
# Track all objects → trajectories
# ============================================================

def track_motion(video_path, video_masks):
    """
    Computes:
     - Flow for each frame
     - Motion vector for each object mask
     - Trajectory list per object
    """
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        return []

    trajectories = []  # list of [frame → (dx,dy,speed)]

    f = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute flow
        flow = compute_optical_flow(prev, frame)

        frame_motions = []
        if f < len(video_masks):
            for mask in video_masks[f]:
                dx, dy, speed = extract_motion_vectors(flow, mask)
                frame_motions.append((dx, dy, speed))

        trajectories.append(frame_motions)
        prev = frame
        f += 1

    cap.release()
    return trajectories
