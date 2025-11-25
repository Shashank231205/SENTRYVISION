import cv2
import numpy as np
import torch
from mmpose.apis import init_model, inference_topdown


# ============================================================
# Load RTMPose Model (Lite version)
# ============================================================

def load_pose_model():
    config = "https://raw.githubusercontent.com/open-mmlab/mmpose/master/configs/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py"
    checkpoint = "https://download.openmmlab.com/mmpose/top_down/rtmpose/rtmpose-t_8xb256-420e_coco-256x192-ccps.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config, checkpoint, device=device)
    return model


# ============================================================
# Pose Estimation for One Frame
# ============================================================

def estimate_pose(model, frame):
    """
    Returns:
    keypoints: (17, 3) -> x, y, score
    """
    h, w = frame.shape[:2]

    # Create fake detection box for whole frame
    results = inference_topdown(model, frame, person_results=[{'bbox': [0, 0, w, h]}])

    if len(results) == 0:
        return None

    pose = results[0]['keypoints']   # shape (17, 3)
    return pose


# ============================================================
# Extract Joint Angles (Elbow, Knee, etc.)
# ============================================================

def angle(a, b, c):
    """
    Compute angle between three joints a-b-c (in degrees)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return float(deg)


def compute_joint_angles(keypoints):
    """
    Uses COCO keypoint format (17 points)
    """
    kp = keypoints

    angles = {
        "left_elbow": angle(kp[5], kp[7], kp[9]),
        "right_elbow": angle(kp[6], kp[8], kp[10]),
        "left_knee": angle(kp[11], kp[13], kp[15]),
        "right_knee": angle(kp[12], kp[14], kp[16]),
    }
    return angles


# ============================================================
# Draw Skeleton on Frame
# ============================================================

COCO_PAIRS = [
    (5,7), (7,9),
    (6,8), (8,10),
    (11,13), (13,15),
    (12,14), (14,16),
    (5,6), (11,12), (5,11), (6,12)
]

def draw_skeleton(frame, kps):
    for (a, b) in COCO_PAIRS:
        x1, y1, s1 = kps[a]
        x2, y2, s2 = kps[b]
        if s1 > 0.3 and s2 > 0.3:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 255, 255), 2)

    # Draw joints
    for x, y, s in kps:
        if s > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    return frame


# ============================================================
# Process Entire Video (Extract pose per frame)
# ============================================================

def run_pose_estimation(video_path, model):
    cap = cv2.VideoCapture(video_path)
    poses = []
    angles_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose = estimate_pose(model, frame)
        if pose is None:
            poses.append(None)
            angles_per_frame.append({})
            continue

        angles = compute_joint_angles(pose)
        poses.append(pose.tolist())
        angles_per_frame.append(angles)

    cap.release()
    return poses, angles_per_frame
