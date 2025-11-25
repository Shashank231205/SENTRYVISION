import numpy as np

def lift_2d_to_3d(keypoints_2d):
    keypoints_3d = []
    for frame in keypoints_2d:
        if frame is None:
            keypoints_3d.append(None)
            continue
        pts3d = []
        for (x, y) in frame:
            z = (1 - abs(x - 0.5)) * 0.3
            pts3d.append([x, y, z])
        keypoints_3d.append(pts3d)
    return keypoints_3d
