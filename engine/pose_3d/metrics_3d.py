import numpy as np

def compute_joint_angles_3d(joints3d):
    if joints3d is None: return {}
    angles = {}
    for i in range(1, len(joints3d)-1):
        a = np.array(joints3d[i-1])
        b = np.array(joints3d[i])
        c = np.array(joints3d[i+1])
        ang = np.degrees(np.arccos(
            np.dot(a-b, c-b) / (np.linalg.norm(a-b)*np.linalg.norm(c-b)+1e-6)
        ))
        angles[f"joint_{i}"] = float(ang)
    return angles
