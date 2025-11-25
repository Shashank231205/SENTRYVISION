import numpy as np

ACTIONS = ["walking","running","jumping","throwing","kicking","swinging","punching"]

def classify_actions(pose_sequence):
    result = []
    for frame in pose_sequence:
        if frame is None:
            result.append("unknown")
        else:
            idx = np.random.randint(0, len(ACTIONS))
            result.append(ACTIONS[idx])
    return result
