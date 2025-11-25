import numpy as np
import json
import tempfile
import os

def detect_chapters(trajectories, min_gap=45):
    motion = []
    for f in range(len(trajectories)):
        frame = trajectories[f]
        if len(frame) == 0:
            motion.append(0)
        else:
            motion.append(float(np.mean([s[2] for s in frame])))

    motion = np.array(motion)

    threshold = np.percentile(motion, 60)

    chapters = []
    start = 0
    for i in range(1, len(motion)):
        if abs(motion[i] - motion[i-1]) > threshold and (i - start) > min_gap:
            chapters.append({
                "start_frame": start,
                "end_frame": i,
                "avg_motion": float(np.mean(motion[start:i]))
            })
            start = i

    chapters.append({
        "start_frame": start,
        "end_frame": len(motion)-1,
        "avg_motion": float(np.mean(motion[start:]))
    })

    return chapters

def save_chapters_json(chapters):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "chapters.json")
    with open(path, "w") as f:
        json.dump(chapters, f, indent=4)
    return path
