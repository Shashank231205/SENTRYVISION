import cv2
import numpy as np
import os, tempfile

def detect_shot_changes(video_path, threshold=35):
    cap = cv2.VideoCapture(video_path)
    prev = None
    keyframes = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            score = np.mean(diff)
            if score > threshold:
                keyframes.append((idx, frame))
        prev = gray
        idx += 1

    cap.release()
    return keyframes

def save_keyframes(keyframes):
    files = []
    for i, (idx, frame) in enumerate(keyframes):
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, f"keyframe_{i}.jpg")
        cv2.imwrite(out, frame)
        files.append(out)
    return files
