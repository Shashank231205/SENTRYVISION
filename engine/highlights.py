import cv2
import numpy as np
import os
import tempfile

def motion_score(flow):
    fx = flow[...,0]
    fy = flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)
    return float(np.mean(mag))

def extract_highlights(video_path, trajectories, window=40, top_k=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scores = []
    for i in range(min(len(trajectories), total_frames-1)):
        frame_motion = trajectories[i]
        avg_speed = 0.0
        if len(frame_motion) > 0:
            avg_speed = np.mean([s[2] for s in frame_motion])
        scores.append(avg_speed)

    scores = np.array(scores)
    clip_scores = []
    for i in range(0, len(scores)-window, window):
        clip_scores.append((i, float(np.mean(scores[i:i+window]))))

    clip_scores.sort(key=lambda x: x[1], reverse=True)
    best_clips = clip_scores[:top_k]

    output_clips = []
    for start, _ in best_clips:
        start_time = max(0, start / fps)
        end_time = min((start + window) / fps, total_frames / fps)

        out = save_clip(video_path, start_time, end_time)
        output_clips.append(out)

    return output_clips

def save_clip(video_path, start_s, end_s):
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "highlight.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w = int(cap.get(3))
    h = int(cap.get(4))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    f = start_frame
    while f < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()

    return out_path
