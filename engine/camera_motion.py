import cv2
import numpy as np

def analyze_camera_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None

    pan, tilt, zoom, shake = 0,0,0,0
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None, 0.5,3,15,3,5,1.2,0)
            fx, fy = flow[...,0], flow[...,1]

            pan += np.mean(fx)
            tilt += np.mean(fy)
            zoom += np.mean(np.abs(fx) + np.abs(fy))
            shake += np.std(fx) + np.std(fy)

        prev_gray = gray
        frames += 1

    cap.release()

    return {
        "pan_score": float(pan / frames),
        "tilt_score": float(tilt / frames),
        "zoom_score": float(zoom / frames),
        "shake_score": float(shake / frames),
        "cinematic_rating": rate_camera_motion(pan, tilt, zoom, shake)
    }

def rate_camera_motion(pan, tilt, zoom, shake):
    if shake > 20:
        return "Shaky / Handheld"
    if zoom > 50:
        return "Zoom Motion"
    if abs(pan) > abs(tilt):
        return "Smooth Pan"
    if abs(tilt) > abs(pan):
        return "Smooth Tilt"
    return "Static Camera"
