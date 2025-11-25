import cv2

def webcam_stream(index=0):
    cap = cv2.VideoCapture(index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
