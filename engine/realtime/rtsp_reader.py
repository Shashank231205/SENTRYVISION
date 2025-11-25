import cv2

def rtsp_stream(url):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
