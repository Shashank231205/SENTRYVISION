def detect_eye_contact(gaze):
    if gaze["direction"] == "center":
        return True
    return False
