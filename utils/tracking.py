def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    return inter / (area1 + area2 - inter + 1e-5)


def track_objects(boxes_per_frame):
    """
    boxes_per_frame = [
        [(x1,y1,x2,y2), ...],  # frame0
        [(x1,y1,x2,y2), ...],  # frame1
    ]

    Returns:
        tracks = {track_id: [boxes...]}
    """

    tracks = {}
    track_id = 0
    last = None

    for frame_boxes in boxes_per_frame:
        if last is None:
            if len(frame_boxes):
                tracks[track_id] = [frame_boxes[0]]
                track_id += 1
            last = frame_boxes
            continue

        for box in frame_boxes:
            best_iou = 0
            best_id = None

            for tid, prev_boxes in tracks.items():
                prev_box = prev_boxes[-1]
                score = iou(box, prev_box)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_iou > 0.3:
                tracks[best_id].append(box)
            else:
                tracks[track_id] = [box]
                track_id += 1

        last = frame_boxes

    return tracks
