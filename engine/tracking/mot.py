import numpy as np

def assign_ids(boxes):
    ids=[]
    for i,b in enumerate(boxes):
        ids.append({"id":i,"box":b})
    return ids

def track_sequence(all_boxes):
    seq=[]
    for frame_boxes in all_boxes:
        seq.append(assign_ids(frame_boxes))
    return seq
