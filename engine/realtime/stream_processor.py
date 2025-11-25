import cv2, torch
from engine.segmentation import extract_video_masks
from engine.detection import extract_video_boxes
from engine.pose import run_pose_estimation, draw_skeleton
from engine.flow import compute_flow_frame

def process_live_frame(frame, pe_spatial, pose_model, prev_gray):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0).cuda() / 255

    with torch.no_grad(), torch.autocast("cuda"):
        spatial = pe_spatial.encode_image(tensor)

    masks = extract_video_masks([spatial], [(frame.shape[0], frame.shape[1])])[0]
    boxes = extract_video_boxes([masks])[0]

    poses, _ = run_pose_estimation(frame, pose_model, single_frame=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = compute_flow_frame(prev_gray, gray) if prev_gray is not None else None

    for b in boxes:
        x1,y1,x2,y2 = b
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    if poses:
        frame = draw_skeleton(frame, poses)

    prev_gray = gray
    return frame, prev_gray
