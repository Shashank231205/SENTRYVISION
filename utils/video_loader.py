import torch
from core.vision_encoder.video_utils import load_video_tensor

def load_video(video_path, num_frames=64):
    """
    Loads a video into a tensor shape [1, T, C, H, W]
    """
    tensor = load_video_tensor(video_path, num_frames=num_frames)
    return tensor.cuda()
