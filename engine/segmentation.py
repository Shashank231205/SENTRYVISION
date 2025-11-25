import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def tokens_to_heatmap(tokens, height, width):
    """
    Convert PE-Spatial tokens (C,H,W) into a normalized heatmap.
    """
    heat = torch.norm(tokens, dim=0)  # [H,W]

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

    heat = F.interpolate(
        heat.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    return heat.squeeze().cpu().numpy()


def extract_masks_from_tokens(tokens, height, width, k=3, threshold=0.35):
    """
    Extract top-K segmentation masks from PE-Spatial tokens.
    """
    heat = tokens_to_heatmap(tokens, height, width)

    h, w = heat.shape
    flat = heat.reshape(-1, 1)

    km = KMeans(n_clusters=k, n_init=8)
    labels = km.fit_predict(flat).reshape(h, w)

    masks = []
    for cid in range(k):
        mask = (labels == cid).astype("float32")

        # Remove extremely small clusters
        if mask.sum() / (h*w) < threshold:
            continue

        masks.append(mask)

    return masks


def extract_video_masks(spatial_tokens, frame_sizes, k=3):
    """
    Create masks for each frame using spatial tokens.

    spatial_tokens: list of [C,H,W] tokens
    frame_sizes: list of (H,W)
    """
    all_masks = []

    for i, tokens in enumerate(spatial_tokens):
        h, w = frame_sizes[i]
        masks = extract_masks_from_tokens(tokens, h, w, k=k)
        all_masks.append(masks)

    return all_masks
