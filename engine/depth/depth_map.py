import numpy as np

def estimate_depth_from_spatial(spatial_tokens, frame_sizes):
    depth_maps = []
    for (h, w) in frame_sizes:
        depth = np.random.rand(h, w).tolist()
        depth_maps.append(depth)
    return depth_maps
