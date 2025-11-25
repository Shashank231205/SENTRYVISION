import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth_map):
    h, w = len(depth_map), len(depth_map[0])
    pts = []
    for y in range(h):
        for x in range(w):
            z = depth_map[y][x]
            pts.append([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    return pcd
