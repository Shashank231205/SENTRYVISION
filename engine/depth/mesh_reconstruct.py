import open3d as o3d

def create_mesh_from_pointcloud(pcd):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    return mesh
