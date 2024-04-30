import open3d as o3d
import trimesh
import numpy as np

PATH_DATA = './vis_results/Area_5_office_35_pred.obj'

if __name__ == '__main__':
    mesh = trimesh.load(PATH_DATA)

    vertices = np.array(mesh.vertices)
    vertex_colors = np.array(mesh.visual.vertex_colors)/255.
    print(vertex_colors.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(vertex_colors[:,:3])

    o3d.visualization.draw_geometries([pcd])
