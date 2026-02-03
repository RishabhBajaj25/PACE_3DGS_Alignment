import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import open3d as o3d
import os.path as osp
import os

base_dir = "/home/pace-ubuntu/datasets/leica/EAST/registration"
output_name = "rough"
output_dir = osp.join(base_dir, output_name)
os.makedirs(output_dir, exist_ok=True)

result_T_m2_m1 = np.array([
    [-1.548305869102, 0.002351583214, -0.016247715801, 1787.160034179688],
    [0.016255933791, 0.005465727299, -1.548297882080, 2714.197021484375],
    [-0.002294085687, -1.548381447792, -0.005490107927, 177.568634033203],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
])

R_s = result_T_m2_m1[:3, :3]

# Compute scale as the mean of the column norms
scale = np.mean([np.linalg.norm(R_s[:,0]),
                 np.linalg.norm(R_s[:,1]),
                 np.linalg.norm(R_s[:,2])])
print("Scale factor:", scale)

R = R_s / scale

rot = R_scipy.from_matrix(R)
euler_angles = rot.as_euler('zyx', degrees=True)  # or degrees=False
print("Euler angles (zyx):", euler_angles)

t = result_T_m2_m1[:3, 3]
print("Translation:", t)

# test_M2 = o3d.io.read_point_cloud('/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/density_test.ply')
test_M2 = o3d.io.read_point_cloud("/home/pace-ubuntu/datasets/leica/EAST/registration/supersplat_export_50000 - Cloud.ply")

test_M2.scale(scale, center = (0,0,0))
test_M2.rotate(R, center = (0,0,0))
test_M2.translate(t)

o3d.io.write_point_cloud(osp.join(output_dir, "transformed_m2.ply"), test_M2)
