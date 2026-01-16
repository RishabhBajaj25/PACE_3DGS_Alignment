import os

import open3d.examples.visualization.draw

import numpy as np
import open3d
import pycolmap
import matplotlib.pyplot as plt
import os.path as osp
import cv2
import pandas as pd
import open3d as o3d
import numpy as np
import os

# base_dir = "/media/rishabh/SSD_1/Data/Blender_Renders/ct_scan_foot"
# output_name = "project_2_images"

base_dir = "/home/pace-ubuntu/datasets/leica/EAST/registration"
output_name = "rough"

output_dir = osp.join(base_dir, output_name)

os.makedirs(output_dir, exist_ok=True)

result_T_m2_m1 = np.eye(4)

from scipy.spatial.transform import Rotation as R
import numpy as np

axis = np.array([0, 0, 1])   # rotate around Z
axis = axis / np.linalg.norm(axis)

theta = np.deg2rad(45)

R_mat = R.from_rotvec(axis * theta).as_matrix()

T = np.eye(4)
T[:3, :3] = R_mat


# test_M2 = o3d.io.read_point_cloud('/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/density_test.ply')
test_M2 = o3d.io.read_point_cloud("/home/pace-ubuntu/datasets/leica/EAST/registration/supersplat_export_50000 - Cloud.ply")
test_M2.rotate(R_mat, center = (0,0,0))

o3d.io.write_point_cloud(osp.join(output_dir, "rot.ply"), test_M2)