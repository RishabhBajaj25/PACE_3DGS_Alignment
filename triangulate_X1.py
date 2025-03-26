import pandas as pd
import pycolmap
import cv2
import numpy as np
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils

# Load the reconstruction from the specified directory.
reconstruction1 = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_b_20250324_120708_frames_10_fps/sparse/0"
)

image_names1 = {}
for image_id, image in reconstruction1.images.items():
    print(image_id, image.name)
    image_names1[image.name] = image_id - 1  # Adjust to 0-based index

# Static selection of image number (change this as needed)
closest_image_number = 27  # Example: selecting image 27

# Format the image name based on the number (e.g., converts 27 to 'frame_0027.jpg')
closest_image_name = f"frame_{closest_image_number:04d}.jpg"

# Check if the closest image exists in the reconstruction1
if closest_image_name not in image_names1:
    print(f"Image with name {closest_image_name} not found in reconstruction1. Exiting.")
    exit()

# Retrieve the corresponding image object
for image_id, image in reconstruction1.images.items():
    if image.name == closest_image_name:
        close_image = image
        break

# Get the associated camera for the closest image
camera1 = reconstruction1.cameras[close_image.camera.camera_id]

# Load the reconstruction from the specified directory.
reconstruction2 = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/sparse/0"
)

image_names2 = {}
for image_id, image in reconstruction2.images.items():
    print(image_id, image.name)
    image_names2[image.name] = image_id - 1  # Adjust to 0-based index

# Static selection of image number (change this as needed)
query_image_number = 0  # Example: selecting image 27

# Format the image name based on the number (e.g., converts 27 to 'frame_0027.jpg')
query_image_name = f"frame_{query_image_number:04d}.jpg"

# Check if the closest image exists in the reconstruction2
if query_image_name not in image_names2:
    print(f"Image with name {query_image_name} not found in reconstruction2. Exiting.")
    exit()

# Retrieve the corresponding image object
for image_id, image in reconstruction2.images.items():
    if image.name == query_image_name:
        query_image = image
        break

# Get the associated camera for the closest image
camera2 = reconstruction2.cameras[query_image.camera.camera_id]

K1 = np.array([[camera1.focal_length_x, 0, camera1.principal_point_x],
      [0, camera1.focal_length_y, camera1.principal_point_y],
      [0, 0, 1]])

# https://github.com/colmap/colmap/issues/2358
rot1_qvec = close_image.cam_from_world.rotation.quat
rot1 = utils.qvec2rotmat(rot1_qvec)
trans1 = close_image.cam_from_world.translation
trans_vec1 = trans1.T.reshape((-1,1))
hola1 = np.hstack((rot1, trans_vec1))
ext_mat1 = np.vstack((hola1, [0, 0, 0, 1]))
P1 = K1 @ ext_mat1[:3, :]

K2 = np.array([[camera2.focal_length_x, 0, camera2.principal_point_x],
      [0, camera2.focal_length_y, camera2.principal_point_y],
      [0, 0, 1]])

rot2_qvec = query_image.cam_from_world.rotation.quat
rot2 = utils.qvec2rotmat(rot2_qvec)
trans2= query_image.cam_from_world.translation
trans_vec2 = trans2.T.reshape((-1,1))
hola2 = np.hstack((rot2, trans_vec2))
ext_mat2 = np.vstack((hola2, [0, 0, 0, 1]))
P2 = K2 @ ext_mat2[:3, :]

P_list = [P1, P2]

uvs =pd.read_csv("/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images/matched_keypoints.csv")[['Keypoint_X', 'Keypoint_Y']].to_numpy()
for uv in uvs:
    uv = np.append(uv, 1)
    X_now = utils.triangulate_nviews(P_list, [uv, uv])
    print(utils.triangulate_nviews(P_list, [uv, uv]))
# Use the extrinsic matrices from the original script
utils.plot_camera_centers_and_orientations(ext_mat1, ext_mat2)

query_img_path = "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/images/frame_0000.jpg"
