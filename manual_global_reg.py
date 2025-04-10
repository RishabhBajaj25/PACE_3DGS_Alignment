import open3d.examples.visualization.draw

from utils import *

base_dir = "/media/rishabh/SSD_1/Data/Blender_Renders/ct_scan_foot"
output_dir = osp.join(base_dir, "project_2_images")

# scaled_M2_path = osp.join(output_dir, "scaled_M2.ply")
# scaled_M2 = o3d.io.read_point_cloud(scaled_M2_path)
#
# # # Load the reconstruction from the specified directory.
# # reconstruction2 = pycolmap.Reconstruction(
# #     # "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/sparse/0"
# #     # '/media/rishabh/SSD_1/Data/Table_vid_reg/sub_1_20250327_143742_frames_5_fps/2/sparse/0'
# #     osp.join(base_dir, "sub_1_20250327_143742_frames_5_fps/2/sparse/0")
# # )
# #
# image_names2 = {}
# for image_id, image in reconstruction2.images.items():
#     image_names2[image.name] = image_id - 1  # Adjust to 0-based index
#
# # Static selection of image number (change this as needed)
# query_image_number = 25  # Example: selecting image 27
#
# # Format the image name based on the number (e.g., converts 27 to 'frame_0027.jpg')
# query_image_name = f"frame_{query_image_number:04d}.jpg"
# # Check if the closest image exists in the reconstruction2
# if query_image_name not in image_names2:
#     print(f"Image with name {query_image_name} not found in reconstruction2. Exiting.")
#     exit()
#
# # Retrieve the corresponding image object
# for image_id, image in reconstruction2.images.items():
#     if image.name == query_image_name:
#         query_image = image
#         break
# # Get the associated camera for the closest image
# camera2 = reconstruction2.cameras[query_image.camera.camera_id]
#
# K2 = np.array([[camera2.focal_length_x, 0, camera2.principal_point_x],
#       [0, camera2.focal_length_y, camera2.principal_point_y],
#       [0, 0, 1]])
#
# output_csv_path = osp.join(output_dir, "matched_data.csv")
# matched_data = pd.read_csv(output_csv_path)
#
# X1 = np.array(matched_data[['X_3D', 'Y_3D', 'Z_3D']])
# m2 = np.array(matched_data[['Query_KP_x', 'Query_KP_y']])
#
# # _, rvecs, tvecs, inliers  = cv2.solvePnPRansac(objp, corners2, mtx, dist)
# _, rvecs, tvecs, inliers = cv2.solvePnPRansac(X1, m2, K2, None, None)
# rot_matrix = cv2.Rodrigues(rvecs)[0]
# t = rot_matrix.dot(tvecs)

# transformation_matrix = np.hstack((rot_matrix, t))
# transformation_matrix = np.array([[ 0.03281417, -0.47197083, -0.08237366, -0.43157973],
#        [-0.02093014, -0.08389998,  0.47237842,  0.16128158],
#        [-0.4786479 , -0.02868767, -0.0263032 , -0.22229815],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
#
#
# rot2_qvec = query_image.cam_from_world.rotation.quat
# rot2_qvec = np.roll(rot2_qvec,
#                    shift=1)  # Somehow this is needed to make the rotation matrix correct (as per colmap format)
#
# rot2 = qvec2rotmat(rot2_qvec)
# trans2= query_image.cam_from_world.translation
# trans_vec2 = trans2.T.reshape((-1,1))
# hola2 = np.hstack((rot2, trans_vec2))
# ext_mat2 = np.vstack((hola2, [0, 0, 0, 1]))
# P2 = K2 @ ext_mat2[:3, :]
#
# result_T_m1_m2 = transformation_matrix @ np.linalg.inv(ext_mat2)

# obtained from manually aligning the 2pcd using point picking in CloudCompare
result_T_m2_m1 = np.array([[ 0.03281417, -0.47197083, -0.08237366, -0.43157973],
       [-0.02093014, -0.08389998,  0.47237842,  0.16128158],
       [-0.4786479 , -0.02868767, -0.0263032 , -0.22229815],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]) # THIS TF matrix from cloud compare contains scaling factor too (0.480228)


rot = R.from_matrix(result_T_m2_m1[:3,:3])
# rot_transformed = np.eye(4)
# rot_transformed[:3,:3] = rot.as_matrix()
# scaled_M2.transform(rot_transformed)
# o3d.io.write_point_cloud(osp.join(output_dir, "rot_scaled_M2.ply"), scaled_M2)

euler_angles = rot.as_euler('zyx', degrees=False)

test_M2 = o3d.io.read_point_cloud('/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/density_test.ply')
test_M2.transform(result_T_m2_m1)
o3d.io.write_point_cloud(osp.join(output_dir, "scaled_M2_transformed.ply"), test_M2)

# The TF matrix from cloud compare contains scaling factor too (yes, but still need to store it (to apply to gaussians later and not point cloud))
# s =0.480228
scale = np.loadtxt(osp.join(output_dir, "scale.txt"))

# Open the CSV file for writing
with open(osp.join(output_dir, "global_reg_result.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the scale value as a header
    writer.writerow(["scale"])

    # Write the scale value (as a single row)
    writer.writerow([scale])

    writer.writerow([])  # Blank line separator


    writer.writerow([])  # Blank line separator

    # Write matrix result_T_m2_m1
    writer.writerow(["result_T_m2_m1"])
    for row in result_T_m2_m1:
        writer.writerow(row)

    writer.writerow([])  # Blank line separator

    # Write Euler angles
    writer.writerow(["euler_angles_zyx"])
    writer.writerow(euler_angles)

print(f"Matrices and Euler angles saved successfully!")
print("Transformation matrix obtained from manually aligning the 2pcd using point picking in CloudCompare!")