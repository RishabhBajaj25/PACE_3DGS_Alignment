import pycolmap
import cv2
import numpy as np
import pandas as pd
import os.path as osp
# Function to save matrices in a human-readable format
def save_matrices_to_txt(filename, matrices):
    with open(filename, "w") as f:
        for name, matrix in matrices.items():
            f.write(f"{name}:\n")
            np.savetxt(f, matrix, fmt="%.6f")
            f.write("\n" + "-" * 40 + "\n")


# https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L53
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

output_dir = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images"


# Load the reconstruction from the specified directory.
reconstruction2 = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/sparse/0"
)

image_names2 = {}
for image_id, image in reconstruction2.images.items():
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

K2 = np.array([[camera2.focal_length_x, 0, camera2.principal_point_x],
      [0, camera2.focal_length_y, camera2.principal_point_y],
      [0, 0, 1]])

output_csv_path = osp.join(output_dir, "matched_data.csv")
matched_data = pd.read_csv(output_csv_path)

X1 = np.array(matched_data[['X_3D', 'Y_3D', 'Z_3D']])
m2 = np.array(matched_data[['Query_KP_x', 'Query_KP_y']])

# _, rvecs, tvecs, inliers  = cv2.solvePnPRansac(objp, corners2, mtx, dist)
_, rvecs, tvecs, inliers = cv2.solvePnPRansac(X1, m2, K2, None, None)
rot_matrix = cv2.Rodrigues(rvecs)[0]
t = rot_matrix.dot(tvecs)

transformation_matrix = np.hstack((rot_matrix, t))
transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

rot2_qvec = query_image.cam_from_world.rotation.quat
rot2_qvec = np.roll(rot2_qvec,
                   shift=1)  # Somehow this is needed to make the rotation matrix correct (as per colmap format)

rot2 = qvec2rotmat(rot2_qvec)
trans2= query_image.cam_from_world.translation
trans_vec2 = trans2.T.reshape((-1,1))
hola2 = np.hstack((rot2, trans_vec2))
ext_mat2 = np.vstack((hola2, [0, 0, 0, 1]))
P2 = K2 @ ext_mat2[:3, :]

result_T_m1_m2 = transformation_matrix @ np.linalg.inv(ext_mat2)

result_inverse = np.linalg.inv(result_T_m1_m2)

result_b = transformation_matrix @ ext_mat2
result_b_inverse = np.linalg.inv(result_b)


# Save matrices
save_matrices_to_txt("/media/rishabh/SSD_1/Data/lab_videos_reg/archive_first_trial_results/results/saved_matrices.txt", {
    "result_T_m1_m2": result_T_m1_m2,
    "result_inverse": result_inverse,
    "result_b": result_b,
    "result_b_inverse": result_b_inverse
})

print("Matrices saved in 'saved_matrices.txt' successfully!")

print("Transformation matrix found!")