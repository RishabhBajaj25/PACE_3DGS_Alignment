from utils import *

min_track_len = 3
remove_statistical_outlier = True

reconstruction1_path = "/media/rishabh/SSD_1/Data/Table_vid_reg/sub_1_20250327_143742_frames_5_fps/1/sparse/0"
reconstruction2_path = "/media/rishabh/SSD_1/Data/Table_vid_reg/sub_1_20250327_143742_frames_5_fps/2/sparse/0"

dense1_path = osp.join(osp.dirname(osp.dirname(reconstruction1_path)), 'fused.ply')
dense2_path = osp.join(osp.dirname(osp.dirname(reconstruction2_path)), 'fused.ply')

output_path = osp.join(osp.dirname((osp.dirname(osp.dirname(osp.dirname(reconstruction1_path))))),'project_2_images')

# Load first reconstruction
reconstruction1, image_names1 = load_reconstruction(reconstruction1_path)

# Select and process two images from reconstruction1
close_image_numbers = [27, 30]
close_image_names = [f"frame_{num:04d}.jpg" for num in close_image_numbers]

close_frames = []
close_K1_a, close_rot_mat_a, close_trans_vec_a, close_ext_mat1_a, close_P1_a, close_frames = get_camera_and_pose(reconstruction1, close_image_names[0], close_frames)
close_K1_b, close_rot_mat_b, close_trans_vec_b, close_ext_mat1_b, close_P1_b, close_frames = get_camera_and_pose(reconstruction1, close_image_names[1], close_frames)

# Generate point cloud for the first reconstruction
pcd1 = process_point_cloud(reconstruction1, min_track_len, remove_statistical_outlier)

# Visualize the point cloud and frames for the first reconstruction
visualize_point_cloud_and_frames(pcd1, close_frames)

close_dist = np.linalg.norm(close_trans_vec_a - close_trans_vec_b)

# Load second reconstruction
reconstruction2, image_names2 = load_reconstruction(reconstruction2_path)

# Select and process two images from reconstruction2
query_image_numbers = [27, 30]
query_image_names = [f"frame_{num:04d}.jpg" for num in query_image_numbers]

query_frames = []
query_K2_a, query_rot_mat_a, query_trans_vec_a, query_ext_mat2_a, query_P2_a, query_frames = get_camera_and_pose(reconstruction2, query_image_names[0], query_frames)
query_K2_b, query_rot_mat_b, query_trans_vec_b, query_ext_mat2_b, query_P2_b, query_frames = get_camera_and_pose(reconstruction2, query_image_names[1], query_frames)

# Generate point cloud for the second reconstruction
pcd2 = process_point_cloud(reconstruction2, min_track_len, remove_statistical_outlier)

# Visualize the point cloud and frames for the second reconstruction
visualize_point_cloud_and_frames(pcd2, query_frames)

query_dist = np.linalg.norm(query_trans_vec_a - query_trans_vec_b)

# Calculate scale
scale = close_dist / query_dist
print("Scale:", scale)

dense2_pcd = open3d.io.read_point_cloud(dense2_path)
dense2_pcd.scale(scale, center = [0,0,0])
open3d.io.write_point_cloud(osp.join(output_path, 'scaled_M2.ply'), dense2_pcd)