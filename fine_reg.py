from utils import *

display_original_pcds = False
# Load two point clouds
output_dir = "/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images"
scene_dir = osp.join(osp.dirname(output_dir), "sub_1_20250327_143742_frames_5_fps")

scaled_transformed_M2 = o3d.io.read_point_cloud(osp.join(output_dir, "scaled_M2_transformed.ply")) # source
dense_M1 = o3d.io.read_point_cloud(osp.join( scene_dir, '1','fused.ply')) # target

# source = o3d.io.read_point_cloud("/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images/scaled_transform_2.ply") # Transformation applied to the source point cloud
# target = o3d.io.read_point_cloud("/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images/1_fused.ply")
# Visualize raw point clouds before filtering
# o3d.visualization.draw_geometries([source], window_name="Raw Source Point Cloud")
# o3d.visualization.draw_geometries([target], window_name="Raw Target Point Cloud")

# Remove statistical outliers
source_clean, source_inliers = scaled_transformed_M2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
target_clean, target_inliers = dense_M1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

if display_original_pcds:
    o3d.visualization.draw_geometries([source_clean], window_name="Filtered Source Point Cloud")
    o3d.visualization.draw_geometries([target_clean], window_name="Filtered Target Point Cloud")

# Initial alignment using a rough transformation (identity if unknown)
initial_transformation = np.identity(4)
# draw_registration_result(source_clean, target_clean, initial_transformation)

# Apply ICP algorithm
threshold = 0.65 # Distance threshold for point correspondence
icp_result = o3d.pipelines.registration.registration_icp(
    source_clean, target_clean, threshold, initial_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)

print(icp_result)
# Display the final alignment
print("ICP Transformation Matrix:")
print(icp_result.transformation)
draw_registration_result(source_clean, target_clean, icp_result.transformation)

rot = R.from_matrix(icp_result.transformation[:3,:3])
euler_angles = rot.as_euler('zyx', degrees=False)

# Open the CSV file for writing
with open(osp.join(output_dir, "fine_reg_result.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write matrix result_T_m1_m2
    writer.writerow(["icp_result.transformation"])
    for row in icp_result.transformation:
        writer.writerow(row)

    writer.writerow([])  # Blank line separator

    # Write Euler angles
    writer.writerow(["euler_angles_zyx"])
    writer.writerow(euler_angles)

# Save the aligned point cloud
output_path = osp.join(output_dir, "aligned_M2.ply")
o3d.io.write_point_cloud(output_path, source_clean.transform(icp_result.transformation))