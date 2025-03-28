import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target_temp.paint_uniform_color([0, 1, 0])  # Green
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Load two point clouds
source = o3d.io.read_point_cloud("/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images/scaled_transform_2.ply") # Transformation applied to the source point cloud
target = o3d.io.read_point_cloud("/media/rishabh/SSD_1/Data/Table_vid_reg/project_2_images/1_fused.ply")

# Visualize raw point clouds before filtering
# o3d.visualization.draw_geometries([source], window_name="Raw Source Point Cloud")
# o3d.visualization.draw_geometries([target], window_name="Raw Target Point Cloud")

# Remove statistical outliers
source_clean, source_inliers = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
target_clean, target_inliers = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Visualize filtered and downsampled point clouds
o3d.visualization.draw_geometries([source_clean], window_name="Filtered & Downsampled Source Point Cloud")
o3d.visualization.draw_geometries([target_clean], window_name="Filtered & Downsampled Target Point Cloud")

# Initial alignment using a rough transformation (identity if unknown)
initial_transformation = np.identity(4)
# draw_registration_result(source_clean, target_clean, initial_transformation)

# Apply ICP algorithm
threshold = 0.6  # Distance threshold for point correspondence
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