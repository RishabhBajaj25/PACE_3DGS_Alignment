from utils import *

display_original_pcds = False
# Load two point clouds

base_dir = "/home/pace-ubuntu/datasets/leica/EAST/registration"
output_name = "rough"

source = o3d.io.read_point_cloud(osp.join(base_dir, output_name, "transformed_m2.ply"))
target = o3d.io.read_point_cloud(osp.join(base_dir, "BB_U7_579 EAST_JUNE 2024_ 1 - Cloud.ply"))

trans_init = np.eye(4)
threshold = 0.02

# draw_registration_result(source, target, trans_init)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)

output_dir = osp.join(base_dir, output_name)
icp_result = reg_p2p.transformation

rot = R.from_matrix(icp_result[:3,:3])
euler_angles = rot.as_euler('zyx', degrees=False)

# Open the CSV file for writing
with open(osp.join(output_dir, "fine_reg_result.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write matrix result_T_m1_m2
    writer.writerow(["icp_result.transformation"])
    for row in icp_result:
        writer.writerow(row)

    writer.writerow([])  # Blank line separator

    # Write Euler angles
    writer.writerow(["euler_angles_zyx"])
    writer.writerow(euler_angles)

# Save the aligned point cloud
output_path = osp.join(output_dir, "fine_aligned_m2.ply")
o3d.io.write_point_cloud(output_path, source.transform(icp_result))