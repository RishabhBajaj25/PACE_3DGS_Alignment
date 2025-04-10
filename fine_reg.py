from utils import *

display_original_pcds = False
# Load two point clouds
base_dir = "/media/rishabh/SSD_1/Data/Blender_Renders/ct_scan_foot"
output_dir = osp.join(base_dir, "project_2_images")

icp_result = np.eye(4)

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
output_path = osp.join(output_dir, "aligned_M2.ply")