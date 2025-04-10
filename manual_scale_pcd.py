import numpy as np

from utils import *

min_track_len = 3
remove_statistical_outlier = True

reconstruction1_path = "/media/rishabh/SSD_1/Data/Blender_Renders/ct_scan_foot/combined/sparse/0"
# reconstruction2_path = "/media/rishabh/SSD_1/Data/Table_vid_reg/sub_1_20250327_143742_frames_5_fps/2/sparse/0"

dense1_path = osp.join(osp.dirname(osp.dirname(reconstruction1_path)), 'fused.ply')
dense2_path = '/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/density_test.ply'

output_path = osp.join(osp.dirname((osp.dirname(osp.dirname(osp.dirname(reconstruction1_path))))), 'ct_scan_foot', 'project_2_images')

# # Scale obtained from manually aligning the 2pcd using point picking in CloudCompare, get scale but not scale the pcd
scale = 0.480228
print("Scale:", scale)
np.savetxt(osp.join(output_path, 'scale.txt'), [scale])