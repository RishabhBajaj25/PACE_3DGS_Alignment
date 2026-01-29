import csv
import os.path as osp
import numpy as np
import subprocess



def read_csv_transformations(csv_path, transform_key):
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        transform_matrix = None
        euler_angles = None

        for i, row in enumerate(rows):
            if row and row[0] == transform_key:
                matrix = [list(map(float, rows[i + j])) for j in range(1, 5)]
                transform_matrix = np.array(matrix)
            elif row and row[0] == "euler_angles_zyx":
                euler_angles = list(map(float, rows[i + 1]))
                break

        translations = transform_matrix[:3, 3] if transform_matrix is not None else [0, 0, 0]
        return translations, euler_angles

output_folder_path ='/home/pace-ubuntu/datasets/leica/EAST/registration/rough'

# Read global transformation parameters
global_translations, global_euler_zyx = read_csv_transformations(
    osp.join(output_folder_path, 'global_reg_result.csv'), "result_T_m2_m1"
)

# Read fine transformation parameters
fine_translations, fine_euler_zyx = read_csv_transformations(
    osp.join(output_folder_path, 'fine_reg_result.csv'), "icp_result.transformation"
)

print("Debug")