import pandas as pd
import pycolmap
import cv2
import numpy as np
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d
import os.path as osp

# https://gist.github.com/enjoylife/01356c4f553411aeaf7dca54eace0706#file-triangulation-py-L9
def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]

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

def plot_camera_centers_and_orientations(ext_mat1, ext_mat2):
    """
    Plot camera centers and their orientation axes in 3D space.

    Parameters:
    ext_mat1 (np.array): Extrinsic matrix for camera 1
    ext_mat2 (np.array): Extrinsic matrix for camera 2
    """
    # Extract camera centers (world coordinates of camera origin)
    # Camera center is -R^T * t, where R is rotation and t is translation
    R1 = ext_mat1[:3, :3]
    t1 = ext_mat1[:3, 3]
    camera1_center = -R1.T @ t1

    R2 = ext_mat2[:3, :3]
    t2 = ext_mat2[:3, 3]
    camera2_center = -R2.T @ t2

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera centers
    ax.scatter(camera1_center[0], camera1_center[1], camera1_center[2],
               color='red', s=100, label='Camera 1 Center')
    ax.scatter(camera2_center[0], camera2_center[1], camera2_center[2],
               color='blue', s=100, label='Camera 2 Center')

    # Plot camera orientation axes
    axis_length = 0.5  # Length of orientation axes

    # Camera 1 axes
    for i, color in enumerate(['r', 'g', 'b']):
        axis = R1[:, i] * axis_length
        ax.quiver(camera1_center[0], camera1_center[1], camera1_center[2],
                  axis[0], axis[1], axis[2],
                  color=color, alpha=0.6,
                  label=f'Camera 1 Axis {i + 1}')

    # Camera 2 axes
    for i, color in enumerate(['r', 'g', 'b']):
        axis = R2[:, i] * axis_length
        ax.quiver(camera2_center[0], camera2_center[1], camera2_center[2],
                  axis[0], axis[1], axis[2],
                  color=color, alpha=0.6,
                  label=f'Camera 2 Axis {i + 1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Centers and Orientations')
    ax.legend()

    plt.tight_layout()
    plt.show()

def draw_camera(K, R, t, w, h, scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    print((points_in_world[0]).shape)

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]

def save_matrices_to_txt(filename, matrices):
    with open(filename, "w") as f:
        for name, matrix in matrices.items():
            f.write(f"{name}:\n")
            np.savetxt(f, matrix, fmt="%.6f")
            f.write("\n" + "-" * 40 + "\n")

def load_reconstruction(reconstruction_path):
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    image_names = {image.name: image_id - 1 for image_id, image in reconstruction.images.items()}
    return reconstruction, image_names

def get_camera_and_pose(reconstruction, image_name, frames):
    for image_id, image in reconstruction.images.items():
        if image.name == image_name:
            image_now = image
            break

    camera = reconstruction.cameras[image_now.camera.camera_id]

    K = np.array([
        [camera.focal_length_x, 0, camera.principal_point_x],
        [0, camera.focal_length_y, camera.principal_point_y],
        [0, 0, 1]
    ])

    rot_qvec = image.cam_from_world.rotation.quat
    rot_qvec = np.roll(rot_qvec, shift=1)  # Adjust for COLMAP format

    rot_mat = qvec2rotmat(rot_qvec)

    trans_vec_ = image.cam_from_world.translation.T
    trans_vec = -rot_mat.T @ (trans_vec_)

    rot_mat = rot_mat.T

    ext_mat = np.vstack((np.hstack((rot_mat, trans_vec.reshape(-1, 1))), [0, 0, 0, 1]))
    P = K @ ext_mat[:3, :]

    # Draw the camera model (axis, plane, pyramid)
    cam_model = draw_camera(K, rot_mat, trans_vec, camera.width, camera.height, 1)
    frames.extend(cam_model)

    return K, rot_mat, trans_vec, ext_mat, P, frames

def process_point_cloud(reconstruction, min_track_len, remove_statistical_outlier=True):
    pcd = open3d.geometry.PointCloud()

    xyz = []
    rgb = []
    for point3D in reconstruction.points3D.values():
        track_len = len(point3D.track.elements)
        if track_len < min_track_len:
            continue
        xyz.append(point3D.xyz)
        rgb.append(point3D.color / 255)

    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(rgb)

    # Remove obvious outliers if enabled
    if remove_statistical_outlier:
        [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def visualize_point_cloud_and_frames(pcd, frames):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for frame in frames:
        vis.add_geometry(frame)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
