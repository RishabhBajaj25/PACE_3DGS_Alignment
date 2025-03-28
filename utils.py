import numpy as np
import open3d
import pycolmap
import matplotlib.pyplot as plt
import os.path as osp
import cv2
import pandas as pd
import open3d as o3d
import numpy as np
import copy

def triangulate_nviews(P, ip):
    """
    Triangulate a 3D point visible in multiple camera views using Direct Linear Transformation (DLT).

    This function estimates the 3D coordinates of a point by solving a linear least squares problem
    using the projection matrices and corresponding 2D image points from multiple cameras.

    Parameters:
    -----------
    P : list of numpy.ndarray
        List of camera projection matrices (3x4). Each matrix maps 3D world coordinates to 2D image coordinates.
    ip : list or numpy.ndarray
        List of homogeneous image points. Each point is a [x, y, 1] coordinate.
        Must have the same length as P.

    Returns:
    --------
    numpy.ndarray
        Homogeneous 3D point coordinates [X, Y, Z, W], normalized by the last coordinate.

    Raises:
    -------
    ValueError
        If the number of projection matrices does not match the number of image points.

    Notes:
    ------
    - Implementation based on a triangulation method for multiple views.
    - Source: https://gist.github.com/enjoylife/01356c4f553411aeaf7dca54eace0706#file-triangulation-py-L9
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras must be equal.')

    n = len(P)
    M = np.zeros([3 * n, 4 + n])

    for i, (x, p) in enumerate(zip(ip, P)):
        M[3 * i:3 * i + 3, :4] = p
        M[3 * i:3 * i + 3, 4 + i] = -x

    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def qvec2rotmat(qvec):
    """
    Convert a quaternion vector to a 3x3 rotation matrix.

    Parameters:
    -----------
    qvec : numpy.ndarray
        Quaternion vector [w, x, y, z] where w is the scalar component.

    Returns:
    --------
    numpy.ndarray
        3x3 rotation matrix corresponding to the input quaternion.

    Notes:
    ------
    - Adapted from COLMAP's quaternion to rotation matrix conversion.
    - Source: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L53
    """
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ])


def plot_camera_centers_and_orientations(ext_mat1, ext_mat2):
    """
    Visualize camera centers and their orientation axes in a 3D plot.

    Parameters:
    -----------
    ext_mat1 : numpy.ndarray
        4x4 extrinsic matrix for the first camera.
    ext_mat2 : numpy.ndarray
        4x4 extrinsic matrix for the second camera.

    Notes:
    ------
    - Creates a 3D matplotlib plot showing:
      * Camera centers as scatter points
      * Camera orientation axes as colored quiver plots
    - Helps visualize relative camera positions and orientations
    """
    # Extract camera centers
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
    """
    Create Open3D geometries to visualize a camera's pose and intrinsics.

    Parameters:
    -----------
    K : numpy.ndarray
        3x3 camera intrinsic matrix (calibration matrix)
    R : numpy.ndarray
        3x3 rotation matrix representing camera orientation
    t : numpy.ndarray
        Translation vector representing camera position
    w : int
        Image width in pixels
    h : int
        Image height in pixels
    scale : float, optional
        Scale factor for camera visualization (default: 1)
    color : list, optional
        RGB color for camera visualization (default: [0.8, 0.2, 0.8])

    Returns:
    --------
    list
        List of Open3D geometries:
        - Coordinate axis
        - Image plane
        - Camera pyramid lines

    Notes:
    ------
    - Creates a visual representation of a camera's coordinate system
    - Useful for understanding camera placement in 3D reconstruction
    """
    # Intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation matrix
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # Coordinate axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # Points in pixel coordinates
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # Convert pixel coordinates to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # Image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # Camera pyramid lines
    points_in_world = [(R @ p + t) for p in points]

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for _ in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    return [axis, plane, line_set]


def save_matrices_to_txt(filename, matrices):
    """
    Save a dictionary of matrices to a text file.

    Parameters:
    -----------
    filename : str
        Path to the output text file
    matrices : dict
        Dictionary of matrices to save, with matrix names as keys

    Notes:
    ------
    - Writes each matrix with its name, separated by dashes
    - Uses numpy's savetxt for formatting
    """
    with open(filename, "w") as f:
        for name, matrix in matrices.items():
            f.write(f"{name}:\n")
            np.savetxt(f, matrix, fmt="%.6f")
            f.write("\n" + "-" * 40 + "\n")


def load_reconstruction(reconstruction_path):
    """
    Load a COLMAP reconstruction from a specified path.

    Parameters:
    -----------
    reconstruction_path : str
        Path to the COLMAP reconstruction directory

    Returns:
    --------
    tuple
        A tuple containing:
        - pycolmap.Reconstruction object
        - Dictionary mapping image names to their 0-based index
    """
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    image_names = {image.name: image_id - 1 for image_id, image in reconstruction.images.items()}
    return reconstruction, image_names


def get_camera_and_pose(reconstruction, image_name, frames):
    """
    Extract camera intrinsics, extrinsics, and create visualization frames for a specific image.

    Parameters:
    -----------
    reconstruction : pycolmap.Reconstruction
        COLMAP reconstruction object
    image_name : str
        Name of the image to extract camera parameters for
    frames : list
        List to extend with camera visualization geometries

    Returns:
    --------
    tuple
        A tuple containing:
        - K (3x3 intrinsic matrix)
        - Rotation matrix
        - Translation vector
        - 4x4 extrinsic matrix
        - Projection matrix
        - Updated frames list with camera visualization geometries

    Notes:
    ------
    - Handles quaternion to rotation matrix conversion
    - Adjusts translation to world coordinate system
    - Creates Open3D camera visualization geometries
    """
    # Find the specific image in the reconstruction
    for image_id, image in reconstruction.images.items():
        if image.name == image_name:
            image_now = image
            break

    # Get camera parameters
    camera = reconstruction.cameras[image_now.camera.camera_id]

    # Intrinsic matrix
    K = np.array([
        [camera.focal_length_x, 0, camera.principal_point_x],
        [0, camera.focal_length_y, camera.principal_point_y],
        [0, 0, 1]
    ])

    # Convert quaternion to rotation matrix
    rot_qvec = image.cam_from_world.rotation.quat
    rot_qvec = np.roll(rot_qvec, shift=1)  # Adjust for COLMAP format

    rot_mat = qvec2rotmat(rot_qvec)

    # Calculate camera center and translation
    trans_vec_ = image.cam_from_world.translation.T
    trans_vec = -rot_mat.T @ (trans_vec_)

    rot_mat = rot_mat.T

    # Construct extrinsic and projection matrices
    ext_mat = np.vstack((np.hstack((rot_mat, trans_vec.reshape(-1, 1))), [0, 0, 0, 1]))
    P = K @ ext_mat[:3, :]

    # Draw the camera model (axis, plane, pyramid)
    cam_model = draw_camera(K, rot_mat, trans_vec, camera.width, camera.height, 1)
    frames.extend(cam_model)

    return K, rot_mat, trans_vec, ext_mat, P, frames


def process_point_cloud(reconstruction, min_track_len, remove_statistical_outlier=True):
    """
    Process and filter point cloud from a COLMAP reconstruction.

    Parameters:
    -----------
    reconstruction : pycolmap.Reconstruction
        COLMAP reconstruction object
    min_track_len : int
        Minimum number of image tracks a 3D point must appear in to be included
    remove_statistical_outlier : bool, optional
        Whether to remove statistical outliers (default: True)

    Returns:
    --------
    open3d.geometry.PointCloud
        Processed point cloud with filtered points

    Notes:
    ------
    - Filters points based on track length
    - Optionally removes statistical outliers using Open3D's method
    """
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
    """
    Visualize a point cloud with camera frames using Open3D.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Point cloud to visualize
    frames : list
        List of Open3D geometries representing camera frames

    Notes:
    ------
    - Creates an interactive Open3D visualization
    - Adds point cloud and camera frames to the scene
    - Blocks execution until visualization window is closed
    """
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for frame in frames:
        vis.add_geometry(frame)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target_temp.paint_uniform_color([0, 1, 0])  # Green
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])