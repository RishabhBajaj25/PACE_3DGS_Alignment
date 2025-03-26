from utils import *


def load_reconstruction(reconstruction_path):
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    image_names = {image.name: image_id - 1 for image_id, image in reconstruction.images.items()}
    return reconstruction, image_names


def get_camera_and_pose(reconstruction, image_name, frames):
    # image_names = {}
    # for image_id, image in reconstruction.images.items():
    #     print(image_id, image.name)
    #     image_names[image.name] = image_id - 1  # Adjust to 0-based index
    # if image_name not in image_names:
    #     print(f"Image with name {image_name} not found in reconstruction. Exiting.")
    #     exit()

    # image = reconstruction.images[image_name]
    # camera = reconstruction.cameras[image.camera.camera_id]

    # Retrieve the corresponding image object
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
    rot_qvec = np.roll(rot_qvec, shift=1) # Somehow this is needed to make the rotation matrix correct (as per colmap format)

    rot_mat = qvec2rotmat(rot_qvec)

    trans_vec_ = image.cam_from_world.translation.T
    trans_vec = -rot_mat.T @ (trans_vec_)

    rot_mat = rot_mat.T

    ext_mat = np.vstack((np.hstack((rot_mat, trans_vec.reshape(-1,1))), [0, 0, 0, 1]))
    P = K @ ext_mat[:3, :]

    # create axis, plane and pyramid geometries that will be drawn
    cam_model = draw_camera(K, rot_mat, trans_vec, camera.width, camera.height, 1)
    frames.extend(cam_model)

    return K, rot_mat, trans_vec, ext_mat, P, frames



plant_recon, plant_image_names = load_reconstruction("/media/rishabh/SSD_1/Data/plant/20250326_123143_frames_10_fps/sparse/0")

min_track_len = 3
remove_statistical_outlier = True

pcd = open3d.geometry.PointCloud()

xyz = []
rgb = []
for point3D in plant_recon.points3D.values():
    track_len = len(point3D.track.elements)
    if track_len < min_track_len:
        continue
    xyz.append(point3D.xyz)
    rgb.append(point3D.color / 255)

pcd.points = open3d.utility.Vector3dVector(xyz)
pcd.colors = open3d.utility.Vector3dVector(rgb)

# remove obvious outliers
if remove_statistical_outlier:
    [pcd, _] = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

# vis.add_geometry(pcd)
plant_image_names = []
plant_frames = []
for imgs in plant_recon.images.items():
    image_name = imgs[1].name
    plant_image_names.append(image_name)

for img_name in plant_image_names:
    plant_K1_a, plant_rot_mat_a, plant_trans_vec_a, plant_ext_mat1_a, plant_P1_a, plant_frames = get_camera_and_pose(
        plant_recon, img_name, plant_frames)
vis = open3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd)
for i in plant_frames:
    vis.add_geometry(i)

vis.poll_events()
vis.update_renderer()
vis.run()
vis.destroy_window()



# Load first reconstruction
reconstruction1, image_names1 = load_reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_b_20250324_120708_frames_10_fps/sparse/0"
)

# Select and process two images from reconstruction1
close_image_numbers = [27, 30]
close_image_names = [f"frame_{num:04d}.jpg" for num in close_image_numbers]

close_frames = []
close_K1_a, close_rot_mat_a, close_trans_vec_a, close_ext_mat1_a, close_P1_a, close_frames = get_camera_and_pose(reconstruction1, close_image_names[0], close_frames)
close_K1_b, close_rot_mat_b, close_trans_vec_b, close_ext_mat1_b, close_P1_b, close_frames = get_camera_and_pose(reconstruction1, close_image_names[1], close_frames)

vis = open3d.visualization.Visualizer()
vis.create_window()


pcd = open3d.geometry.PointCloud()

xyz = []
rgb = []
for point3D in reconstruction1.points3D.values():
    track_len = len(point3D.track.elements)
    if track_len < min_track_len:
        continue
    xyz.append(point3D.xyz)
    rgb.append(point3D.color / 255)

pcd.points = open3d.utility.Vector3dVector(xyz)
pcd.colors = open3d.utility.Vector3dVector(rgb)

# remove obvious outliers
if remove_statistical_outlier:
    [pcd, _] = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

vis.add_geometry(pcd)

for i in close_frames:
    vis.add_geometry(i)

vis.poll_events()
vis.update_renderer()
vis.run()
vis.destroy_window()

close_dist = np.linalg.norm(close_trans_vec_a-close_trans_vec_b)

# Load second reconstruction
reconstruction2, image_names2 = load_reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/sparse/0"
)

# Select and process two images from reconstruction2
query_image_numbers = [0, 7]
query_image_names = [f"frame_{num:04d}.jpg" for num in query_image_numbers]

query_frames = []
query_K2_a, query_rot_mat_a, query_trans_vec_a, query_ext_mat2_a, query_P2_a, query_frames = get_camera_and_pose(reconstruction2, query_image_names[0], query_frames)
query_K2_b, query_rot_mat_b, query_trans_vec_b, query_ext_mat2_b, query_P2_b, query_frames = get_camera_and_pose(reconstruction2, query_image_names[1], query_frames)

query_dist = np.linalg.norm(query_trans_vec_a-query_trans_vec_b)

pcd = open3d.geometry.PointCloud()
xyz = []
rgb = []
for point3D in reconstruction2.points3D.values():
    track_len = len(point3D.track.elements)
    if track_len < min_track_len:
        continue
    xyz.append(point3D.xyz)
    rgb.append(point3D.color / 255)

pcd.points = open3d.utility.Vector3dVector(xyz)
pcd.colors = open3d.utility.Vector3dVector(rgb)

# remove obvious outliers
if remove_statistical_outlier:
    [pcd, _] = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

vis = open3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

for i in close_frames:
    vis.add_geometry(i)

vis.poll_events()
vis.update_renderer()
vis.run()
vis.destroy_window()

scale = query_dist/close_dist
print("Hello")