import pycolmap
import cv2
import numpy as np
import os
import csv

# Load the reconstruction from the specified directory.
reconstruction = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_b_20250324_120708_frames_10_fps/sparse/0"
)

# Write the reconstruction in text format for inspection.
reconstruction.write_text(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_b_20250324_120708_frames_10_fps/sparse/0"
)

# Print all available images and their corresponding IDs in the reconstruction.
print("\nAvailable Image IDs and Names:")
image_names = {}
for image_id, image in reconstruction.images.items():
    print(image_id, image.name)
    image_names[image.name] = image_id - 1  # Adjust to 0-based index

# Static selection of image number (change this as needed)
selected_image_number = 27  # Example: selecting image 27

# Format the image name based on the number (e.g., converts 27 to 'frame_0027.jpg')
selected_image_name = f"frame_{selected_image_number:04d}.jpg"

# Check if the selected image exists in the reconstruction
if selected_image_name not in image_names:
    print(f"Image with name {selected_image_name} not found in reconstruction. Exiting.")
    exit()

# Retrieve the corresponding image object
for image_id, image in reconstruction.images.items():
    if image.name == selected_image_name:
        close_image = image
        break

# Get the associated camera for the selected image
camera = reconstruction.cameras[close_image.camera.camera_id]

# Define the input image path
image_path = f"/media/rishabh/SSD_1/Data/lab_videos_reg/1_20250324_120708_frames_10_fps/images/{close_image.name}"

# Define output directory for projected images
output_dir = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{close_image.name.split('.')[0]}_uv.png"

# Load the image using OpenCV
img = cv2.imread(image_path)
if img is None:
    print(f"Error loading image: {image_path}")
    exit()

# List to store UV coordinates and corresponding 3D points
uvs_and_3d_points = []

# Process 3D points and project them onto the selected image
for point3D_id, point3D in reconstruction.points3D.items():
    # Project the 3D point to the image
    uv = camera.img_from_cam(close_image.cam_from_world * point3D.xyz)
    uvs_and_3d_points.append((point3D.xyz, uv[0]))  # Store both 3D point and UV coordinates

    # Convert UV coordinates to integer pixel values
    u, v = int(uv[0][0]), int(uv[0][1])

    # Draw a red circle at the projected UV coordinates
    cv2.circle(img, (u, v), 3, (0, 0, 255), -1)  # Red dots

# Save the image with UV points overlaid
cv2.imwrite(output_path, img)
print(f"Image with UV points saved at: {output_path}")

# Save UV coordinates and corresponding 3D points to a CSV file
uv_file_path = f"{output_dir}/{close_image.name.split('.')[0]}_uv_coordinates_3d.csv"
with open(uv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["X_3D", "Y_3D", "Z_3D", "UV_X", "UV_Y"])  # CSV header row
    for (point3D, uv) in uvs_and_3d_points:
        writer.writerow([point3D[0], point3D[1], point3D[2], uv[0], uv[1]])

print(f"UV coordinates and 3D points saved at: {uv_file_path}")
