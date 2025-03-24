import pycolmap
import cv2
import numpy as np
import os
import csv

# Load the reconstruction
reconstruction = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_20250324_120708_frames_10_fps/dense/0/sparse")

for image_id, image in reconstruction.images.items():
    print(image_id, image)

# Print summary
print(reconstruction.summary())

# Print available image IDs and names
print("\nAvailable Image IDs and Names:")
image_names = {}
for image_id, image in reconstruction.images.items():
    print(image_id, image.name)
    image_names[image.name] = image_id - 1  # Subtract 1 to make it 0-based

# Static numeric input (e.g., image number 26)
selected_image_number = 26  # You can change this number

# Format the image name based on the number
selected_image_name = f"frame_{selected_image_number:04d}.jpg"  # Formats as "frame_0026.jpg"

# Check if the image name exists in the reconstruction
if selected_image_name not in image_names:
    print(f"Image with name {selected_image_name} not found in reconstruction. Exiting.")
    exit()

for image_id, image in reconstruction.images.items():
    if image.name == selected_image_name :
        query_image = image
        break

# Get the corresponding image and camera
# image = query_image
# camera = query_camera_id
#
# image = reconstruction.images[selected_image_id]
camera = reconstruction.cameras[query_image.camera.camera_id]

# Load the image
image_path = f"/media/rishabh/SSD_1/Data/lab_videos_reg/1_20250324_120708_frames_10_fps/images/{query_image.name}"

output_dir = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images"
os.makedirs(output_dir, exist_ok=True)
output_path = f"/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images/{query_image.name.split('.')[0]}_uv.png"

img = cv2.imread(image_path)
if img is None:
    print(f"Error loading image: {image_path}")
    exit()

uvs = []
# Process 3D points for the selected image
for point3D_id, point3D in reconstruction.points3D.items():
    uv = camera.img_from_cam(query_image.cam_from_world * point3D.xyz)
    uvs.append(uv[0])
    # Convert UV to integer pixel coordinates
    u, v = int(uv[0][0]), int(uv[0][1])

    # Draw a circle at the UV coordinates
    cv2.circle(img, (u, v), 3, (0, 0, 255), -1)  # Red dots

# Save the image with UV points
cv2.imwrite(output_path, img)
print(f"Image with UV points saved at: {output_path}")

# Save UV coordinates to a CSV file
uv_file_path = f"/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images/{query_image.name.split('.')[0]}_uv_coordinates.csv"
with open(uv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["UV_X", "UV_Y"])  # Header row
    for (u, v) in uvs:
        writer.writerow([u, v])

print(f"UV coordinates saved at: {uv_file_path}")