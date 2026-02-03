import csv
import cv2
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
# OLD FILE DONOT RUN

# File detects the points in query image that match the projected points in closest image.
# This is to be fed to the PnP problem.

output_dir = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images"
# Load the query image
query_img_path = "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/images/frame_0000.jpg"
img = cv2.imread(query_img_path)

if img is None:
    print(f"Error loading image: {query_img_path}")
    exit()

# Initialize ORB feature detector (you can use SIFT or SURF if available)
orb = cv2.ORB_create()

# Detect features in the image
keypoints, descriptors = orb.detectAndCompute(img, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
cv2.imwrite(osp.join(output_dir, "query_img_keypoints.jpg"), img_with_keypoints)

# cv2.imshow('Detected Features', img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Static numeric input for I_close (e.g., image number 26)
selected_image_number = 27

# Format the image name based on the number
selected_image_name = f"frame_{selected_image_number:04d}.jpg"

csv_path = osp.join(output_dir, f"{selected_image_name.split('.')[0]}_uv_coordinates.csv")
# Load UV coordinates
uv_coordinates = pd.read_csv(csv_path)

matched_keypoints = []

# Check if the UV points match any detected keypoints with a loading bar
for (u, v) in tqdm(zip(np.array(uv_coordinates.UV_X), np.array(uv_coordinates.UV_Y)),
                   total=len(uv_coordinates.UV_X),
                   desc="Matching UV points to keypoints"):
    for kp in keypoints:
        # Match based on proximity (within a threshold distance)
        if abs(kp.pt[0] - u) < 5 and abs(kp.pt[1] - v) < 5:
            matched_keypoints.append([u, v, kp.pt[0], kp.pt[1]])
            # Draw a circle at the UV location
            cv2.circle(img_with_keypoints, (int(u), int(v)), 3, (255, 0, 0), -1)  # Blue circle at UV point

# Save matched keypoints to a CSV file
csv_path = osp.join(output_dir, "matched_keypoints.csv")
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["UV_X", "UV_Y", "Keypoint_X", "Keypoint_Y"])  # Header
    writer.writerows(matched_keypoints)

print(f"Matched keypoints saved to {csv_path}")

# Save the image with detected keypoints
output_img_path = osp.join(output_dir,"query_image_features_detected.png")
cv2.imwrite(output_img_path, img_with_keypoints)
print(f"Image with detected features saved at: {output_img_path}")