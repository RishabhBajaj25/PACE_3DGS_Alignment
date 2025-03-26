import csv
import cv2
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

# This script detects keypoints in the query image that match the projected points in the closest image.
# The resulting matched points will be used for solving the PnP problem.

# Set the output directory for saving results
output_dir = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images"

# Load the query image (frame to be matched)
query_img_path = "/media/rishabh/SSD_1/Data/lab_videos_reg/2_b_20250324_120731_frames_10_fps/images/frame_0000.jpg"
query_img = cv2.imread(query_img_path)

if query_img is None:
    print(f"Error loading image: {query_img_path}")
    exit()

# Define the closest image number to the query frame
selected_image_number = 27  # Example: This is a manually chosen image

# Construct the filename of the closest matching image
selected_image_name = f"frame_{selected_image_number:04d}.jpg"
M1_database = "/media/rishabh/SSD_1/Data/lab_videos_reg/1_b_20250324_120708_frames_10_fps/images"
closest_img_path = osp.join(M1_database, selected_image_name)

# Load the closest matching image
closest_img = cv2.imread(closest_img_path)

if closest_img is None:
    print(f"Error loading image: {closest_img_path}")
    exit()

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Compute keypoints and descriptors for both images
kp1, des1 = sift.detectAndCompute(closest_img, None)  # Keypoints and descriptors for closest image
kp2, des2 = sift.detectAndCompute(query_img, None)  # Keypoints and descriptors for query image

# Use Brute-Force Matcher (BFMatcher) to find matches between descriptors
bf = cv2.BFMatcher()

# Find the two best matches for each keypoint
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to filter good matches
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Stricter threshold to filter strong matches
        good.append([m])  # Wrap each match in a list as required by cv2.drawMatchesKnn()

# Proceed only if there are enough good matches
if len(good) > 10:
    # Extract corresponding keypoints from both images
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute the fundamental matrix using RANSAC (outlier rejection)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 5.0)

    # Select only inlier matches based on the computed mask
    inliers = [good[i] for i in range(len(good)) if mask[i] == 1]

    print(f"Number of inliers: {len(inliers)}")

else:
    print("Not enough good matches found for RANSAC.")

# Draw matches for visualization
img_matches = cv2.drawMatchesKnn(
    closest_img, kp1, query_img, kp2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save the matched feature visualization
plt.imsave(osp.join(output_dir, "matched_features.png"), img_matches)

# Extract inlier keypoints (matched points) from both images
close_matched_points = np.float32([kp1[m[0].queryIdx].pt for m in inliers]).reshape(-1, 1, 2)[:, 0, :]
query_matched_points = np.float32([kp2[m[0].trainIdx].pt for m in inliers]).reshape(-1, 1, 2)[:, 0, :]

# Load the precomputed UV coordinates and 3D points from a CSV file
csv_path = osp.join(output_dir, f"{selected_image_name.split('.')[0]}_uv_coordinates_3d.csv")
XYZ_uv_coordinates = pd.read_csv(csv_path)

# Lists to store matched values
matched_X = []
matched_uv = []
matched_kp_close = []
matched_kp_close_sanity_check = []
matched_kp_query = []

# Counter variable to track the index of query keypoints
counter = 0

# List to store matched data before writing to CSV
matched_data = []

# Iterate through each inlier match and associate it with 3D points
for inlier_matches in close_matched_points:
    for uv, X in zip(
        XYZ_uv_coordinates[['UV_X', 'UV_Y']].to_numpy(),
        XYZ_uv_coordinates[['X_3D', 'Y_3D', 'Z_3D']].to_numpy()
    ):
        # Check if the UV coordinate is within a threshold distance from the inlier match
        if abs(inlier_matches[0] - uv[0]) < 5 and abs(inlier_matches[1] - uv[1]) < 5:
            # Append matched points to respective lists
            matched_X.append(X)
            matched_kp_query.append(query_matched_points[counter])
            matched_uv.append(uv)
            matched_kp_close.append(inlier_matches)
            matched_kp_close_sanity_check.append(close_matched_points[counter])

            # Store matched data as a list (to be written to CSV later)
            matched_data.append([
                uv[0], uv[1], X[0], X[1], X[2],  # UV and 3D coordinates
                matched_kp_query[-1][0], matched_kp_query[-1][1],  # Query keypoint
                matched_kp_close[-1][0], matched_kp_close[-1][1],  # Close image keypoint
                matched_kp_close_sanity_check[-1][0], matched_kp_close_sanity_check[-1][1]  # Sanity check keypoint
            ])
    counter += 1  # Increment counter for next match

# Define CSV output path
output_csv = osp.join(output_dir, "matched_data.csv")

# Write the collected matched data to CSV after the loop
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header row
    writer.writerow([
        "UV_X", "UV_Y", "X_3D", "Y_3D", "Z_3D",
        "Query_KP_x", "Query_KP_y", "Close_KP_x", "Close_KP_y",
        "Close_KP_Sanity_x", "Close_KP_Sanity_y"
    ])

    # Write matched data rows
    writer.writerows(matched_data)

print(f"Matched points saved to {output_csv}")
