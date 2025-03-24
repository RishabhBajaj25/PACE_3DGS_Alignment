import csv
import cv2
import numpy as np
import pandas as pd

# File detects the points in query image that match the projected points in closest image.
# This is to be fed to the PnP problem.

# Load the query image
image_path = "/media/rishabh/SSD_1/Data/lab_videos_reg/2_20250324_120731_frames_10_fps/images/frame_0000.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error loading image: {image_path}")
    exit()

# Initialize ORB feature detector (you can use SIFT or SURF if available)
orb = cv2.ORB_create()

# Detect features in the image
keypoints, descriptors = orb.detectAndCompute(img, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
cv2.imshow('Detected Features', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Static numeric input (e.g., image number 26)
selected_image_number = 27  # You can change this number

# Format the image name based on the number
selected_image_name = f"frame_{selected_image_number:04d}.jpg"

# Load UV coordinates
uv_coordinates = pd.read_csv(f"/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images/{selected_image_name.split('.')[0]}_uv_coordinates.csv")

# Check if the UV points match any detected keypoints
for (u, v) in zip(np.array(uv_coordinates.UV_X), np.array(uv_coordinates.UV_Y)):
    # Check if the UV point is close to any detected keypoint
    for kp in keypoints:
        # Match based on proximity (within a threshold distance)
        if abs(kp.pt[0] - u) < 30 and abs(kp.pt[1] - v) < 30:
            # Draw a circle at the UV location
            cv2.circle(img_with_keypoints, (int(u), int(v)), 3, (255, 0, 0), -1)  # Blue circle at UV point

# Save the image with detected keypoints
output_path = "/media/rishabh/SSD_1/Data/lab_videos_reg/project_2_images/query_image_features_detected.png"
cv2.imwrite(output_path, img_with_keypoints)
print(f"Image with detected features saved at: {output_path}")
