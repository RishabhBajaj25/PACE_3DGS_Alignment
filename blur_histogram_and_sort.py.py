import cv2
import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# ----------------------------
# Argument parsing
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--images", required=True,
    help="path to input directory of images"
)
ap.add_argument(
    "-t", "--threshold", type=float, default=100.0,
    help="focus measures below this value are considered blurry"
)
args = vars(ap.parse_args())



image_dir = args["images"]
threshold = args["threshold"]

hist_out = os.path.join(image_dir, "blur_histogram.png")

# ----------------------------
# Output directories
# ----------------------------
blurry_dir = os.path.join(image_dir, "blurry")
sharp_dir  = os.path.join(image_dir, "sharp")

os.makedirs(blurry_dir, exist_ok=True)
os.makedirs(sharp_dir, exist_ok=True)

# ----------------------------
# Process images
# ----------------------------
blur_values = []
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

for fname in tqdm(sorted(os.listdir(image_dir))):
    if not fname.lower().endswith(valid_exts):
        continue

    src_path = os.path.join(image_dir, fname)

    # Skip files already in output folders
    if src_path.startswith(blurry_dir) or src_path.startswith(sharp_dir):
        continue

    image = cv2.imread(src_path)
    if image is None:
        print(f"[WARN] Could not read {fname}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    blur_values.append(fm)

    # Copy to appropriate folder
    if fm < threshold:
        dst_path = os.path.join(blurry_dir, fname)
    else:
        dst_path = os.path.join(sharp_dir, fname)

    shutil.copy2(src_path, dst_path)

# ----------------------------
# Report
# ----------------------------
blur_values = np.array(blur_values)
print(f"Processed {len(blur_values)} images")
print(f"Blurry (< {threshold}): {np.sum(blur_values < threshold)}")
print(f"Sharp (>= {threshold}): {np.sum(blur_values >= threshold)}")

# ----------------------------
# Histogram
# ----------------------------
plt.figure(figsize=(10, 6))
plt.hist(
    blur_values,
    bins=50,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Blur threshold = {threshold}"
)

plt.xlabel("Variance of Laplacian")
plt.ylabel("Number of Images")
plt.title("Blur Detection Histogram")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(hist_out, dpi=300)
plt.show()
