import numpy as np
from plyfile import PlyData, PlyElement

INPUT_PLY  = "/home/pace-ubuntu/datasets/leica/EAST/registration/BB_U7_579 EAST_JUNE 2024_ 1 - Cloud - Cloud.ply"
OUTPUT_PLY = "/home/pace-ubuntu/datasets/leica/EAST/registration/center_cloud_gaussian.ply"

# -----------------------------
# Load input PLY
# -----------------------------
ply = PlyData.read(INPUT_PLY)
v = ply["vertex"].data

xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0

N = xyz.shape[0]

# -----------------------------
# Fabricated Gaussian params
# -----------------------------

# Scale: choose something visible
# Simple robust default based on scene size
# bbox = xyz.max(axis=0) - xyz.min(axis=0)
# scene_scale = np.linalg.norm(bbox)
#
# sigma = 0.005 * scene_scale       # tweak if needed
# log_sigma = np.log(sigma)
#
# scales = np.full((N, 3), log_sigma, dtype=np.float32)

# --- Gaussian scale (meters) ---
# TLS spacing is typically 2–10 mm
sigma = 0.01   # 2 mm is a good Leica default

scales = np.full((N, 3), sigma, dtype=np.float32)


# Rotation: identity quaternion
rots = np.zeros((N, 4), dtype=np.float32)
rots[:, 0] = 1.0   # w

# Opacity: logit space
# alpha = 0.8
# opacity = np.full((N, 1), np.log(alpha / (1 - alpha)), dtype=np.float32)
opacity = np.full((N, 1), 0.5, dtype=np.float32)


# SH coefficients
f_dc = rgb.astype(np.float32)
f_rest = np.zeros((N, 45), dtype=np.float32)

# -----------------------------
# Build structured array
# -----------------------------
dtype = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ("opacity", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
]

for i in range(45):
    dtype.append((f"f_rest_{i}", "f4"))

out = np.empty(N, dtype=dtype)

out["x"] = xyz[:, 0]
out["y"] = xyz[:, 1]
out["z"] = xyz[:, 2]

out["scale_0"] = scales[:, 0]
out["scale_1"] = scales[:, 1]
out["scale_2"] = scales[:, 2]

out["opacity"] = opacity[:, 0]

out["rot_0"] = rots[:, 0]
out["rot_1"] = rots[:, 1]
out["rot_2"] = rots[:, 2]
out["rot_3"] = rots[:, 3]

out["f_dc_0"] = f_dc[:, 0]
out["f_dc_1"] = f_dc[:, 1]
out["f_dc_2"] = f_dc[:, 2]

for i in range(45):
    out[f"f_rest_{i}"] = 0.0

# -----------------------------
# Write GS-compatible PLY
# -----------------------------
el = PlyElement.describe(out, "vertex")
PlyData([el], text=False).write(OUTPUT_PLY)

print("Saved:", OUTPUT_PLY)
