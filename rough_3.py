# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

from plyfile import PlyData, PlyElement
import numpy as np


# ------------------------------------------------------------
# Quaternion + rotation utilities
# ------------------------------------------------------------

def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def quat_multiply(q0, q1):
    """Hamilton product of quaternions."""
    w0, x0, y0, z0 = np.split(q0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)

    return np.concatenate((
        w1*w0 - x1*x0 - y1*y0 - z1*z0,
        w1*x0 + x1*w0 + y1*z0 - z1*y0,
        w1*y0 - x1*z0 + y1*w0 + z1*x0,
        w1*z0 + x1*y0 - y1*x0 + z1*w0,
    ), axis=-1)


# ------------------------------------------------------------
# Gaussian transforms (Open3D-equivalent)
# ------------------------------------------------------------

def rescale(gauss, scale: float):
    if scale == 1.0:
        return

    gauss["x"] *= scale
    gauss["y"] *= scale
    gauss["z"] *= scale

    log_s = np.log(scale)
    gauss["scale_0"] += log_s
    gauss["scale_1"] += log_s
    gauss["scale_2"] += log_s

    print(f"rescaled by {scale}")


def rotate_by_matrix(gauss, R):
    """
    Apply rotation exactly like Open3D:
        xyz' = R @ xyz
    (implemented as row-vectors → xyz @ R.T)
    """

    xyz = np.stack([gauss["x"], gauss["y"], gauss["z"]], axis=1)
    xyz = xyz @ R.T

    gauss["x"], gauss["y"], gauss["z"] = xyz.T

    # rotate Gaussian orientations
    q_rot = rotmat2qvec(R)[None, :]

    q_old = np.stack([
        gauss["rot_0"],
        gauss["rot_1"],
        gauss["rot_2"],
        gauss["rot_3"],
    ], axis=1)

    q_new = quat_multiply(q_old, q_rot)
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True)

    gauss["rot_0"], gauss["rot_1"], gauss["rot_2"], gauss["rot_3"] = q_new.T

    print("rotation applied")


def translate(gauss, t):
    tx, ty, tz = t
    if tx == 0 and ty == 0 and tz == 0:
        return

    gauss["x"] += tx
    gauss["y"] += ty
    gauss["z"] += tz

    print("translation applied")


# ------------------------------------------------------------
# Load + apply transform
# ------------------------------------------------------------

gs = PlyData.read(
    "/home/pace-ubuntu/datasets/leica/EAST/pycolmap/all_24_horizontal_yaw_strict_match/supersplat_export_50000.ply"
)

vertex = gs["vertex"]

print("Number of points:", vertex.count)
print("Vertex properties:")
for name in vertex.data.dtype.names:
    print(" -", name)


# ---- Decomposed Open3D transform ----
scale = 1.5483928824968205

R = np.array([
    [-1.548305869102, 0.002351583214, -0.016247715801],
    [0.016255933791, 0.005465727299, -1.548297882080],
    [-0.002294085687, -1.548381447792, -0.005490107927],
])

t = np.array([
    -1787.16003,
    -2714.19702,
    -177.56863
])


# ---- Apply in correct order ----
rescale(vertex, scale)
rotate_by_matrix(vertex, R)
translate(vertex, t)


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------

vertex_out = PlyElement.describe(vertex.data, "vertex")
elements = [vertex_out]

for elem in gs.elements:
    if elem.name != "vertex":
        elements.append(elem)

out_ply = PlyData(elements, text=gs.text)
out_ply.write(
    "/home/pace-ubuntu/datasets/leica/EAST/transform_tests/open3d_equivalent.ply"
)

print("Done.")
