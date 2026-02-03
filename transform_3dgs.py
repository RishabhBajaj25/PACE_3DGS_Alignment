# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np

def euler_zyx_to_matrix(x, y, z, order= "zyx" ):

    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,    0, 1]
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    correction = np.array([
        [ 0, -1, 0],
        [ 0,  0, 1],
        [-1,  0, 0]
    ])
    if order == "zyx":
        return Rz @ Ry @ Rx

    if order == "xyz":
        return Rx @ Ry @ Rz


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# https://github.com/yzslab/gaussian-splatting-lightning/blob/73acf406f6346f2f4241f1deb2ad6c5031258b7e/gaussian_transform.py#L33
def rotate_by_euler_angles(gauss, x: float, y: float, z: float):
    """
    rotate in z-y-x order, radians as unit
    """

    if x == 0. and y == 0. and z == 0.:
        return

    rotation_matrix = euler_zyx_to_matrix(x, y, z)

    rotate_by_matrix(gauss, rotation_matrix)

    # rotate via quaternions
def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), axis=-1)

def rotate_by_matrix(gauss, rotation_matrix, keep_sh_degree: bool = True):
    # rotate xyz
    xyz = np.stack([gauss["x"], gauss["y"], gauss["z"]], axis=1)
    xyz_rot = xyz @ rotation_matrix.T

    gauss["x"] = xyz_rot[:, 0]
    gauss["y"] = xyz_rot[:, 1]
    gauss["z"] = xyz_rot[:, 2]

    # --- rotate gaussian orientations ---
    q_rot = rotmat2qvec(rotation_matrix)[None, :]  # (1,4)

    q_old = np.stack([
        gauss["rot_0"],
        gauss["rot_1"],
        gauss["rot_2"],
        gauss["rot_3"],
    ], axis=1)

    q_new = quat_multiply(q_old, q_rot)
    q_new /= np.linalg.norm(q_new, axis=1, keepdims=True)

    gauss["rot_0"] = q_new[:, 0]
    gauss["rot_1"] = q_new[:, 1]
    gauss["rot_2"] = q_new[:, 2]
    gauss["rot_3"] = q_new[:, 3]

    print("rotation transform applied")

    # TODO: rotate shs
    # if keep_sh_degree is False:
    #     print("set sh_degree=0 when rotation transform enabled")
    #     self.sh_degrees = 0


def translation(gauss, x: float, y: float, z: float):
    if x == 0. and y == 0. and z == 0.:
        return

    gauss["x"] += x
    gauss["y"] += y
    gauss["z"] += z

    print("translation transform applied")



def rescale(gauss, scale: float):
    if scale != 1.:
        gauss["x"]*= scale
        gauss["y"]*= scale
        gauss["z"]*= scale

        gauss["scale_0"]+= np.log(scale)
        gauss["scale_1"]+= np.log(scale)
        gauss["scale_2"]+= np.log(scale)

        print("rescaled with factor {}".format(scale))

gs = PlyData.read("/home/pace-ubuntu/datasets/leica/EAST/pycolmap/all_24_horizontal_yaw_strict_match/supersplat_export_50000.ply")

vertex = gs["vertex"]

print("Number of points:", vertex.count)
print("Vertex properties:")
for name in vertex.data.dtype.names:
    print(" -", name)

rescale(vertex, 1.5483928824968205)
# rotate_by_euler_angles(vertex, np.deg2rad(-179.91297863256196+90), np.deg2rad(-0.6012315326457325), np.deg2rad(90.20316420347123+90))
# rotate_by_euler_angles(vertex, np.deg2rad(-179.91297863256196), np.deg2rad(-0.6012315326457325), np.deg2rad(90.20316420347123))
#
translation(vertex, -1787.16003, -2714.19702, -177.56863)
# rotate_by_euler_angles(vertex, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2))
# translation(vertex, 5, 5, 5)

# Create a new PlyElement from the modified vertex data
vertex_out = PlyElement.describe(
    vertex.data,
    "vertex"
)
elements = [vertex_out]

for elem in gs.elements:
    if elem.name != "vertex":
        elements.append(elem)
out_ply = PlyData(elements, text=gs.text)
out_ply.write("/home/pace-ubuntu/datasets/leica/EAST/transform_tests/translate.ply")

print("Debug")