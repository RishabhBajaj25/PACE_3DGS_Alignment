`#!/usr/bin/env python3
import argparse
import numpy as np


# ----------------------------
# Quaternion utils
# ----------------------------
def quat_normalize(q):
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n

def quat_mul_wxyz(q1, q2):
    """
    Hamilton product, both in (w, x, y, z).
    Returns q = q1 ⊗ q2
    q1,q2: (N,4)
    """
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w,x,y,z], axis=1)

def rotmat_to_quat_wxyz(R):
    """
    Convert 3x3 rotation matrix to quaternion (w,x,y,z).
    Robust enough for typical registration rotations.
    """
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w,x,y,z], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


# ----------------------------
# PLY read/write (binary little endian, float32 vertex table only)
# ----------------------------
EXPECTED_PROPS = [
    "x","y","z",
    "scale_0","scale_1","scale_2",
    "opacity",
    "rot_0","rot_1","rot_2","rot_3",
    "f_dc_0","f_dc_1","f_dc_2",
    *[f"f_rest_{i}" for i in range(45)],
]

def read_ply_binary_f32(path):
    with open(path, "rb") as f:
        header_lines = []
        line = f.readline()
        if not line.startswith(b"ply"):
            raise ValueError("Not a PLY file (missing 'ply').")
        header_lines.append(line)

        fmt = None
        n_verts = None
        props = []

        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading header.")
            header_lines.append(line)
            s = line.decode("utf-8", errors="replace").strip()

            if s.startswith("format "):
                fmt = s.split()[1]
                if fmt != "binary_little_endian":
                    raise ValueError(f"Only binary_little_endian supported, got {fmt}")
            elif s.startswith("element vertex"):
                n_verts = int(s.split()[2])
            elif s.startswith("property "):
                # property float name
                toks = s.split()
                if len(toks) >= 3 and toks[1] == "float":
                    props.append(toks[2])
                else:
                    raise ValueError(f"Only 'property float <name>' supported. Got: {s}")
            elif s == "end_header":
                break

        if n_verts is None:
            raise ValueError("Missing 'element vertex <N>' in header.")

        # Sanity check that properties match what SuperSplat expects
        if props != EXPECTED_PROPS:
            raise ValueError(
                "PLY properties do not match expected Gaussian Splat layout.\n"
                f"Found {len(props)} props.\n"
                "If this file is valid but differs, paste the full header and I’ll adapt the parser.\n"
                f"First few found props: {props[:12]}"
            )

        n_props = len(props)
        data = np.fromfile(f, dtype="<f4", count=n_verts * n_props)
        if data.size != n_verts * n_props:
            raise ValueError("Binary payload size does not match header counts.")
        data = data.reshape((n_verts, n_props))

    return header_lines, props, data

def write_ply_binary_f32(path, header_lines, data):
    data_f32 = np.asarray(data, dtype="<f4")
    with open(path, "wb") as f:
        for line in header_lines:
            f.write(line)
        data_f32.tofile(f)


# ----------------------------
# Transform logic
# ----------------------------
def extract_similarity_from_T(T):
    R_s = T[:3,:3]
    t = T[:3,3].astype(np.float64)

    col_norms = np.array([np.linalg.norm(R_s[:,0]),
                          np.linalg.norm(R_s[:,1]),
                          np.linalg.norm(R_s[:,2])], dtype=np.float64)
    s = float(col_norms.mean())
    if s <= 0:
        raise ValueError("Extracted scale <= 0. Bad transform?")

    R = (R_s / s).astype(np.float64)
    return s, R, t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ply")
    ap.add_argument("output_ply")
    ap.add_argument("--matrix", required=True, help="Path to 4x4 transform matrix .npy OR a .txt with 16 numbers")
    ap.add_argument("--scale_mode", choices=["linear","log"], default="log",
                    help="How scale_0..2 are stored. Many 3DGS PLYs use log-scales. Default: log")
    ap.add_argument("--quat_order", choices=["wxyz","xyzw"], default="wxyz",
                    help="Order of rot_0..3 in file. Default: wxyz")
    ap.add_argument("--compose", choices=["left","right"], default="left",
                    help="How to compose rotation: left means q' = qR ⊗ q (typical world transform). Default: left")
    args = ap.parse_args()

    # Load matrix
    if args.matrix.endswith(".npy"):
        T = np.load(args.matrix).astype(np.float64)
    else:
        vals = np.loadtxt(args.matrix, dtype=np.float64).reshape(4,4)
        T = vals
    if T.shape != (4,4):
        raise ValueError("Transform must be 4x4.")

    header_lines, props, data = read_ply_binary_f32(args.input_ply)

    # Indices
    ix, iy, iz = 0, 1, 2
    is0, is1, is2 = 3, 4, 5
    iop = 6
    ir0, ir1, ir2, ir3 = 7, 8, 9, 10

    # Extract similarity transform
    s, R, t = extract_similarity_from_T(T)

    # Transform positions
    xyz = data[:, [ix,iy,iz]].astype(np.float64)
    xyz_new = (s * (xyz @ R.T)) + t  # (N,3)
    data[:, [ix,iy,iz]] = xyz_new.astype(np.float32)

    # Transform per-Gaussian scales
    scales = data[:, [is0,is1,is2]].astype(np.float64)
    if args.scale_mode == "linear":
        scales_new = scales * s
    else:
        # log-scale: log(a*s) = log(a) + log(s)
        scales_new = scales + np.log(s)
    data[:, [is0,is1,is2]] = scales_new.astype(np.float32)

    # Transform quaternion rotations
    q = data[:, [ir0,ir1,ir2,ir3]].astype(np.float64)

    if args.quat_order == "xyzw":
        # convert to wxyz for math
        q = q[:, [3,0,1,2]]

    q = quat_normalize(q)

    qR = rotmat_to_quat_wxyz(R)  # (4,)
    qR = np.tile(qR[None, :], (q.shape[0], 1))

    if args.compose == "left":
        q_new = quat_mul_wxyz(qR, q)     # q' = qR ⊗ q
    else:
        q_new = quat_mul_wxyz(q, qR)     # q' = q ⊗ qR

    q_new = quat_normalize(q_new)

    if args.quat_order == "xyzw":
        # convert back
        q_new = q_new[:, [1,2,3,0]]

    data[:, [ir0,ir1,ir2,ir3]] = q_new.astype(np.float32)

    # Write output
    write_ply_binary_f32(args.output_ply, header_lines, data)
    print("Wrote:", args.output_ply)
    print("Applied similarity: scale =", s)
    print("Translation:", t)
    print("Rotation R:\n", R)


if __name__ == "__main__":
    main()

 