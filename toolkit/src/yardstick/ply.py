# Copyright 2026 Martin Fabbri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary little-endian PLY writer and reader for cloud.ply (YS-50, Q032)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("confidence", "u1"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("range_m", "<f4"),
    ]
)


def write_cloud_ply(
    out_path: Path,
    points_xyz: np.ndarray,
    confidences: np.ndarray,
    colors_rgb: np.ndarray,
    ranges_m: np.ndarray,
) -> int:
    """Write binary_little_endian 1.0 PLY with x, y, z, confidence, red, green, blue, range_m (YS-50)."""
    n_verts = int(points_xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Yardstick standard capture point cloud (yardstick-capture/1)\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar confidence\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float range_m\n"
        "end_header\n"
    ).encode("ascii")

    verts = np.empty((n_verts,), dtype=PLY_VERTEX_DTYPE)
    if n_verts > 0:
        verts["x"] = points_xyz[:, 0].astype(np.float32)
        verts["y"] = points_xyz[:, 1].astype(np.float32)
        verts["z"] = points_xyz[:, 2].astype(np.float32)
        verts["confidence"] = confidences.astype(np.uint8)
        verts["red"] = colors_rgb[:, 0].astype(np.uint8)
        verts["green"] = colors_rgb[:, 1].astype(np.uint8)
        verts["blue"] = colors_rgb[:, 2].astype(np.uint8)
        verts["range_m"] = ranges_m.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        if n_verts > 0:
            f.write(verts.tobytes())
    return n_verts


def read_cloud_ply(ply_path: Path) -> np.ndarray:
    """Read binary_little_endian 1.0 PLY written by write_cloud_ply and return structured array."""
    raw = ply_path.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("Invalid PLY file: missing end_header")
    header_text = raw[:idx].decode("ascii", errors="strict")
    if "format binary_little_endian 1.0" not in header_text:
        raise ValueError("PLY file is not binary_little_endian 1.0")
    n_verts = 0
    for line in header_text.splitlines():
        if line.startswith("element vertex "):
            n_verts = int(line.split()[-1])
            break
    body = raw[idx + len(marker) :]
    expected_bytes = n_verts * PLY_VERTEX_DTYPE.itemsize
    if len(body) != expected_bytes:
        raise ValueError(f"PLY body byte size {len(body)} != expected {expected_bytes} for {n_verts} vertices")
    return np.frombuffer(body, dtype=PLY_VERTEX_DTYPE).copy()
