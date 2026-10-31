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

"""ArUco DICT_5X5_250 verification, ChArUco detection, and independent LiDAR marker scale check (YS-30, YS-33)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from yardstick.geometry import pose_to_matrix


def detect_aruco_markers_opencv(image_bgr_or_gray: np.ndarray) -> dict[int, np.ndarray]:
    """Detect ArUco DICT_5X5_250 markers (IDs 0..5 and ChArUco IDs 100..126) using OpenCV.

    Returns dict mapping marker_id -> 4x2 float64 corner coordinates (c0, c1, c2, c3).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image_bgr_or_gray)
    detected: dict[int, np.ndarray] = {}
    if ids is not None and corners is not None:
        for c, mid_val in zip(corners, np.asarray(ids).ravel()):
            mid = int(mid_val)
            detected[mid] = c.reshape(4, 2).astype(np.float64)
    return detected


def measure_marker_corners_3d_on_lidar_plane(
    corners_uv: np.ndarray,
    depth_mm: np.ndarray,
    conf_img: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_width: int,
    rgb_height: int,
    T_world_from_cam: np.ndarray,
    inward_erosion_ratio: float = 0.10,
) -> tuple[np.ndarray, float, float] | None:
    """Project 4 RGB marker corners onto the 3D plane fitted to interior LiDAR depth pixels (YS-33, Q027).

    Applies 10% inward border erosion on the 2D marker quadrilateral before sampling depth pixels.
    Returns (corners_3d_world (4x3), frame_size_mm, range_m) or None if outside close pass criteria.
    """
    h_d, w_d = depth_mm.shape
    scale_x = float(w_d) / float(rgb_width)
    scale_y = float(h_d) / float(rgb_height)
    fx_d = fx * scale_x
    fy_d = fy * scale_y
    cx_d = cx * scale_x
    cy_d = cy * scale_y

    # Scale corners to depth image coordinates and erode 10% inward toward quad center
    corners_d = np.column_stack([corners_uv[:, 0] * scale_x, corners_uv[:, 1] * scale_y])
    center_d = np.mean(corners_d, axis=0)
    eroded_d = center_d + (1.0 - inward_erosion_ratio) * (corners_d - center_d)

    mask = np.zeros((h_d, w_d), dtype=np.uint8)
    poly_int = np.rint(eroded_d).astype(np.int32)
    cv2.fillConvexPoly(mask, poly_int, 1)

    valid = (mask > 0) & (depth_mm > 0) & (conf_img >= 1)
    if int(np.sum(valid)) < 4:
        # Fallback to uneroded quad if marker is small on 256x192 depth map
        mask_full = np.zeros((h_d, w_d), dtype=np.uint8)
        cv2.fillConvexPoly(mask_full, np.rint(corners_d).astype(np.int32), 1)
        valid = (mask_full > 0) & (depth_mm > 0) & (conf_img >= 1)
        if int(np.sum(valid)) < 4:
            return None

    v_idx, u_idx = np.nonzero(valid)
    z_val = depth_mm[v_idx, u_idx].astype(np.float64) / 1000.0
    x_cam = (u_idx.astype(np.float64) - cx_d) * z_val / fx_d
    y_cam = -(v_idx.astype(np.float64) - cy_d) * z_val / fy_d
    z_cam = -z_val
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    centroid_cam = np.mean(pts_cam, axis=0)
    range_m = float(np.linalg.norm(centroid_cam))
    if not (0.60 <= range_m <= 2.00):
        return None

    # Fit plane in camera space via SVD
    if pts_cam.shape[0] >= 6:
        centered = pts_cam - centroid_cam
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal_cam = vh[-1, :]
        if normal_cam[2] < 0:
            normal_cam = -normal_cam
        # Ensure viewing angle is reasonable (<= 60 deg from normal)
        view_dir = -centroid_cam / max(range_m, 1e-9)
        if abs(float(np.dot(normal_cam, view_dir))) < math.cos(math.radians(60.0)):
            normal_cam = view_dir
    else:
        normal_cam = -centroid_cam / max(range_m, 1e-9)

    d_cam = -float(np.dot(normal_cam, centroid_cam))

    # Intersect the 4 full-res RGB corner rays with the fitted LiDAR depth plane
    corners_3d_cam = np.zeros((4, 3), dtype=np.float64)
    for i in range(4):
        u_rgb, v_rgb = float(corners_uv[i, 0]), float(corners_uv[i, 1])
        ray = np.array([(u_rgb - cx) / fx, -(v_rgb - cy) / fy, -1.0], dtype=np.float64)
        denom = float(np.dot(normal_cam, ray))
        if abs(denom) < 1e-9:
            return None
        t_ray = -d_cam / denom
        if t_ray <= 0:
            return None
        corners_3d_cam[i] = t_ray * ray

    edges_m = [
        float(np.linalg.norm(corners_3d_cam[(i + 1) % 4] - corners_3d_cam[i]))
        for i in range(4)
    ]
    frame_size_mm = 1000.0 * float(np.mean(edges_m))

    R_wc = T_world_from_cam[:3, :3]
    t_wc = T_world_from_cam[:3, 3]
    corners_3d_world = (corners_3d_cam @ R_wc.T) + t_wc

    return corners_3d_world, frame_size_mm, range_m


def evaluate_marker_scale_errors(
    capture_dir: Path,
    markers_rows: list[dict[str, Any]],
    poses_by_index: dict[int, dict[str, Any]],
    passes_by_marker: dict[int, list[list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Compute multi-frame fused marker scale error and standard error across frames (YS-33, Q027)."""
    if not markers_rows:
        return {
            "scale_error_ratio": 0.0,
            "scale_error_se": 0.0,
            "scale_error_eff": 0.0,
            "sightings_used": 0,
            "per_marker": [],
        }

    per_marker_results: list[dict[str, Any]] = []
    all_pass_errors: list[float] = []
    all_pass_ses: list[float] = []
    total_sightings_used = 0

    for mid, m_passes in sorted(passes_by_marker.items()):
        printed_sizes: list[float] = []
        frame_measured_sizes: list[float] = []
        fused_pass_sizes: list[float] = []

        for m_pass in m_passes:
            pass_corners_world: list[np.ndarray] = []
            for row in m_pass:
                s_idx = int(row["saved_index"])
                printed_mm = float(row["size_mm_printed"])
                printed_sizes.append(printed_mm)

                depth_path = capture_dir / "depth" / f"{s_idx:06d}.png"
                conf_path = capture_dir / "confidence" / f"{s_idx:06d}.png"
                pose_row = poses_by_index.get(s_idx)

                has_corners = all(k in row and row[k] != "" for k in ("c0_u", "c0_v", "c1_u", "c1_v", "c2_u", "c2_v", "c3_u", "c3_v"))
                if has_corners and depth_path.exists() and conf_path.exists() and pose_row is not None:
                    corners_uv = np.array(
                        [
                            [float(row["c0_u"]), float(row["c0_v"])],
                            [float(row["c1_u"]), float(row["c1_v"])],
                            [float(row["c2_u"]), float(row["c2_v"])],
                            [float(row["c3_u"]), float(row["c3_v"])],
                        ],
                        dtype=np.float64,
                    )
                    depth_mm = np.array(Image.open(depth_path), dtype=np.uint16)
                    conf_img = np.array(Image.open(conf_path), dtype=np.uint8)
                    T_wc = pose_to_matrix(
                        float(pose_row["px"]),
                        float(pose_row["py"]),
                        float(pose_row["pz"]),
                        float(pose_row["qx"]),
                        float(pose_row["qy"]),
                        float(pose_row["qz"]),
                        float(pose_row["qw"]),
                    )
                    res = measure_marker_corners_3d_on_lidar_plane(
                        corners_uv=corners_uv,
                        depth_mm=depth_mm,
                        conf_img=conf_img,
                        fx=float(pose_row["fx"]),
                        fy=float(pose_row["fy"]),
                        cx=float(pose_row["cx"]),
                        cy=float(pose_row["cy"]),
                        rgb_width=int(pose_row["width"]),
                        rgb_height=int(pose_row["height"]),
                        T_world_from_cam=T_wc,
                    )
                    if res is not None:
                        corners_w, f_size_mm, _ = res
                        pass_corners_world.append(corners_w)
                        frame_measured_sizes.append(f_size_mm)
                        continue

                # Fallback to logged size_mm_measured if depth frame outside close range or synthetic
                logged_mm = float(row["size_mm_measured"])
                if logged_mm > 0:
                    frame_measured_sizes.append(logged_mm)

            if pass_corners_world:
                mean_corners_w = np.mean(np.stack(pass_corners_world, axis=0), axis=0)
                fused_edges_m = [
                    float(np.linalg.norm(mean_corners_w[(i + 1) % 4] - mean_corners_w[i]))
                    for i in range(4)
                ]
                fused_pass_sizes.append(1000.0 * float(np.mean(fused_edges_m)))

        if not frame_measured_sizes:
            continue

        printed_ref = float(np.median(printed_sizes))
        if fused_pass_sizes:
            measured_mm = float(np.mean(fused_pass_sizes))
        else:
            measured_mm = float(np.mean(frame_measured_sizes))

        err_ratio = (measured_mm - printed_ref) / printed_ref
        frame_errs = [(s - printed_ref) / printed_ref for s in frame_measured_sizes]
        n_used = len(frame_errs)
        if n_used >= 2:
            se = float(np.std(frame_errs, ddof=1) / math.sqrt(n_used))
        else:
            # Single-frame sighting fallback standard error (~0.5%)
            se = 0.005

        total_sightings_used += n_used
        all_pass_errors.append(err_ratio)
        all_pass_ses.append(se)

        per_marker_results.append(
            {
                "marker_id": int(mid),
                "size_mm_printed": round(printed_ref, 3),
                "size_mm_measured": round(measured_mm, 3),
                "scale_error_ratio": round(err_ratio, 6),
                "scale_error_se": round(se, 6),
                "sightings_count": n_used,
            }
        )

    if not per_marker_results:
        return {
            "scale_error_ratio": 0.0,
            "scale_error_se": 0.0,
            "scale_error_eff": 0.0,
            "sightings_used": 0,
            "per_marker": [],
        }

    # Aggregate across markers: take the marker with the largest effective error or median
    idx_med = int(np.argmax([abs(e) for e in all_pass_errors]))
    agg_err = float(all_pass_errors[idx_med])
    agg_se = float(all_pass_ses[idx_med])
    # YS-41 / Q027: scale_error_eff = max(0, |scale_error| - 2 * SE)
    agg_eff = max(0.0, abs(agg_err) - 2.0 * agg_se)

    return {
        "scale_error_ratio": round(agg_err, 6),
        "scale_error_se": round(agg_se, 6),
        "scale_error_eff": round(agg_eff, 6),
        "sightings_used": total_sightings_used,
        "per_marker": per_marker_results,
    }
