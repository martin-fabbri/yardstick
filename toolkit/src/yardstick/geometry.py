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

"""Coordinate transforms, Hamilton quaternions, depth unprojection, RANSAC ground plane,
Marker 0 yard frame, and scanned-ground coverage polygon (YS-05, YS-31, YS-50, YS-51, YS-52).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert Hamilton unit quaternion (qx, qy, qz, qw) to 3x3 rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to Hamilton unit quaternion (qx, qy, qz, qw)."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if qw < 0:
        norm = -norm
    return (float(qx / norm), float(qy / norm), float(qz / norm), float(qw / norm))


def pose_to_matrix(px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Build 4x4 homogeneous transform matrix from position and Hamilton quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
    T[:3, 3] = [px, py, pz]
    return T


def is_orthonormal_se3(T: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify 4x4 matrix is a valid rigid SE(3) transform within tolerance (YS-23)."""
    if T.shape != (4, 4):
        return False
    if not np.all(np.isfinite(T)):
        return False
    if np.max(np.abs(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > tol:
        return False
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > tol:
        return False
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    return ortho_err <= tol


def camera_pitch_deg_from_pose(T_world_from_cam: np.ndarray, up_axis: np.ndarray | None = None) -> float:
    """Compute grip-invariant camera pitch in degrees from optical viewing ray (YS-05).

    Optical look ray is -Z in ARKit camera coordinates: v_look = R @ [0, 0, -1].
    Pitch is arcsin(v_look . up_axis) in degrees (negative = tilted down toward ground).
    """
    if up_axis is None:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_axis = up_axis / np.linalg.norm(up_axis)
    R = T_world_from_cam[:3, :3]
    v_look = R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    v_look = v_look / np.linalg.norm(v_look)
    sin_pitch = float(np.clip(np.dot(v_look, up_axis), -1.0, 1.0))
    return math.degrees(math.asin(sin_pitch))


def unproject_depth_frame(
    depth_mm: np.ndarray,
    conf_img: np.ndarray,
    rgb_img: np.ndarray | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_width: int,
    rgb_height: int,
    T_target_from_cam: np.ndarray,
    min_conf: int = 1,
    min_range_m: float = 0.20,
    max_range_m: float = 4.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unproject 16-bit depth (mm) into 3D points in target frame (YS-04, YS-50).

    Returns (points_xyz, colors_rgb, confidences, ranges_m).
    """
    h_d, w_d = depth_mm.shape
    scale_x = float(w_d) / float(rgb_width)
    scale_y = float(h_d) / float(rgb_height)
    fx_d = fx * scale_x
    fy_d = fy * scale_y
    cx_d = cx * scale_x
    cy_d = cy * scale_y

    z_d = depth_mm.astype(np.float64) / 1000.0
    valid = (depth_mm > 0) & (conf_img >= min_conf)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.float64),
        )

    v_idx, u_idx = np.nonzero(valid)
    z_val = z_d[v_idx, u_idx]
    x_cam = (u_idx.astype(np.float64) - cx_d) * z_val / fx_d
    y_cam = -(v_idx.astype(np.float64) - cy_d) * z_val / fy_d
    z_cam = -z_val
    ranges = np.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)

    in_range = (ranges >= min_range_m) & (ranges <= max_range_m)
    if not np.any(in_range):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.float64),
        )

    x_cam = x_cam[in_range]
    y_cam = y_cam[in_range]
    z_cam = z_cam[in_range]
    ranges = ranges[in_range]
    u_valid = u_idx[in_range]
    v_valid = v_idx[in_range]
    confs = conf_img[v_valid, u_valid].astype(np.uint8)

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = T_target_from_cam[:3, :3]
    t = T_target_from_cam[:3, 3]
    pts_target = (pts_cam @ R.T) + t

    if rgb_img is not None and rgb_img.shape[0] > 0 and rgb_img.shape[1] > 0:
        rh, rw = rgb_img.shape[:2]
        u_rgb = np.clip((u_valid.astype(np.float64) * rw / w_d).astype(np.int32), 0, rw - 1)
        v_rgb = np.clip((v_valid.astype(np.float64) * rh / h_d).astype(np.int32), 0, rh - 1)
        colors = rgb_img[v_rgb, u_rgb, :3].astype(np.uint8)
    else:
        colors = np.full((pts_target.shape[0], 3), 160, dtype=np.uint8)

    return pts_target, colors, confs, ranges


def fit_ground_plane_ransac(
    points_world: np.ndarray,
    camera_positions_world: np.ndarray,
    marker_0_pos_world: np.ndarray | None = None,
    max_tilt_deg: float = 15.0,
    inlier_threshold_m: float = 0.03,
    iterations: int = 250,
    seed: int = 42,
) -> tuple[np.ndarray, float, float]:
    """Fit ground plane n . p + d = 0 in ARKit world coordinates (+Y up) (YS-51, Q030).

    Enforces |n . y_world| >= cos(max_tilt_deg) so walls/fences/steps cannot become the ground.
    Returns (unit_normal_up, d_offset, median_abs_residual_m).
    """
    y_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cos_min = math.cos(math.radians(max_tilt_deg))

    if points_world.shape[0] < 10:
        ref_y = float(marker_0_pos_world[1]) if marker_0_pos_world is not None else 0.0
        return y_up, -ref_y, 0.0

    if marker_0_pos_world is not None:
        ref_y = float(marker_0_pos_world[1])
        band_mask = np.abs(points_world[:, 1] - ref_y) <= 0.40
    elif camera_positions_world.shape[0] > 0:
        ref_y = float(np.percentile(camera_positions_world[:, 1], 15)) - 1.20
        band_mask = np.abs(points_world[:, 1] - ref_y) <= 0.50
    else:
        ref_y = float(np.percentile(points_world[:, 1], 15))
        band_mask = np.abs(points_world[:, 1] - ref_y) <= 0.50

    candidates = points_world[band_mask]
    if candidates.shape[0] < 10:
        candidates = points_world

    rng = np.random.default_rng(seed)
    n_pts = candidates.shape[0]
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_normal = y_up.copy()
    best_d = -ref_y

    for _ in range(iterations):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = candidates[idx[0]], candidates[idx[1]], candidates[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len
        if normal[1] < 0.0:
            normal = -normal
        # Gravity-normal tilt gate (YS-51 / Q030)
        if float(np.dot(normal, y_up)) < cos_min:
            continue
        d = -float(np.dot(normal, p0))
        dists = np.abs(candidates @ normal + d)
        inliers = dists <= inlier_threshold_m
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_inliers is not None and best_count >= 6:
        inlier_pts = candidates[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        refined_normal = vh[-1, :]
        if refined_normal[1] < 0.0:
            refined_normal = -refined_normal
        if float(np.dot(refined_normal, y_up)) >= cos_min:
            best_normal = refined_normal / np.linalg.norm(refined_normal)
            best_d = -float(np.dot(best_normal, centroid))
        residuals = np.abs(inlier_pts @ best_normal + best_d)
        median_residual = float(np.median(residuals))
    else:
        residuals = np.abs(candidates @ best_normal + best_d)
        median_residual = float(np.median(residuals)) if residuals.size > 0 else 0.0

    return best_normal, best_d, median_residual


def cluster_marker_passes(
    sightings: list[dict[str, Any]],
    camera_poses_by_index: dict[int, np.ndarray],
    min_time_gap_s: float = 3.0,
    min_excursion_m: float = 1.0,
) -> dict[int, list[list[dict[str, Any]]]]:
    """Group sightings of each marker_id into passes separated by time and trajectory excursion."""
    by_marker: dict[int, list[dict[str, Any]]] = {}
    for s in sightings:
        mid = int(s["marker_id"])
        by_marker.setdefault(mid, []).append(s)

    passes_by_marker: dict[int, list[list[dict[str, Any]]]] = {}
    for mid, rows in by_marker.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["t"]))
        passes: list[list[dict[str, Any]]] = [[rows_sorted[0]]]
        for r in rows_sorted[1:]:
            prev = passes[-1][-1]
            dt = float(r["t"]) - float(prev["t"])
            if dt >= min_time_gap_s:
                idx_prev = int(prev["saved_index"])
                idx_curr = int(r["saved_index"])
                # Measure max camera excursion from initial pass marker position during [idx_prev, idx_curr]
                m_pos = np.array([float(prev["x"]), float(prev["y"]), float(prev["z"])], dtype=np.float64)
                max_exc = 0.0
                for k in range(idx_prev, idx_curr + 1):
                    if k in camera_poses_by_index:
                        cam_p = camera_poses_by_index[k][:3, 3]
                        max_exc = max(max_exc, float(np.linalg.norm(cam_p - m_pos)))
                if max_exc >= min_excursion_m or dt >= 10.0:
                    passes.append([r])
                    continue
            passes[-1].append(r)
        passes_by_marker[mid] = passes
    return passes_by_marker


def detect_pose_jumps(
    poses_rows: list[dict[str, Any]],
    jump_threshold_m: float = 0.08,
    max_dt_s: float = 0.10,
) -> list[dict[str, Any]]:
    """Detect non-physical single-frame ARKit relocalization snaps (> 0.08 m within 0.1 s) (YS-32, Q026)."""
    jumps: list[dict[str, Any]] = []
    for i in range(1, len(poses_rows)):
        p_prev = np.array(
            [float(poses_rows[i - 1]["px"]), float(poses_rows[i - 1]["py"]), float(poses_rows[i - 1]["pz"])],
            dtype=np.float64,
        )
        p_curr = np.array(
            [float(poses_rows[i]["px"]), float(poses_rows[i]["py"]), float(poses_rows[i]["pz"])],
            dtype=np.float64,
        )
        dt = float(poses_rows[i]["t"]) - float(poses_rows[i - 1]["t"])
        dist = float(np.linalg.norm(p_curr - p_prev))
        if 0.0 < dt <= max_dt_s and dist > jump_threshold_m:
            jumps.append(
                {
                    "saved_index_from": int(poses_rows[i - 1]["saved_index"]),
                    "saved_index_to": int(poses_rows[i]["saved_index"]),
                    "t_s": round(float(poses_rows[i]["t"]), 6),
                    "jump_m": round(dist, 6),
                }
            )
    return jumps


def build_yard_frame(
    ground_normal_world: np.ndarray,
    ground_d_world: float,
    ground_residual_m: float,
    passes_by_marker: dict[int, list[list[dict[str, Any]]]],
    first_camera_pose_world: np.ndarray,
    pose_jumps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct world_to_yard_transform (4x4) and frame residuals (YS-31, YS-32, YS-51)."""
    z_yard = ground_normal_world / np.linalg.norm(ground_normal_world)
    jump_sum = sum(float(j["jump_m"]) for j in pose_jumps)

    loop_closure_gaps: list[dict[str, Any]] = []
    for mid, m_passes in sorted(passes_by_marker.items()):
        if len(m_passes) >= 2:
            first_pass = m_passes[0]
            last_pass = m_passes[-1]
            p_first = np.median(
                np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in first_pass], dtype=np.float64),
                axis=0,
            )
            p_last = np.median(
                np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in last_pass], dtype=np.float64),
                axis=0,
            )
            raw_gap = float(np.linalg.norm(p_last - p_first))
            eff_gap = max(raw_gap, jump_sum) if mid == 0 else raw_gap
            loop_closure_gaps.append(
                {
                    "marker_id": int(mid),
                    "first_t_s": round(float(first_pass[0]["t"]), 6),
                    "last_t_s": round(float(last_pass[-1]["t"]), 6),
                    "gap_m": round(eff_gap, 6),
                }
            )

    if 0 in passes_by_marker and len(passes_by_marker[0]) >= 1:
        first_pass_m0 = passes_by_marker[0][0]
        p_m0 = np.median(
            np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in first_pass_m0], dtype=np.float64),
            axis=0,
        )
        x_axes = []
        z_axes = []
        for r in first_pass_m0:
            R_m = quat_to_rotmat(float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
            x_axes.append(R_m[:, 0])
            z_axes.append(R_m[:, 2])
        v_x_m0 = np.mean(np.stack(x_axes, axis=0), axis=0)
        v_z_m0 = np.mean(np.stack(z_axes, axis=0), axis=0)
        v_z_m0 = v_z_m0 / max(float(np.linalg.norm(v_z_m0)), 1e-12)

        # Check Marker 0 normal is within 30 deg of ground up (YS-31 / Q030)
        cos_30 = math.cos(math.radians(30.0))
        marker_0_normal_ok = float(np.dot(v_z_m0, z_yard)) >= cos_30

        origin_height_m = float(np.dot(z_yard, p_m0) + ground_d_world)
        o_yard = p_m0 - origin_height_m * z_yard

        x_proj = v_x_m0 - float(np.dot(v_x_m0, z_yard)) * z_yard
        norm_x = float(np.linalg.norm(x_proj))
        if norm_x < 1e-6:
            # Degenerate horizontal projection fallback
            marker_0_normal_ok = False
            x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64) - z_yard[0] * z_yard
            norm_x = float(np.linalg.norm(x_proj))
        x_yard = x_proj / norm_x
        y_yard = np.cross(z_yard, x_yard)
        y_yard = y_yard / np.linalg.norm(y_yard)

        R_yard_from_world = np.stack([x_yard, y_yard, z_yard], axis=0)
        t_yard_from_world = -R_yard_from_world @ o_yard
        T_yard_from_world = np.eye(4, dtype=np.float64)
        T_yard_from_world[:3, :3] = R_yard_from_world
        T_yard_from_world[:3, 3] = t_yard_from_world

        origin_residual_m: float | None = None
        if len(passes_by_marker[0]) >= 2:
            last_pass_m0 = passes_by_marker[0][-1]
            p_m0_last = np.median(
                np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in last_pass_m0], dtype=np.float64),
                axis=0,
            )
            p_last_yard = R_yard_from_world @ p_m0_last + t_yard_from_world
            p_first_yard = R_yard_from_world @ p_m0 + t_yard_from_world
            origin_residual_m = round(float(np.linalg.norm(p_last_yard - p_first_yard)), 6)

        return {
            "method": "marker",
            "marker_id": 0,
            "marker_0_normal_ok": bool(marker_0_normal_ok),
            "world_to_yard_transform": [round(float(x), 8) for x in T_yard_from_world.flatten()],
            "T_matrix": T_yard_from_world,
            "origin_height_m": round(origin_height_m, 6),
            "origin_residual_m": origin_residual_m,
            "ground_plane_residual_m": round(ground_residual_m, 6),
            "loop_closure_gaps": loop_closure_gaps,
            "pose_jumps": pose_jumps,
        }

    # Fallback: first camera pose projected to ground plane (YS-31 floor)
    p_cam0 = first_camera_pose_world[:3, 3]
    origin_height_m = float(np.dot(z_yard, p_cam0) + ground_d_world)
    o_yard = p_cam0 - origin_height_m * z_yard

    forward_world = first_camera_pose_world[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_proj = forward_world - float(np.dot(forward_world, z_yard)) * z_yard
    norm_x = float(np.linalg.norm(x_proj))
    if norm_x < 1e-6:
        x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64) - z_yard[0] * z_yard
        norm_x = float(np.linalg.norm(x_proj))
    x_yard = x_proj / norm_x
    y_yard = np.cross(z_yard, x_yard)
    y_yard = y_yard / np.linalg.norm(y_yard)

    R_yard_from_world = np.stack([x_yard, y_yard, z_yard], axis=0)
    t_yard_from_world = -R_yard_from_world @ o_yard
    T_yard_from_world = np.eye(4, dtype=np.float64)
    T_yard_from_world[:3, :3] = R_yard_from_world
    T_yard_from_world[:3, 3] = t_yard_from_world

    return {
        "method": "first_pose",
        "marker_id": None,
        "marker_0_normal_ok": False,
        "world_to_yard_transform": [round(float(x), 8) for x in T_yard_from_world.flatten()],
        "T_matrix": T_yard_from_world,
        "origin_height_m": round(origin_height_m, 6),
        "origin_residual_m": None,
        "ground_plane_residual_m": round(ground_residual_m, 6),
        "loop_closure_gaps": loop_closure_gaps,
        "pose_jumps": pose_jumps,
    }


def calibrate_north_in_yard_frame(
    location_obj: dict[str, Any],
    T_yard_from_world: np.ndarray,
) -> dict[str, Any]:
    """Calibrate True North azimuth and uncertainty in the Yard Frame from Marker 0 start samples (YS-14, Q020)."""
    loc_status = location_obj.get("status", "unavailable")
    if loc_status == "denied":
        return {
            "status": "denied",
            "north_angle_deg": None,
            "north_uncertainty_deg": None,
            "sample_count": 0,
        }
    samples = location_obj.get("north_samples", [])
    valid_samples = [
        s
        for s in samples
        if s.get("true_heading_deg") is not None
        and s.get("heading_accuracy_deg") is not None
        and 0.0 <= float(s["heading_accuracy_deg"]) <= 25.0
    ]
    if len(valid_samples) < 10:
        if any(
            s.get("heading_accuracy_deg") is not None and float(s["heading_accuracy_deg"]) > 25.0
            for s in samples
        ) or loc_status == "accuracy_exceeded":
            return {
                "status": "accuracy_exceeded",
                "north_angle_deg": None,
                "north_uncertainty_deg": None,
                "sample_count": len(valid_samples),
            }
        return {
            "status": "insufficient_samples",
            "north_angle_deg": None,
            "north_uncertainty_deg": None,
            "sample_count": len(valid_samples),
        }

    # Compute Yard Frame +X azimuth in World horizontal XZ plane:
    # In ARKit world (+Y up), camera_yaw_world_deg is the counter-clockwise yaw in the horizontal plane
    # of the camera's forward look ray relative to world +X, while true_heading_deg is clockwise from True North.
    R_yw = T_yard_from_world[:3, :3]
    north_angles_rad: list[float] = []
    accuracies_deg: list[float] = []

    for s in valid_samples:
        yaw_w_rad = math.radians(float(s["camera_yaw_world_deg"]))
        heading_rad = math.radians(float(s["true_heading_deg"]))
        acc_deg = float(s["heading_accuracy_deg"])

        # Camera horizontal look vector in world (ARKit: X, 0, Z where yaw is CCW from +X in XZ)
        v_cam_w = np.array([math.cos(yaw_w_rad), 0.0, -math.sin(yaw_w_rad)], dtype=np.float64)
        v_cam_y = R_yw @ v_cam_w
        yaw_cam_yard = math.atan2(float(v_cam_y[1]), float(v_cam_y[0]))
        # True heading is clockwise angle from True North to camera look vector:
        # yaw_cam_yard = north_angle_rad - heading_rad  (since +Y_yard is CCW from +X_yard)
        north_rad = yaw_cam_yard + heading_rad
        north_angles_rad.append(north_rad)
        accuracies_deg.append(acc_deg)

    sin_mean = float(np.mean([math.sin(a) for a in north_angles_rad]))
    cos_mean = float(np.mean([math.cos(a) for a in north_angles_rad]))
    mean_angle_rad = math.atan2(sin_mean, cos_mean)
    mean_angle_deg = (math.degrees(mean_angle_rad) + 360.0) % 360.0

    # Circular standard error in degrees
    diffs_deg = [
        math.degrees(math.atan2(math.sin(a - mean_angle_rad), math.cos(a - mean_angle_rad)))
        for a in north_angles_rad
    ]
    sample_se_deg = float(np.std(diffs_deg, ddof=1) / math.sqrt(len(diffs_deg))) if len(diffs_deg) > 1 else 0.0
    med_acc_deg = float(np.median(accuracies_deg))
    uncertainty_deg = math.sqrt(sample_se_deg * sample_se_deg + med_acc_deg * med_acc_deg)

    return {
        "status": "ok",
        "north_angle_deg": round(mean_angle_deg, 3),
        "north_uncertainty_deg": round(uncertainty_deg, 3),
        "sample_count": len(valid_samples),
    }


def voxel_downsample_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    confidences: np.ndarray,
    ranges_m: np.ndarray,
    voxel_size_m: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-downsample point cloud at 2.5 cm: centroid xyz, mean RGB, max confidence, min range (YS-50)."""
    if points_xyz.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
        )
    voxel_coords = np.floor(points_xyz / voxel_size_m).astype(np.int64)
    _, inverse_idx, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    n_voxels = counts.shape[0]

    out_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(out_xyz, inverse_idx, points_xyz)
    out_xyz /= counts[:, None]

    out_rgb = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(out_rgb, inverse_idx, colors_rgb.astype(np.float64))
    out_rgb = np.clip(np.rint(out_rgb / counts[:, None]), 0, 255).astype(np.uint8)

    out_conf = np.zeros((n_voxels,), dtype=np.uint8)
    np.maximum.at(out_conf, inverse_idx, confidences)

    out_range = np.full((n_voxels,), np.inf, dtype=np.float64)
    np.minimum.at(out_range, inverse_idx, ranges_m)

    return out_xyz.astype(np.float32), out_rgb, out_conf, out_range.astype(np.float32)


def _polygon_signed_area(pts: np.ndarray) -> float:
    """Signed shoelace area (> 0 for counter-clockwise)."""
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Return True if closed line segments ab and cd intersect."""
    def ccw(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

    d1 = ccw(a, b, c)
    d2 = ccw(a, b, d)
    d3 = ccw(c, d, a)
    d4 = ccw(c, d, b)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def is_simple_polygon(vertices: list[list[float]]) -> bool:
    """Return True if polygon has >= 3 vertices, non-zero area, and zero non-adjacent edge intersections."""
    n = len(vertices)
    if n < 3:
        return False
    pts = np.array(vertices, dtype=np.float64)
    if abs(_polygon_signed_area(pts)) <= 1e-6:
        return False
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        if np.linalg.norm(b - a) < 1e-6:
            return False
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c = pts[j]
            d = pts[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def compute_scanned_ground_coverage(
    points_xyz: np.ndarray,
    confidences: np.ndarray,
    ranges_m: np.ndarray,
    z_max_abs_m: float = 0.25,
    range_max_m: float = 4.0,
    grid_cell_m: float = 0.20,
    closing_cells: int = 1,
    max_filled_hole_m2: float = 1.0,
    simplify_epsilon_m: float = 0.10,
) -> dict[str, Any]:
    """Compute scanned ground outline and interior holes > 1.0 m^2 (YS-52, Q031)."""
    mask = (np.abs(points_xyz[:, 2]) <= z_max_abs_m) & (ranges_m <= range_max_m) & (confidences >= 1)
    ground_pts = points_xyz[mask]
    ground_confs = confidences[mask]

    if ground_pts.shape[0] == 0:
        raise ValueError("No valid ground points found for coverage polygon (YS-52)")

    ix = np.floor(ground_pts[:, 0] / grid_cell_m).astype(np.int32)
    iy = np.floor(ground_pts[:, 1] / grid_cell_m).astype(np.int32)

    pad = closing_cells + 3
    min_ix, max_ix = int(np.min(ix)) - pad, int(np.max(ix)) + pad
    min_iy, max_iy = int(np.min(iy)) - pad, int(np.max(iy)) + pad
    w = max_ix - min_ix + 1
    h = max_iy - min_iy + 1

    occ_grid = np.zeros((h, w), dtype=bool)
    high_grid = np.zeros((h, w), dtype=bool)

    gx = ix - min_ix
    gy = iy - min_iy
    occ_grid[gy, gx] = True
    high_mask = ground_confs == 2
    high_grid[gy[high_mask], gx[high_mask]] = True

    # 1-cell morphological closing
    struct = np.ones((2 * closing_cells + 1, 2 * closing_cells + 1), dtype=bool)
    closed = binary_closing(occ_grid, structure=struct)

    # Largest connected component
    labeled, n_comp = label(closed)
    if n_comp == 0:
        raise ValueError("Empty occupancy grid after closing")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    region = labeled == largest_label

    # Identify interior holes: fill only small occlusion holes <= max_filled_hole_m2 (<= 25 cells)
    filled_all = binary_fill_holes(region)
    holes_mask = filled_all & (~region)
    cell_area_m2 = grid_cell_m * grid_cell_m
    max_hole_cells = int(round(max_filled_hole_m2 / cell_area_m2))

    hole_labels, n_holes = label(holes_mask)
    large_holes_mask = np.zeros_like(region)
    for h_id in range(1, n_holes + 1):
        h_comp = hole_labels == h_id
        if int(np.sum(h_comp)) <= max_hole_cells:
            region = region | h_comp
        else:
            large_holes_mask = large_holes_mask | h_comp

    # Extract outer contour of `filled_all` (the outer boundary of the largest region)
    outer_uint8 = (filled_all.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(outer_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Could not extract outer boundary contour")
    outer_contour = max(contours, key=cv2.contourArea)

    def _contour_to_xy(cnt: np.ndarray, eps_m: float, ccw: bool) -> list[list[float]]:
        pts_grid = cnt[:, 0, :].astype(np.float64)
        pts_m = np.column_stack(
            [
                (pts_grid[:, 0] + min_ix + 0.5) * grid_cell_m,
                (pts_grid[:, 1] + min_iy + 0.5) * grid_cell_m,
            ]
        )
        for factor in [1.0, 0.7, 0.5, 0.25, 0.1, 0.0]:
            eps = eps_m * factor
            approx = cv2.approxPolyDP(pts_m.astype(np.float32), epsilon=eps, closed=True)[:, 0, :].astype(np.float64)
            if approx.shape[0] < 3:
                continue
            area = _polygon_signed_area(approx)
            if (ccw and area < 0) or ((not ccw) and area > 0):
                approx = approx[::-1]
            verts = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in approx]
            if is_simple_polygon(verts):
                return verts
        # Box fallback if degenerate
        x0, y0 = float(np.min(pts_m[:, 0])), float(np.min(pts_m[:, 1]))
        x1, y1 = float(np.max(pts_m[:, 0])), float(np.max(pts_m[:, 1]))
        box = [[round(x0, 4), round(y0, 4)], [round(x1, 4), round(y0, 4)], [round(x1, 4), round(y1, 4)], [round(x0, 4), round(y1, 4)]]
        return box if ccw else box[::-1]

    outer_poly = _contour_to_xy(outer_contour, simplify_epsilon_m, ccw=True)
    outer_area = abs(_polygon_signed_area(np.array(outer_poly, dtype=np.float64)))

    holes_xy_m: list[list[list[float]]] = []
    holes_area = 0.0
    if np.any(large_holes_mask):
        h_uint8 = (large_holes_mask.astype(np.uint8)) * 255
        h_contours, _ = cv2.findContours(h_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for hc in h_contours:
            h_poly = _contour_to_xy(hc, simplify_epsilon_m, ccw=False)
            h_a = abs(_polygon_signed_area(np.array(h_poly, dtype=np.float64)))
            if h_a >= max_filled_hole_m2:
                holes_xy_m.append(h_poly)
                holes_area += h_a

    net_area = max(0.01, round(outer_area - holes_area, 4))
    occ_in_region = occ_grid & region
    total_occ = int(np.sum(occ_in_region))
    high_occ = int(np.sum(high_grid & region))
    high_share = round(float(high_occ) / float(total_occ), 4) if total_occ > 0 else 0.0

    return {
        "contract": "yardstick-capture/1",
        "polygon_xy_m": outer_poly,
        "holes_xy_m": holes_xy_m,
        "area_m2": net_area,
        "high_confidence_share": high_share,
        "parameters": {
            "z_max_abs_m": z_max_abs_m,
            "range_max_m": range_max_m,
            "grid_cell_m": grid_cell_m,
            "closing_cells": closing_cells,
            "max_filled_hole_m2": max_filled_hole_m2,
            "simplify_epsilon_m": simplify_epsilon_m,
        },
    }
