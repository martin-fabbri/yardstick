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

"""ArUco DICT_5X5_250 verification, single-frame LiDAR plane measurement, one-frame pair extraction,
and drift-free marker layout graph solver (YS-30, YS-33, YS-69, YS-115).
"""

from __future__ import annotations

import csv
import heapq
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from yardstick.geometry import pose_to_matrix, quat_to_rotmat, rotmat_to_quat

PAIR_CSV_COLUMNS = [
    "t",
    "saved_index",
    "marker_i",
    "marker_j",
    "dist_3d_m",
    "dist_horiz_m",
    "dz_gravity_m",
    "dx_grav_i_m",
    "dy_grav_i_m",
    "rel_yaw_ij_rad",
    "dx_cam_m",
    "dy_cam_m",
    "dz_cam_m",
    "range_i_m",
    "range_j_m",
    "angle_i_deg",
    "angle_j_deg",
    "size_i_mm_measured",
    "size_j_mm_measured",
    "sigma_dist_m",
    "sigma_yaw_rad",
]

MARKERS_CSV_BASE_COLUMNS = [
    "t",
    "saved_index",
    "marker_id",
    "size_mm_printed",
    "size_mm_measured",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "c0_u",
    "c0_v",
    "c1_u",
    "c1_v",
    "c2_u",
    "c2_v",
    "c3_u",
    "c3_v",
]

MARKERS_CSV_CAMERA_COLUMNS = [
    "cam_x",
    "cam_y",
    "cam_z",
    "cam_qx",
    "cam_qy",
    "cam_qz",
    "cam_qw",
    "range_m",
    "angle_deg",
    "edge_disagreement_ratio",
    "plane_rmse_m",
    "depth_sample_count",
]


def _wrap_angle(rad: float) -> float:
    """Wrap angle in radians to (-pi, pi]."""
    w = (rad + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if w == -math.pi else w


def check_refined_corner_shift(
    raw_corners_uv: np.ndarray,
    refined_corners_uv: np.ndarray,
    max_corner_shift_ratio: float = 0.03,
) -> bool:
    """Return True if every refined corner moved by <= max_corner_shift_ratio (3%) of the mean 2D edge length (YS-69).

    Includes a 2.2 px floor to accommodate integer-pixel contour quantization from CORNER_REFINE_NONE on small/far quads.
    """
    raw_q = np.asarray(raw_corners_uv, dtype=np.float64).reshape(4, 2)
    ref_q = np.asarray(refined_corners_uv, dtype=np.float64).reshape(4, 2)
    edges_px = [float(np.linalg.norm(raw_q[(i + 1) % 4] - raw_q[i])) for i in range(4)]
    mean_edge_px = float(np.mean(edges_px))
    if mean_edge_px <= 1e-6:
        return False
    floor_px = 2.5 if mean_edge_px < 65.0 else 2.2
    max_allowed_px = max(max_corner_shift_ratio * mean_edge_px, floor_px)
    shifts_px = [float(np.linalg.norm(ref_q[i] - raw_q[i])) for i in range(4)]
    return max(shifts_px) <= max_allowed_px


def detect_aruco_markers_opencv(
    image_bgr_or_gray: np.ndarray,
    *,
    max_corner_shift_ratio: float = 0.03,
) -> dict[int, np.ndarray]:
    """Detect ArUco DICT_5X5_250 markers using OpenCV with sub-pixel refinement (window 5) (YS-30, YS-69).

    Rejects any quad whose sub-pixel refined corners move by > 3% of the unrefined mean 2D edge length.
    Returns dict mapping marker_id -> 4x2 float64 corner coordinates (c0, c1, c2, c3).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    params_raw = cv2.aruco.DetectorParameters()
    params_raw.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector_raw = cv2.aruco.ArucoDetector(aruco_dict, params_raw)
    corners_raw, ids_raw, _ = detector_raw.detectMarkers(image_bgr_or_gray)
    raw_by_id: dict[int, np.ndarray] = {}
    if ids_raw is not None and corners_raw is not None:
        for c, mid_val in zip(corners_raw, np.asarray(ids_raw).ravel()):
            raw_by_id[int(mid_val)] = c.reshape(4, 2).astype(np.float64)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image_bgr_or_gray)
    detected: dict[int, np.ndarray] = {}
    if ids is not None and corners is not None:
        for c, mid_val in zip(corners, np.asarray(ids).ravel()):
            mid = int(mid_val)
            ref_q = c.reshape(4, 2).astype(np.float64)
            if mid in raw_by_id and not check_refined_corner_shift(
                raw_by_id[mid], ref_q, max_corner_shift_ratio=max_corner_shift_ratio
            ):
                continue
            detected[mid] = ref_q
    return detected


def measure_marker_3d_camera_frame(
    corners_uv: np.ndarray,
    depth_mm: np.ndarray,
    conf_img: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_width: int,
    rgb_height: int,
    T_world_from_cam: np.ndarray | None = None,
    *,
    inward_erosion_ratio: float | None = None,
    max_edge_disagreement_ratio: float = 0.04,
    min_range_m: float = 0.50,
    max_range_m: float = 3.50,
    max_angle_deg: float = 60.0,
    max_plane_rmse_m: float = 0.015,
    min_depth_samples: int = 10,
) -> dict[str, Any] | None:
    """Fit a 3D plane to interior LiDAR depth pixels in camera coordinates and intersect 4 RGB corner rays (YS-33, YS-69, YS-115).

    Uses 10% inward erosion at <= 2.20 m and 15% inward erosion at > 2.20 m (unless overridden),
    plus 1-step Huber/MAD outlier rejection on depth samples.
    """
    h_d, w_d = depth_mm.shape
    scale_x = float(w_d) / float(rgb_width)
    scale_y = float(h_d) / float(rgb_height)
    fx_d = fx * scale_x
    fy_d = fy * scale_y
    cx_d = cx * scale_x
    cy_d = cy * scale_y

    corners_d = np.column_stack([corners_uv[:, 0] * scale_x, corners_uv[:, 1] * scale_y])
    center_d = np.mean(corners_d, axis=0)

    eff_erosion = inward_erosion_ratio if inward_erosion_ratio is not None else 0.10
    eroded_d = center_d + (1.0 - eff_erosion) * (corners_d - center_d)

    mask = np.zeros((h_d, w_d), dtype=np.uint8)
    poly_int = np.rint(eroded_d).astype(np.int32)
    cv2.fillConvexPoly(mask, poly_int, 1)

    valid = (mask > 0) & (depth_mm > 0) & (conf_img >= 1)
    if int(np.sum(valid)) < max(10, min_depth_samples):
        for scale_f in (0.85, 1.0, 1.35):
            scaled_d = center_d + scale_f * (corners_d - center_d)
            mask_full = np.zeros((h_d, w_d), dtype=np.uint8)
            cv2.fillConvexPoly(mask_full, np.rint(scaled_d).astype(np.int32), 1)
            valid = (mask_full > 0) & (depth_mm > 0) & (conf_img >= 1)
            if int(np.sum(valid)) >= max(10, min_depth_samples):
                break
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
    if not (min_range_m <= range_m <= max_range_m):
        return None

    # If range > 2.20 m and caller did not override inward_erosion_ratio, re-erode by 15% if sufficient samples exist
    if inward_erosion_ratio is None and range_m > 2.20:
        eroded_15 = center_d + 0.85 * (corners_d - center_d)
        mask_15 = np.zeros((h_d, w_d), dtype=np.uint8)
        cv2.fillConvexPoly(mask_15, np.rint(eroded_15).astype(np.int32), 1)
        valid_15 = (mask_15 > 0) & (depth_mm > 0) & (conf_img >= 1)
        if int(np.sum(valid_15)) >= 10:
            v_idx, u_idx = np.nonzero(valid_15)
            z_val = depth_mm[v_idx, u_idx].astype(np.float64) / 1000.0
            x_cam = (u_idx.astype(np.float64) - cx_d) * z_val / fx_d
            y_cam = -(v_idx.astype(np.float64) - cy_d) * z_val / fy_d
            z_cam = -z_val
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
            centroid_cam = np.mean(pts_cam, axis=0)
            range_m = float(np.linalg.norm(centroid_cam))

    view_dir = -centroid_cam / max(range_m, 1e-9)
    plane_rmse_m = 0.0
    inlier_count = int(pts_cam.shape[0])

    if pts_cam.shape[0] < 6:
        return None

    centered = pts_cam - centroid_cam
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal_cam = vh[-1, :]
    if normal_cam[2] < 0:
        normal_cam = -normal_cam
    d_init = -float(np.dot(normal_cam, centroid_cam))
    dists = np.abs(pts_cam @ normal_cam + d_init)
    mad = float(np.median(dists))
    inliers = dists <= max(0.008, 2.5 * 1.4826 * mad)
    if int(np.sum(inliers)) >= 6:
        pts_in = pts_cam[inliers]
        centroid_cam = np.mean(pts_in, axis=0)
        _, _, vh2 = np.linalg.svd(pts_in - centroid_cam, full_matrices=False)
        normal_cam = vh2[-1, :]
        if normal_cam[2] < 0:
            normal_cam = -normal_cam
        d_in = -float(np.dot(normal_cam, centroid_cam))
        res_in = pts_in @ normal_cam + d_in
        plane_rmse_m = float(np.sqrt(np.mean(res_in * res_in)))
        inlier_count = int(pts_in.shape[0])
    else:
        plane_rmse_m = float(np.sqrt(np.mean(dists * dists)))

    if plane_rmse_m > max_plane_rmse_m:
        return None

    cos_inc = float(np.clip(abs(float(np.dot(normal_cam, view_dir))), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_inc))
    if angle_deg > max_angle_deg + 1e-6:
        return None

    d_cam = -float(np.dot(normal_cam, centroid_cam))

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
    mean_edge_m = float(np.mean(edges_m))
    if mean_edge_m <= 1e-9:
        return None
    edge_disagreement = (max(edges_m) - min(edges_m)) / mean_edge_m
    if max_edge_disagreement_ratio > 0.0 and edge_disagreement > max_edge_disagreement_ratio:
        return None

    frame_size_mm = 1000.0 * mean_edge_m
    center_cam = np.mean(corners_3d_cam, axis=0)
    range_center_m = float(np.linalg.norm(center_cam))

    # Local marker frame in camera coordinates: +X along top/bottom edge (c0->c1, c3->c2), +Z along normal_cam
    v_x = (corners_3d_cam[1] - corners_3d_cam[0]) + (corners_3d_cam[2] - corners_3d_cam[3])
    z_axis = normal_cam / max(float(np.linalg.norm(normal_cam)), 1e-12)
    x_proj = v_x - float(np.dot(v_x, z_axis)) * z_axis
    x_norm = float(np.linalg.norm(x_proj))
    if x_norm < 1e-9:
        return None
    x_axis = x_proj / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    R_cam_from_marker = np.column_stack([x_axis, y_axis, z_axis])
    quat_cam = rotmat_to_quat(R_cam_from_marker)

    if T_world_from_cam is not None:
        R_wc = T_world_from_cam[:3, :3]
        t_wc = T_world_from_cam[:3, 3]
        corners_3d_world = (corners_3d_cam @ R_wc.T) + t_wc
        center_world = (R_wc @ center_cam) + t_wc
    else:
        corners_3d_world = corners_3d_cam.copy()
        center_world = center_cam.copy()

    return {
        "corners_3d_cam": corners_3d_cam,
        "center_cam": center_cam,
        "R_cam_from_marker": R_cam_from_marker,
        "quat_cam": quat_cam,
        "size_mm_measured": frame_size_mm,
        "range_m": range_center_m,
        "angle_deg": angle_deg,
        "edge_disagreement_ratio": edge_disagreement,
        "plane_rmse_m": plane_rmse_m,
        "depth_sample_count": inlier_count,
        "corners_3d_world": corners_3d_world,
        "center_world": center_world,
    }


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
    max_edge_disagreement_ratio: float = 0.04,
    max_range_m: float = 3.50,
) -> tuple[np.ndarray, float, float] | None:
    """Project 4 RGB marker corners onto the 3D plane fitted to interior LiDAR depth pixels (YS-33, Q027, YS-69).

    Applies 10% inward border erosion on the 2D marker quadrilateral before sampling depth pixels.
    Rejects any quad whose 3D edge lengths disagree by > 4% ((max - min) / mean > 0.04).
    Returns (corners_3d_world (4x3), frame_size_mm, range_m) or None if outside pass criteria.
    """
    info = measure_marker_3d_camera_frame(
        corners_uv=corners_uv,
        depth_mm=depth_mm,
        conf_img=conf_img,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        T_world_from_cam=T_world_from_cam,
        inward_erosion_ratio=inward_erosion_ratio,
        max_edge_disagreement_ratio=max_edge_disagreement_ratio,
        min_range_m=0.50,
        max_range_m=max_range_m,
    )
    if info is None:
        return None
    return info["corners_3d_world"], float(info["size_mm_measured"]), float(info["range_m"])


def evaluate_marker_scale_errors(
    capture_dir: Path,
    markers_rows: list[dict[str, Any]],
    poses_by_index: dict[int, dict[str, Any]],
    passes_by_marker: dict[int, list[list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Compute single-frame camera-space marker scale error and printed-size scale factor (YS-33, YS-69, YS-115).

    Per YS-115 Condition 1 & 3:
    - Evaluates marker size in single-frame camera coordinates (never averaging corners in world space,
      which would shrink under mid-pass VIO motion).
    - Measures size from whatever valid frames see the marker (no close-range solo lock required).
    - Computes `printed_scale_factor` (printed / measured) and `printed_scale_se` for pair-edge scale correction.
    """
    if not markers_rows:
        return {
            "scale_error_ratio": 0.0,
            "scale_error_se": 0.0,
            "scale_error_eff": 0.0,
            "sightings_used": 0,
            "per_marker": [],
            "printed_scale_factor": None,
            "printed_scale_se": None,
        }

    stills_by_idx: dict[int, str] = {}
    stills_csv_path = capture_dir / "stills.csv"
    if stills_csv_path.exists():
        try:
            with stills_csv_path.open("r", encoding="utf-8", newline="") as sf:
                for srow in csv.DictReader(sf):
                    stills_by_idx[int(srow["saved_index"])] = srow["file"]
        except Exception:
            stills_by_idx = {}

    still_detections_cache: dict[int, dict[int, np.ndarray]] = {}

    def _get_still_detections(s_idx: int) -> dict[int, np.ndarray] | None:
        if s_idx not in stills_by_idx:
            return None
        if s_idx in still_detections_cache:
            return still_detections_cache[s_idx]
        still_fp = capture_dir / stills_by_idx[s_idx]
        if not still_fp.exists():
            return None
        try:
            with Image.open(still_fp) as s_img:
                gray = np.array(s_img.convert("L"), dtype=np.uint8)
            dets = detect_aruco_markers_opencv(gray)
        except Exception:
            dets = {}
        still_detections_cache[s_idx] = dets
        return dets

    per_marker_results: list[dict[str, Any]] = []
    all_pass_errors: list[float] = []
    all_pass_ses: list[float] = []
    all_alpha_samples: list[float] = []
    total_sightings_used = 0

    for mid, m_passes in sorted(passes_by_marker.items()):
        printed_sizes: list[float] = []
        frame_measured_sizes: list[float] = []
        pass_median_sizes: list[float] = []

        for m_pass in m_passes:
            pass_frame_sizes: list[float] = []
            for row in m_pass:
                s_idx = int(row["saved_index"])
                printed_mm = float(row["size_mm_printed"])

                depth_path = capture_dir / "depth" / f"{s_idx:06d}.png"
                conf_path = capture_dir / "confidence" / f"{s_idx:06d}.png"
                pose_row = poses_by_index.get(s_idx)

                still_dets = _get_still_detections(s_idx)
                if still_dets is not None and int(mid) not in still_dets:
                    continue

                has_logged_corners = all(
                    k in row and row[k] != ""
                    for k in ("c0_u", "c0_v", "c1_u", "c1_v", "c2_u", "c2_v", "c3_u", "c3_v")
                )
                if has_logged_corners:
                    corners_uv: np.ndarray | None = np.array(
                        [
                            [float(row["c0_u"]), float(row["c0_v"])],
                            [float(row["c1_u"]), float(row["c1_v"])],
                            [float(row["c2_u"]), float(row["c2_v"])],
                            [float(row["c3_u"]), float(row["c3_v"])],
                        ],
                        dtype=np.float64,
                    )
                    has_corners = True
                elif still_dets is not None and int(mid) in still_dets:
                    corners_uv = still_dets[int(mid)]
                    has_corners = True
                else:
                    corners_uv = None
                    has_corners = False

                if has_corners and corners_uv is not None and depth_path.exists() and conf_path.exists() and pose_row is not None:
                    try:
                        with Image.open(depth_path) as d_img:
                            depth_mm = np.array(d_img, dtype=np.uint16)
                        with Image.open(conf_path) as c_img:
                            conf_img = np.array(c_img, dtype=np.uint8)
                        T_wc = pose_to_matrix(
                            float(pose_row["px"]),
                            float(pose_row["py"]),
                            float(pose_row["pz"]),
                            float(pose_row["qx"]),
                            float(pose_row["qy"]),
                            float(pose_row["qz"]),
                            float(pose_row["qw"]),
                        )
                        info = measure_marker_3d_camera_frame(
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
                            max_edge_disagreement_ratio=0.06,
                            max_range_m=4.20,
                        )
                    except Exception:
                        info = None
                    if info is not None:
                        f_size_mm = float(info["size_mm_measured"])
                        pass_frame_sizes.append(f_size_mm)
                        frame_measured_sizes.append(f_size_mm)
                        printed_sizes.append(printed_mm)
                        if f_size_mm > 1e-3:
                            all_alpha_samples.append(printed_mm / f_size_mm)
                    continue

                logged_mm = float(row["size_mm_measured"])
                if logged_mm > 0:
                    pass_frame_sizes.append(logged_mm)
                    frame_measured_sizes.append(logged_mm)
                    printed_sizes.append(printed_mm)
                    all_alpha_samples.append(printed_mm / logged_mm)

            if pass_frame_sizes:
                # Single-frame camera-space median across the pass (100% invariant to mid-pass VIO motion)
                pass_median_sizes.append(float(np.median(pass_frame_sizes)))

        if not frame_measured_sizes:
            continue

        printed_ref = float(np.median(printed_sizes))
        if pass_median_sizes:
            measured_mm = float(np.mean(pass_median_sizes))
        else:
            measured_mm = float(np.mean(frame_measured_sizes))

        err_ratio = (measured_mm - printed_ref) / printed_ref
        frame_errs = [(s - printed_ref) / printed_ref for s in frame_measured_sizes]
        n_used = len(frame_errs)
        if n_used >= 2:
            se = float(np.std(frame_errs, ddof=1) / math.sqrt(n_used))
        else:
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
            "printed_scale_factor": None,
            "printed_scale_se": None,
        }

    idx_med = int(np.argmax([abs(e) for e in all_pass_errors]))
    agg_err = float(all_pass_errors[idx_med])
    agg_se = float(all_pass_ses[idx_med])
    agg_eff = max(0.0, abs(agg_err) - 2.0 * agg_se)

    if all_alpha_samples:
        printed_scale_factor = float(np.median(all_alpha_samples))
        if len(all_alpha_samples) >= 2:
            printed_scale_se = float(np.std(all_alpha_samples, ddof=1) / math.sqrt(len(all_alpha_samples)))
        else:
            printed_scale_se = 0.005
    else:
        printed_scale_factor = None
        printed_scale_se = None

    return {
        "scale_error_ratio": round(agg_err, 6),
        "scale_error_se": round(agg_se, 6),
        "scale_error_eff": round(agg_eff, 6),
        "sightings_used": total_sightings_used,
        "per_marker": per_marker_results,
        "printed_scale_factor": round(printed_scale_factor, 6) if printed_scale_factor is not None else None,
        "printed_scale_se": round(printed_scale_se, 6) if printed_scale_se is not None else None,
    }


def measure_marker_3d_from_ground_plane_and_printed_size(
    corners_uv: np.ndarray,
    printed_size_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_world_from_cam: np.ndarray | None = None,
    reference_ground_normal_cam: np.ndarray | None = None,
    min_range_m: float = 1.20,
    max_range_m: float = 4.20,
    max_angle_deg: float = 78.0,
) -> dict[str, Any] | None:
    """Estimate a 3D marker pose in camera/world space from 2D corners, ground plane normal, and printed size (YS-115 C5/C6).

    Used when a far marker in a multi-marker frame (`2-4 m`) is detected in 2D (`seen_2d_no_3d`)
    but fails the LiDAR depth plane fit. Records explicit position uncertainty (`position_uncertainty_m`).
    """
    q = np.asarray(corners_uv, dtype=np.float64).reshape(4, 2)
    printed_m = float(printed_size_mm) * 0.001
    if printed_m <= 0.0 or fx <= 0.0 or fy <= 0.0:
        return None

    center_uv = np.mean(q, axis=0)
    ray_center = np.array([(float(center_uv[0]) - cx) / fx, -(float(center_uv[1]) - cy) / fy, -1.0], dtype=np.float64)
    ray_center = ray_center / max(float(np.linalg.norm(ray_center)), 1e-12)
    view_dir = -ray_center

    R_wc = T_world_from_cam[:3, :3] if T_world_from_cam is not None else np.eye(3, dtype=np.float64)
    n_grav_cam = R_wc.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n_grav_cam = n_grav_cam / max(float(np.linalg.norm(n_grav_cam)), 1e-12)
    if float(np.dot(n_grav_cam, view_dir)) < 0.0:
        n_grav_cam = -n_grav_cam

    if reference_ground_normal_cam is not None:
        ref_n = np.asarray(reference_ground_normal_cam, dtype=np.float64)
        ref_n = ref_n / max(float(np.linalg.norm(ref_n)), 1e-12)
        if float(np.dot(ref_n, view_dir)) < 0.0:
            ref_n = -ref_n
        if float(np.dot(n_grav_cam, ref_n)) >= 0.95:
            n_cam = n_grav_cam + ref_n
            n_cam = n_cam / max(float(np.linalg.norm(n_cam)), 1e-12)
        else:
            n_cam = n_grav_cam
    else:
        n_cam = n_grav_cam

    cos_a = float(np.clip(abs(float(np.dot(n_cam, view_dir))), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_a))
    if angle_deg > max_angle_deg:
        return None

    denom_center = float(np.dot(n_cam, ray_center))
    if abs(denom_center) <= 0.12:
        return None

    unit_pts: list[np.ndarray] = []
    for u, v in q:
        r = np.array([(float(u) - cx) / fx, -(float(v) - cy) / fy, -1.0], dtype=np.float64)
        r = r / max(float(np.linalg.norm(r)), 1e-12)
        denom = float(np.dot(n_cam, r))
        if abs(denom) <= 0.08:
            return None
        t_unit = denom_center / denom
        if t_unit <= 0.0:
            return None
        unit_pts.append(r * t_unit)

    unit_edges = [float(np.linalg.norm(unit_pts[(i + 1) % 4] - unit_pts[i])) for i in range(4)]
    mean_unit_edge = float(np.mean(unit_edges))
    if mean_unit_edge <= 1e-6:
        return None
    edge_disagreement = float(max(abs(e - mean_unit_edge) for e in unit_edges) / mean_unit_edge)

    h_scale = printed_m / mean_unit_edge
    if not (min_range_m <= h_scale <= max_range_m):
        return None
    corners_cam = [p * h_scale for p in unit_pts]
    center_cam = np.mean(np.stack(corners_cam, axis=0), axis=0)
    range_m = float(np.linalg.norm(center_cam))
    if not (min_range_m <= range_m <= max_range_m):
        return None

    v01 = corners_cam[1] - corners_cam[0]
    v32 = corners_cam[2] - corners_cam[3]
    x_raw = 0.5 * (v01 + v32)
    x_proj = x_raw - float(np.dot(x_raw, n_cam)) * n_cam
    n_x = float(np.linalg.norm(x_proj))
    if n_x <= 1e-9:
        return None
    x_axis = x_proj / n_x
    y_axis = np.cross(n_cam, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    R_cm = np.column_stack([x_axis, y_axis, n_cam])

    edges_px = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    mean_edge_px = max(10.0, float(np.mean(edges_px)))
    sigma_pixel_m = range_m * (1.2 / mean_edge_px)
    sigma_slope_m = range_m * 0.025 * math.tan(math.radians(min(angle_deg, 65.0)))
    uncertainty_m = max(0.035, math.hypot(sigma_pixel_m, sigma_slope_m))

    res: dict[str, Any] = {
        "center_cam": center_cam,
        "R_cam_from_marker": R_cm,
        "corners_cam": np.stack(corners_cam, axis=0),
        "size_mm_measured": float(printed_size_mm),
        "size_mm_printed": float(printed_size_mm),
        "range_m": range_m,
        "angle_deg": angle_deg,
        "edge_disagreement_ratio": edge_disagreement,
        "plane_rmse_m": 0.0,
        "depth_sample_count": 0,
        "position_uncertainty_m": uncertainty_m,
        "is_ground_plane_printed_size_fallback": True,
    }
    if T_world_from_cam is not None:
        t_wc = T_world_from_cam[:3, 3]
        res["center_world"] = (R_wc @ center_cam) + t_wc
        res["R_world_from_marker"] = R_wc @ R_cm
    return res


def compute_single_frame_pair_measurement(
    t_s: float,
    saved_index: int,
    marker_i: int,
    marker_j: int,
    p_i_cam: np.ndarray,
    R_cam_from_i: np.ndarray,
    size_i_mm: float,
    angle_i_deg: float,
    p_j_cam: np.ndarray,
    R_cam_from_j: np.ndarray,
    size_j_mm: float,
    angle_j_deg: float,
    gravity_up_cam: np.ndarray,
    position_uncertainty_i_m: float = 0.0,
    position_uncertainty_j_m: float = 0.0,
) -> dict[str, Any]:
    """Compute one-frame pair baseline in camera space and Marker i's gravity-leveled frame (YS-115 §2.1)."""
    if marker_i > marker_j:
        marker_i, marker_j = marker_j, marker_i
        p_i_cam, p_j_cam = p_j_cam, p_i_cam
        R_cam_from_i, R_cam_from_j = R_cam_from_j, R_cam_from_i
        size_i_mm, size_j_mm = size_j_mm, size_i_mm
        angle_i_deg, angle_j_deg = angle_j_deg, angle_i_deg
        position_uncertainty_i_m, position_uncertainty_j_m = (
            position_uncertainty_j_m,
            position_uncertainty_i_m,
        )

    g_cam = np.asarray(gravity_up_cam, dtype=np.float64)
    g_cam = g_cam / max(float(np.linalg.norm(g_cam)), 1e-12)

    dp_cam = np.asarray(p_j_cam, dtype=np.float64) - np.asarray(p_i_cam, dtype=np.float64)
    dist_3d_m = float(np.linalg.norm(dp_cam))
    dz_gravity_m = float(np.dot(g_cam, dp_cam))
    dist_horiz_m = math.sqrt(max(0.0, dist_3d_m * dist_3d_m - dz_gravity_m * dz_gravity_m))

    # Project Marker i and Marker j +X axes onto the plane orthogonal to single-frame gravity g_cam
    x_i_cam = R_cam_from_i[:, 0]
    x_i_proj = x_i_cam - float(np.dot(x_i_cam, g_cam)) * g_cam
    n_xi = float(np.linalg.norm(x_i_proj))
    if n_xi < 1e-9:
        x_i_proj = R_cam_from_i[:, 1] - float(np.dot(R_cam_from_i[:, 1], g_cam)) * g_cam
        n_xi = max(float(np.linalg.norm(x_i_proj)), 1e-9)
    x_i_grav = x_i_proj / n_xi
    y_i_grav = np.cross(g_cam, x_i_grav)
    y_i_grav = y_i_grav / max(float(np.linalg.norm(y_i_grav)), 1e-12)

    x_j_cam = R_cam_from_j[:, 0]
    x_j_proj = x_j_cam - float(np.dot(x_j_cam, g_cam)) * g_cam
    n_xj = float(np.linalg.norm(x_j_proj))
    if n_xj < 1e-9:
        x_j_proj = R_cam_from_j[:, 1] - float(np.dot(R_cam_from_j[:, 1], g_cam)) * g_cam
        n_xj = max(float(np.linalg.norm(x_j_proj)), 1e-9)
    x_j_grav = x_j_proj / n_xj

    dx_grav_i_m = float(np.dot(x_i_grav, dp_cam))
    dy_grav_i_m = float(np.dot(y_i_grav, dp_cam))
    rel_yaw_ij_rad = math.atan2(float(np.dot(y_i_grav, x_j_grav)), float(np.dot(x_i_grav, x_j_grav)))

    range_i_m = float(np.linalg.norm(p_i_cam))
    range_j_m = float(np.linalg.norm(p_j_cam))
    mean_range = 0.5 * (range_i_m + range_j_m)
    max_angle = max(angle_i_deg, angle_j_deg)
    base_sigma_dist = max(0.004, 0.003 + 0.0025 * mean_range + 0.002 * math.tan(math.radians(min(max_angle, 65.0))))
    extra_sigma = math.hypot(float(position_uncertainty_i_m), float(position_uncertainty_j_m))
    sigma_dist_m = max(base_sigma_dist, extra_sigma)
    sigma_yaw_rad = max(0.008, 0.006 + 0.004 * mean_range)

    return {
        "t": round(float(t_s), 6),
        "saved_index": int(saved_index),
        "marker_i": int(marker_i),
        "marker_j": int(marker_j),
        "dist_3d_m": round(dist_3d_m, 6),
        "dist_horiz_m": round(dist_horiz_m, 6),
        "dz_gravity_m": round(dz_gravity_m, 6),
        "dx_grav_i_m": round(dx_grav_i_m, 6),
        "dy_grav_i_m": round(dy_grav_i_m, 6),
        "rel_yaw_ij_rad": round(rel_yaw_ij_rad, 6),
        "dx_cam_m": round(float(dp_cam[0]), 6),
        "dy_cam_m": round(float(dp_cam[1]), 6),
        "dz_cam_m": round(float(dp_cam[2]), 6),
        "range_i_m": round(range_i_m, 4),
        "range_j_m": round(range_j_m, 4),
        "angle_i_deg": round(float(angle_i_deg), 2),
        "angle_j_deg": round(float(angle_j_deg), 2),
        "size_i_mm_measured": round(float(size_i_mm), 3),
        "size_j_mm_measured": round(float(size_j_mm), 3),
        "sigma_dist_m": round(sigma_dist_m, 6),
        "sigma_yaw_rad": round(sigma_yaw_rad, 6),
    }


def resolve_marker_camera_pose(
    capture_dir: Path,
    row: dict[str, Any],
    pose_row: dict[str, Any] | None,
    depth_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any] | None:
    """Resolve single-frame camera-space 3D marker pose (p_cam, R_cam_from_marker, size_mm, range_m, angle_deg)."""
    printed_mm = float(row.get("size_mm_printed", 180.0))
    # 1. Explicit camera-space columns in markers.csv
    if all(row.get(c, "") != "" for c in ("cam_x", "cam_y", "cam_z", "cam_qx", "cam_qy", "cam_qz", "cam_qw")):
        p_cam = np.array([float(row["cam_x"]), float(row["cam_y"]), float(row["cam_z"])], dtype=np.float64)
        R_cm = quat_to_rotmat(
            float(row["cam_qx"]),
            float(row["cam_qy"]),
            float(row["cam_qz"]),
            float(row["cam_qw"]),
        )
        range_m = float(row["range_m"]) if row.get("range_m", "") != "" else float(np.linalg.norm(p_cam))
        if row.get("angle_deg", "") != "":
            angle_deg = float(row["angle_deg"])
        else:
            view_dir = -p_cam / max(range_m, 1e-9)
            cos_a = float(np.clip(abs(float(np.dot(R_cam_from_marker_z := R_cm[:, 2], view_dir))), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cos_a))
        size_mm = float(row.get("size_mm_measured", printed_mm))
        return {
            "center_cam": p_cam,
            "R_cam_from_marker": R_cm,
            "size_mm_measured": size_mm,
            "size_mm_printed": printed_mm,
            "range_m": range_m,
            "angle_deg": angle_deg,
        }

    s_idx = int(row["saved_index"])
    # 2. Measure directly on single-frame LiDAR depth plane if corners and depth map exist
    has_corners = all(
        row.get(k, "") != "" for k in ("c0_u", "c0_v", "c1_u", "c1_v", "c2_u", "c2_v", "c3_u", "c3_v")
    )
    if has_corners and pose_row is not None:
        depth_mm: np.ndarray | None = None
        conf_img: np.ndarray | None = None
        if depth_cache is not None and s_idx in depth_cache:
            depth_mm, conf_img = depth_cache[s_idx]
        else:
            dp = capture_dir / "depth" / f"{s_idx:06d}.png"
            cp = capture_dir / "confidence" / f"{s_idx:06d}.png"
            if dp.exists() and cp.exists():
                try:
                    with Image.open(dp) as d_i:
                        depth_mm = np.array(d_i, dtype=np.uint16)
                    with Image.open(cp) as c_i:
                        conf_img = np.array(c_i, dtype=np.uint8)
                    if depth_cache is not None:
                        depth_cache[s_idx] = (depth_mm, conf_img)
                except Exception:
                    depth_mm, conf_img = None, None
        if depth_mm is not None and conf_img is not None:
            corners_uv = np.array(
                [
                    [float(row["c0_u"]), float(row["c0_v"])],
                    [float(row["c1_u"]), float(row["c1_v"])],
                    [float(row["c2_u"]), float(row["c2_v"])],
                    [float(row["c3_u"]), float(row["c3_v"])],
                ],
                dtype=np.float64,
            )
            info = measure_marker_3d_camera_frame(
                corners_uv=corners_uv,
                depth_mm=depth_mm,
                conf_img=conf_img,
                fx=float(pose_row["fx"]),
                fy=float(pose_row["fy"]),
                cx=float(pose_row["cx"]),
                cy=float(pose_row["cy"]),
                rgb_width=int(pose_row["width"]),
                rgb_height=int(pose_row["height"]),
                T_world_from_cam=None,
                max_edge_disagreement_ratio=0.06,
                min_range_m=0.50,
                max_range_m=4.20,
                max_angle_deg=60.0,
                max_plane_rmse_m=0.015,
            )
            if info is not None:
                info["size_mm_printed"] = printed_mm
                return info

    # 3. Fallback: invert same-frame T_world_from_cam (t_wc cancels out between two markers in the same frame)
    if pose_row is not None and all(row.get(k, "") != "" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")):
        T_wc = pose_to_matrix(
            float(pose_row["px"]),
            float(pose_row["py"]),
            float(pose_row["pz"]),
            float(pose_row["qx"]),
            float(pose_row["qy"]),
            float(pose_row["qz"]),
            float(pose_row["qw"]),
        )
        T_wm = pose_to_matrix(
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"]),
        )
        T_cm = np.linalg.inv(T_wc) @ T_wm
        p_cam = T_cm[:3, 3]
        R_cm = T_cm[:3, :3]
        range_m = float(np.linalg.norm(p_cam))
        view_dir = -p_cam / max(range_m, 1e-9)
        cos_a = float(np.clip(abs(float(np.dot(R_cm[:, 2], view_dir))), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_a))
        return {
            "center_cam": p_cam,
            "R_cam_from_marker": R_cm,
            "size_mm_measured": float(row.get("size_mm_measured", printed_mm)),
            "size_mm_printed": printed_mm,
            "range_m": range_m,
            "angle_deg": angle_deg,
            "position_uncertainty_m": float(row.get("position_uncertainty_m", 0.0) or 0.0),
        }

    # 4. Ground-plane + printed-size fallback for far 2D sightings (`seen_2d_no_3d`, YS-115 C5/C6)
    if has_corners and pose_row is not None:
        T_wc = pose_to_matrix(
            float(pose_row["px"]),
            float(pose_row["py"]),
            float(pose_row["pz"]),
            float(pose_row["qx"]),
            float(pose_row["qy"]),
            float(pose_row["qz"]),
            float(pose_row["qw"]),
        )
        corners_uv = np.array(
            [
                [float(row["c0_u"]), float(row["c0_v"])],
                [float(row["c1_u"]), float(row["c1_v"])],
                [float(row["c2_u"]), float(row["c2_v"])],
                [float(row["c3_u"]), float(row["c3_v"])],
            ],
            dtype=np.float64,
        )
        return measure_marker_3d_from_ground_plane_and_printed_size(
            corners_uv=corners_uv,
            printed_size_mm=printed_mm,
            fx=float(pose_row["fx"]),
            fy=float(pose_row["fy"]),
            cx=float(pose_row["cx"]),
            cy=float(pose_row["cy"]),
            T_world_from_cam=T_wc,
        )
    return None


def extract_one_frame_pairs(
    capture_dir: Path,
    markers_rows: list[dict[str, Any]],
    poses_by_index: dict[int, dict[str, Any]],
    ground_normal_world: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Load `pairs.csv` if present and/or extract one-frame neighbour pair measurements from `markers.csv` (YS-115)."""
    up_world = (
        ground_normal_world / max(float(np.linalg.norm(ground_normal_world)), 1e-12)
        if ground_normal_world is not None
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )

    pair_rows: list[dict[str, Any]] = []
    seen_frame_pairs: set[tuple[int, int, int]] = set()

    pairs_csv_path = capture_dir / "pairs.csv"
    if pairs_csv_path.exists():
        with pairs_csv_path.open("r", encoding="utf-8", newline="") as pf:
            for r in csv.DictReader(pf):
                s_idx = int(r["saved_index"])
                mi = int(r.get("marker_i") or r["id_a"])
                mj = int(r.get("marker_j") or r["id_b"])
                if mi > mj:
                    mi, mj = mj, mi
                seen_frame_pairs.add((s_idx, mi, mj))
                dist_3d = float(r.get("dist_3d_m") or r["dist_m"])
                dx_c = float(r["dx_cam_m"])
                dy_c = float(r["dy_cam_m"])
                dz_c = float(r["dz_cam_m"])
                dist_h = float(r.get("dist_horiz_m") or math.hypot(dx_c, dy_c))
                dz_g = float(r.get("dz_gravity_m") or dz_c)
                dx_g = float(r.get("dx_grav_i_m") or dx_c)
                dy_g = float(r.get("dy_grav_i_m") or dy_c)
                rel_yaw = float(r.get("rel_yaw_ij_rad") or r["d_yaw_rad"])
                rng_i = float(r.get("range_i_m") or r["range_a_m"])
                rng_j = float(r.get("range_j_m") or r["range_b_m"])
                ang_i = float(r.get("angle_i_deg") or r["angle_a_deg"])
                ang_j = float(r.get("angle_j_deg") or r["angle_b_deg"])
                w_val = float(r.get("weight") or 0.0)
                sig_d = (
                    float(r["sigma_dist_m"])
                    if r.get("sigma_dist_m") not in (None, "")
                    else (1.0 / math.sqrt(max(w_val, 1e-6)) if w_val > 0 else 0.015)
                )
                sig_y = (
                    float(r["sigma_yaw_rad"])
                    if r.get("sigma_yaw_rad") not in (None, "")
                    else max(0.008, 0.006 + 0.004 * 0.5 * (rng_i + rng_j))
                )
                pair_rows.append(
                    {
                        "t": float(r["t"]),
                        "saved_index": s_idx,
                        "marker_i": mi,
                        "marker_j": mj,
                        "dist_3d_m": dist_3d,
                        "dist_horiz_m": dist_h,
                        "dz_gravity_m": dz_g,
                        "dx_grav_i_m": dx_g,
                        "dy_grav_i_m": dy_g,
                        "rel_yaw_ij_rad": rel_yaw,
                        "dx_cam_m": dx_c,
                        "dy_cam_m": dy_c,
                        "dz_cam_m": dz_c,
                        "range_i_m": rng_i,
                        "range_j_m": rng_j,
                        "angle_i_deg": ang_i,
                        "angle_j_deg": ang_j,
                        "size_i_mm_measured": float(r.get("size_i_mm_measured") or 180.0),
                        "size_j_mm_measured": float(r.get("size_j_mm_measured") or 180.0),
                        "sigma_dist_m": sig_d,
                        "sigma_yaw_rad": sig_y,
                    }
                )

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for mrow in markers_rows:
        by_frame.setdefault(int(mrow["saved_index"]), []).append(mrow)

    depth_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for s_idx, f_rows in sorted(by_frame.items()):
        if len(f_rows) < 2:
            continue
        pose_row = poses_by_index.get(s_idx)
        if pose_row is not None:
            R_wc = quat_to_rotmat(
                float(pose_row["qx"]),
                float(pose_row["qy"]),
                float(pose_row["qz"]),
                float(pose_row["qw"]),
            )
            gravity_cam = R_wc.T @ up_world
        else:
            gravity_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        resolved_by_id: dict[int, tuple[dict[str, Any], float]] = {}
        for r in f_rows:
            mid = int(r["marker_id"])
            info = resolve_marker_camera_pose(capture_dir, r, pose_row, depth_cache=depth_cache)
            if info is None:
                continue
            rng = float(info["range_m"])
            ang = float(info["angle_deg"])
            s_meas = float(info["size_mm_measured"])
            s_print = float(info["size_mm_printed"])
            # One-Frame Pair Gate (§2.3): range in [0.50, 4.20] m, angle <= 60 deg, size ratio in [0.88, 1.12]
            if not (0.50 <= rng <= 4.20):
                continue
            if ang > 60.0 + 1e-6:
                continue
            if s_print > 0 and not (0.88 <= (s_meas / s_print) <= 1.12):
                continue
            resolved_by_id[mid] = (info, float(r["t"]))

        mids = sorted(resolved_by_id.keys())
        for idx_a in range(len(mids)):
            for idx_b in range(idx_a + 1, len(mids)):
                mi, mj = mids[idx_a], mids[idx_b]
                if (s_idx, mi, mj) in seen_frame_pairs:
                    continue
                info_i, t_i = resolved_by_id[mi]
                info_j, _ = resolved_by_id[mj]
                meas = compute_single_frame_pair_measurement(
                    t_s=t_i,
                    saved_index=s_idx,
                    marker_i=mi,
                    marker_j=mj,
                    p_i_cam=info_i["center_cam"],
                    R_cam_from_i=info_i["R_cam_from_marker"],
                    size_i_mm=info_i["size_mm_measured"],
                    angle_i_deg=info_i["angle_deg"],
                    p_j_cam=info_j["center_cam"],
                    R_cam_from_j=info_j["R_cam_from_marker"],
                    size_j_mm=info_j["size_mm_measured"],
                    angle_j_deg=info_j["angle_deg"],
                    gravity_up_cam=gravity_cam,
                    position_uncertainty_i_m=float(info_i.get("position_uncertainty_m", 0.0) or 0.0),
                    position_uncertainty_j_m=float(info_j.get("position_uncertainty_m", 0.0) or 0.0),
                )
                seen_frame_pairs.add((s_idx, mi, mj))
                pair_rows.append(meas)

    pair_rows.sort(key=lambda r: (int(r["saved_index"]), int(r["marker_i"]), int(r["marker_j"])))
    return pair_rows


def _fit_2d_similarity(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    anchor_origin: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit 2D similarity transform dst_xy ~ scale * (src_xy @ R2.T) + t2."""
    if src_xy.shape[0] < 2:
        return 1.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    if anchor_origin:
        # Marker 0 is at (0, 0) in both src and dst
        s_c = src_xy.copy()
        d_c = dst_xy.copy()
        mu_s = np.zeros(2, dtype=np.float64)
        mu_d = np.zeros(2, dtype=np.float64)
    else:
        mu_s = np.mean(src_xy, axis=0)
        mu_d = np.mean(dst_xy, axis=0)
        s_c = src_xy - mu_s
        d_c = dst_xy - mu_d

    denom = float(np.sum(s_c * s_c))
    if denom < 1e-9:
        return 1.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    H = s_c.T @ d_c
    U, S_vals, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1.0
        R2 = Vt.T @ U.T

    scale = float(np.sum(S_vals)) / denom
    scale = float(np.clip(scale, 0.85, 1.15))
    t2 = mu_d - scale * (R2 @ mu_s)
    return scale, R2, t2


def solve_marker_layout_graph(
    markers_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    passes_by_marker: dict[int, list[list[dict[str, Any]]]],
    T_yard_from_world: np.ndarray,
    origin_height_m: float = 0.0,
    marker_scale_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve marker layout graph G = (V, E) from one-frame pair measurements without inter-frame VIO (YS-115 §3).

    Implements:
    - Stage 1: MAD outlier rejection + inverse-variance weighted per-edge aggregation
    - Condition 3: Applies printed-size scale correction (`scale_applied`) and propagates its uncertainty
    - Stage 2: Dijkstra shortest-uncertainty spanning tree rooted at Marker 0
    - Stage 3: Damped Gauss-Newton (Levenberg-Marquardt) loop closure with Huber loss + covariance
    - Disconnected-graph fallback via 2D VIO similarity transform or ARKit world fallback
    """
    R_yw = T_yard_from_world[:3, :3]
    t_yw = T_yard_from_world[:3, 3]

    # Collect seen markers and their raw ARKit open-loop Yard coordinates (from first pass of each marker)
    seen_ids = sorted(passes_by_marker.keys())
    raw_xyz_yard: dict[int, np.ndarray] = {}
    raw_normal_yard: dict[int, np.ndarray] = {}
    raw_yaw_yard: dict[int, float] = {}
    sightings_count_by_id: dict[int, int] = {}

    for mid in seen_ids:
        m_passes = passes_by_marker[mid]
        all_rows = [r for mp in m_passes for r in mp]
        sightings_count_by_id[mid] = len(all_rows)
        first_pass = m_passes[0]
        p_w = np.median(
            np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in first_pass], dtype=np.float64),
            axis=0,
        )
        p_y = (R_yw @ p_w) + t_yw
        raw_xyz_yard[mid] = p_y

        x_axes_y = []
        z_axes_y = []
        for r in first_pass:
            R_wm = quat_to_rotmat(float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"]))
            R_ym = R_yw @ R_wm
            x_axes_y.append(R_ym[:, 0])
            z_axes_y.append(R_ym[:, 2])
        mean_x = np.mean(np.stack(x_axes_y, axis=0), axis=0)
        mean_z = np.mean(np.stack(z_axes_y, axis=0), axis=0)
        mean_z = mean_z / max(float(np.linalg.norm(mean_z)), 1e-12)
        raw_normal_yard[mid] = mean_z
        raw_yaw_yard[mid] = 0.0 if mid == 0 else math.atan2(float(mean_x[1]), float(mean_x[0]))

    # Determine printed-size scale factor (Condition 3)
    printed_scale_factor: float | None = None
    printed_scale_se: float | None = None
    scale_applied = 1.0
    printed_size_mismatch = False

    if marker_scale_info is not None and int(marker_scale_info.get("sightings_used", 0)) >= 1:
        psf = marker_scale_info.get("printed_scale_factor")
        ps_se = marker_scale_info.get("printed_scale_se")
        if psf is None:
            ser = float(marker_scale_info.get("scale_error_ratio", 0.0))
            if ser > -0.9:
                psf = 1.0 / (1.0 + ser)
            ps_se = float(marker_scale_info.get("scale_error_se", 0.005))
        if psf is not None:
            printed_scale_factor = float(psf)
            printed_scale_se = float(ps_se if ps_se is not None else 0.005)
            # Sanity gate against misconfigured sheet size (e.g. 180 mm vs 250 mm, > 6% discrepancy)
            if abs(printed_scale_factor - 1.0) <= 0.06:
                if abs(printed_scale_factor - 1.0) > max(2.0 * printed_scale_se, 0.005):
                    scale_applied = float(np.clip(printed_scale_factor, 0.94, 1.06))
                else:
                    scale_applied = 1.0
            else:
                printed_size_mismatch = True
                scale_applied = 1.0

    # Stage 1: Group pair rows by canonical edge (i, j) with i < j and perform MAD outlier rejection
    by_edge: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for pr in pair_rows:
        mi, mj = int(pr["marker_i"]), int(pr["marker_j"])
        if mi == mj or mi not in passes_by_marker or mj not in passes_by_marker:
            continue
        if mi > mj:
            continue
        by_edge.setdefault((mi, mj), []).append(pr)

    agg_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for (mi, mj), samples in sorted(by_edge.items()):
        b_vecs = np.array([[float(s["dx_grav_i_m"]), float(s["dy_grav_i_m"])] for s in samples], dtype=np.float64)
        dz_vals = np.array([float(s["dz_gravity_m"]) for s in samples], dtype=np.float64)
        yaw_vals = np.array([float(s["rel_yaw_ij_rad"]) for s in samples], dtype=np.float64)
        sig_d = np.array([max(0.003, float(s["sigma_dist_m"])) for s in samples], dtype=np.float64)
        sig_y = np.array([max(0.006, float(s["sigma_yaw_rad"])) for s in samples], dtype=np.float64)

        med_b = np.median(b_vecs, axis=0)
        dists_b = np.linalg.norm(b_vecs - med_b, axis=1)
        mad_b = float(np.median(dists_b))
        sin_med = float(np.median(np.sin(yaw_vals)))
        cos_med = float(np.median(np.cos(yaw_vals)))
        med_yaw = math.atan2(sin_med, cos_med)
        yaw_diffs = np.array([abs(_wrap_angle(float(y) - med_yaw)) for y in yaw_vals], dtype=np.float64)

        inlier_mask = (dists_b <= max(0.035, 3.0 * 1.4826 * mad_b)) & (yaw_diffs <= 0.20)
        if not np.any(inlier_mask):
            inlier_mask = np.ones(len(samples), dtype=bool)

        b_in = b_vecs[inlier_mask]
        dz_in = dz_vals[inlier_mask]
        yaw_in = yaw_vals[inlier_mask]
        sd_in = sig_d[inlier_mask]
        sy_in = sig_y[inlier_mask]
        n_in = int(b_in.shape[0])

        w_d = 1.0 / (sd_in * sd_in)
        w_d_sum = float(np.sum(w_d))
        b_mean_raw = np.sum(b_in * w_d[:, None], axis=0) / w_d_sum
        dz_mean_raw = float(np.sum(dz_in * w_d) / w_d_sum)

        # Apply printed-size scale factor (Condition 3)
        b_mean = scale_applied * b_mean_raw
        dz_mean = scale_applied * dz_mean_raw
        d_horiz = float(np.linalg.norm(b_mean))
        d_3d = math.sqrt(d_horiz * d_horiz + dz_mean * dz_mean)

        w_y = 1.0 / (sy_in * sy_in)
        rel_yaw_mean = math.atan2(
            float(np.sum(w_y * np.sin(yaw_in))),
            float(np.sum(w_y * np.cos(yaw_in))),
        )

        var_b = float(np.sum(w_d * np.sum((b_in - b_mean_raw) ** 2, axis=1)) / w_d_sum) if n_in > 1 else 0.0
        sigma_b_base = math.sqrt((1.0 / w_d_sum) + (var_b / max(1, n_in)))
        sigma_scale_term = (printed_scale_se or 0.0) * d_3d
        sigma_b = max(0.004, math.hypot(scale_applied * sigma_b_base, sigma_scale_term))

        yaw_res = np.array([_wrap_angle(float(y) - rel_yaw_mean) for y in yaw_in], dtype=np.float64)
        var_y = float(np.sum(w_y * (yaw_res * yaw_res)) / float(np.sum(w_y))) if n_in > 1 else 0.0
        sigma_yaw = max(0.008, math.sqrt((1.0 / float(np.sum(w_y))) + (var_y / max(1, n_in))))

        agg_edges[(mi, mj)] = {
            "marker_i": mi,
            "marker_j": mj,
            "sightings_count": n_in,
            "b_ij": b_mean,
            "dz_ij": dz_mean,
            "dist_horiz_measured_m": d_horiz,
            "dist_3d_measured_m": d_3d,
            "rel_yaw_measured_rad": rel_yaw_mean,
            "sigma_dist_m": sigma_b,
            "sigma_yaw_rad": sigma_yaw,
        }

    # Build adjacency list over agg_edges
    adj: dict[int, list[int]] = {mid: [] for mid in seen_ids}
    for mi, mj in agg_edges:
        adj[mi].append(mj)
        adj[mj].append(mi)

    # Stage 2: Dijkstra shortest-uncertainty spanning tree rooted at Marker 0
    connected_ids: list[int] = []
    spanning_tree_edges: set[tuple[int, int]] = set()
    state_init: dict[int, np.ndarray] = {}

    if 0 in passes_by_marker:
        state_init[0] = np.array([0.0, 0.0, float(origin_height_m), 0.0], dtype=np.float64)
        dist_cost: dict[int, float] = {0: 0.0}
        parent_edge: dict[int, tuple[int, tuple[int, int]]] = {}
        pq: list[tuple[float, int]] = [(0.0, 0)]
        visited: set[int] = set()

        while pq:
            c_u, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u in parent_edge:
                p_node, e_key = parent_edge[u]
                spanning_tree_edges.add(e_key)
                e_data = agg_edges[e_key]
                x_p, y_p, z_p, psi_p = state_init[p_node]
                if e_key[0] == p_node:
                    # Forward edge p_node -> u
                    b_vec = e_data["b_ij"]
                    dz_val = e_data["dz_ij"]
                    dpsi = e_data["rel_yaw_measured_rad"]
                    psi_u = _wrap_angle(psi_p + dpsi)
                    c_p, s_p = math.cos(psi_p), math.sin(psi_p)
                    dx_y = c_p * float(b_vec[0]) - s_p * float(b_vec[1])
                    dy_y = s_p * float(b_vec[0]) + c_p * float(b_vec[1])
                    state_init[u] = np.array([x_p + dx_y, y_p + dy_y, z_p + dz_val, psi_u], dtype=np.float64)
                else:
                    # Reverse edge u -> p_node (stored as (u, p_node))
                    b_vec = e_data["b_ij"]
                    dz_val = e_data["dz_ij"]
                    dpsi = e_data["rel_yaw_measured_rad"]
                    psi_u = _wrap_angle(psi_p - dpsi)
                    c_u_ang, s_u_ang = math.cos(psi_u), math.sin(psi_u)
                    dx_y = c_u_ang * float(b_vec[0]) - s_u_ang * float(b_vec[1])
                    dy_y = s_u_ang * float(b_vec[0]) + c_u_ang * float(b_vec[1])
                    state_init[u] = np.array([x_p - dx_y, y_p - dy_y, z_p - dz_val, psi_u], dtype=np.float64)

            for v in sorted(adj.get(u, [])):
                if v in visited:
                    continue
                e_key = (min(u, v), max(u, v))
                w_cost = float(agg_edges[e_key]["sigma_dist_m"]) ** 2
                new_cost = c_u + w_cost
                if new_cost < dist_cost.get(v, float("inf")):
                    dist_cost[v] = new_cost
                    parent_edge[v] = (u, e_key)
                    heapq.heappush(pq, (new_cost, v))

        connected_ids = sorted(visited)

    unconnected_ids = sorted(set(seen_ids) - set(connected_ids))
    conn_edges = [
        e_key for e_key in sorted(agg_edges.keys()) if e_key[0] in state_init and e_key[1] in state_init
    ]
    cycle_count = max(0, len(conn_edges) - len(connected_ids) + 1) if connected_ids else 0

    # Compute open-loop cycle residual on non-tree chord edges before Stage 3 Gauss-Newton optimization
    open_loop_cycle_residual_m: float | None = None
    if cycle_count >= 1:
        chord_residuals: list[float] = []
        for e_key in conn_edges:
            if e_key in spanning_tree_edges:
                continue
            mi, mj = e_key
            ed = agg_edges[e_key]
            xi, yi, zi, psii = state_init[mi]
            xj, yj, zj, _ = state_init[mj]
            c_i, s_i = math.cos(psii), math.sin(psii)
            pred_xj = xi + c_i * float(ed["b_ij"][0]) - s_i * float(ed["b_ij"][1])
            pred_yj = yi + s_i * float(ed["b_ij"][0]) + c_i * float(ed["b_ij"][1])
            pred_zj = zi + float(ed["dz_ij"])
            chord_residuals.append(
                math.sqrt((pred_xj - xj) ** 2 + (pred_yj - yj) ** 2 + (pred_zj - zj) ** 2)
            )
        if chord_residuals:
            open_loop_cycle_residual_m = float(max(chord_residuals))

    # Stage 3: Joint Gauss-Newton / Levenberg-Marquardt optimization over connected_ids \ {0}
    opt_ids = [mid for mid in connected_ids if mid != 0]
    id_to_var_idx = {mid: idx for idx, mid in enumerate(opt_ids)}
    n_vars = 4 * len(opt_ids)
    solved_state: dict[int, np.ndarray] = {mid: state_init[mid].copy() for mid in connected_ids}
    pos_uncertainty_by_id: dict[int, float] = {0: 0.0}

    if n_vars > 0 and conn_edges:
        theta = np.zeros(n_vars, dtype=np.float64)
        for mid, idx in id_to_var_idx.items():
            theta[4 * idx : 4 * idx + 4] = solved_state[mid]

        def _unpack(vec: np.ndarray) -> dict[int, np.ndarray]:
            st: dict[int, np.ndarray] = {0: state_init[0].copy()}
            for mid, idx in id_to_var_idx.items():
                s_m = vec[4 * idx : 4 * idx + 4].copy()
                s_m[3] = _wrap_angle(float(s_m[3]))
                st[mid] = s_m
            return st

        def _residuals(vec: np.ndarray) -> np.ndarray:
            st = _unpack(vec)
            res_list: list[float] = []
            for e_key in conn_edges:
                mi, mj = e_key
                ed = agg_edges[e_key]
                xi, yi, zi, psii = st[mi]
                xj, yj, zj, psij = st[mj]
                dx = xj - xi
                dy = yj - yi
                dz = zj - zi
                c_i, s_i = math.cos(psii), math.sin(psii)
                # R_2(psi_i)^T @ [dx, dy]
                bx_pred = c_i * dx + s_i * dy
                by_pred = -s_i * dx + c_i * dy
                sig_b = float(ed["sigma_dist_m"])
                sig_d = max(0.003, 0.75 * sig_b)
                sig_z = max(0.004, sig_b)
                sig_y = float(ed["sigma_yaw_rad"])

                res_list.append((bx_pred - float(ed["b_ij"][0])) / sig_b)
                res_list.append((by_pred - float(ed["b_ij"][1])) / sig_b)
                chord_pred = math.sqrt(dx * dx + dy * dy + 1e-12)
                res_list.append((chord_pred - float(ed["dist_horiz_measured_m"])) / sig_d)
                res_list.append((dz - float(ed["dz_ij"])) / sig_z)
                res_list.append(_wrap_angle(psij - psii - float(ed["rel_yaw_measured_rad"])) / sig_y)
            return np.array(res_list, dtype=np.float64)

        def _huber_weights(r: np.ndarray, delta: float = 2.5) -> np.ndarray:
            abs_r = np.abs(r)
            return np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-12))

        lam = 1e-3
        eps_fd = 1e-6
        r_curr = _residuals(theta)
        w_curr = _huber_weights(r_curr)
        cost_curr = float(np.sum(w_curr * r_curr * r_curr))

        J = np.zeros(( len(r_curr), n_vars), dtype=np.float64)
        for _ in range(30):
            r_curr = _residuals(theta)
            w_curr = _huber_weights(r_curr)
            for j_col in range(n_vars):
                t_plus = theta.copy()
                t_plus[j_col] += eps_fd
                r_plus = _residuals(t_plus)
                J[:, j_col] = (r_plus - r_curr) / eps_fd

            JW = J.T * w_curr[None, :]
            H = JW @ J
            g = JW @ r_curr
            try:
                step = -np.linalg.solve(H + lam * np.eye(n_vars), g)
            except np.linalg.LinAlgError:
                break
            if float(np.linalg.norm(step)) < 1e-7:
                break
            theta_cand = theta + step
            r_cand = _residuals(theta_cand)
            w_cand = _huber_weights(r_cand)
            cost_cand = float(np.sum(w_cand * r_cand * r_cand))
            if cost_cand < cost_curr:
                theta = theta_cand
                cost_curr = cost_cand
                lam = max(1e-6, lam * 0.5)
            else:
                lam = min(1e4, lam * 4.0)

        solved_state = _unpack(theta)
        r_final = _residuals(theta)
        w_final = _huber_weights(r_final)
        JW_final = J.T * w_final[None, :]
        H_final = JW_final @ J
        cov = np.linalg.inv(H_final + 1e-6 * np.eye(n_vars))
        for mid, idx in id_to_var_idx.items():
            var_xy = max(0.0, float(cov[4 * idx, 4 * idx] + cov[4 * idx + 1, 4 * idx + 1]))
            dist_from_origin = float(np.linalg.norm(solved_state[mid][:2]))
            scale_u = (printed_scale_se or 0.0) * dist_from_origin
            pos_uncertainty_by_id[mid] = math.hypot(math.sqrt(var_xy), scale_u)

    # Evaluate post-closure edge residuals and VIO-to-pair scale ratio
    pair_edges_json: list[dict[str, Any]] = []
    edge_residuals: list[float] = []
    for e_key in sorted(agg_edges.keys()):
        mi, mj = e_key
        ed = agg_edges[e_key]
        if mi in solved_state and mj in solved_state:
            pi = solved_state[mi][:3]
            pj = solved_state[mj][:3]
            d_solved = float(np.linalg.norm(pj - pi))
        else:
            pi = raw_xyz_yard[mi]
            pj = raw_xyz_yard[mj]
            d_solved = float(np.linalg.norm(pj - pi))
        d_meas = float(ed["dist_3d_measured_m"])
        res_m = d_solved - d_meas
        if mi in solved_state and mj in solved_state:
            edge_residuals.append(res_m)
        pair_edges_json.append(
            {
                "marker_i": int(mi),
                "marker_j": int(mj),
                "sightings_count": int(ed["sightings_count"]),
                "dist_3d_measured_m": round(d_meas, 6),
                "dist_3d_solved_m": round(d_solved, 6),
                "dist_horiz_measured_m": round(float(ed["dist_horiz_measured_m"]), 6),
                "dz_measured_m": round(float(ed["dz_ij"]), 6),
                "rel_yaw_measured_rad": round(float(ed["rel_yaw_measured_rad"]), 6),
                "residual_m": round(res_m, 6),
                "sigma_dist_m": round(float(ed["sigma_dist_m"]), 6),
            }
        )

    closed_loop_edge_rmse_m: float | None = (
        float(math.sqrt(np.mean(np.square(edge_residuals)))) if edge_residuals else None
    )
    max_edge_residual_m: float | None = (
        float(np.max(np.abs(edge_residuals))) if edge_residuals else None
    )

    # Fit 2D similarity transform from raw ARKit Yard positions to solved Pair-Graph positions
    vio_to_pair_scale_ratio: float | None = None
    if len(connected_ids) >= 2:
        src_xy = np.array([raw_xyz_yard[m][:2] for m in connected_ids], dtype=np.float64)
        dst_xy = np.array([solved_state[m][:2] for m in connected_ids], dtype=np.float64)
        s_sim, R2_sim, t2_sim = _fit_2d_similarity(src_xy, dst_xy, anchor_origin=(0 in connected_ids))
        vio_to_pair_scale_ratio = round(1.0 / max(s_sim, 1e-6), 6)
        solver_method = "one_frame_pair_graph_v1"
    else:
        s_sim, R2_sim, t2_sim = 1.0, np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        solver_method = "arkit_world_fallback_v1"

    # Build markers_yard list for all seen markers (connected + unconnected fallback)
    markers_yard_list: list[dict[str, Any]] = []
    solved_positions_yard: dict[int, np.ndarray] = {}

    for mid in seen_ids:
        raw_p = raw_xyz_yard[mid]
        n_xyz = raw_normal_yard[mid]
        deg = len(adj.get(mid, []))
        if mid in solved_state and len(connected_ids) >= 2:
            sol_p = solved_state[mid][:3]
            sol_yaw = float(solved_state[mid][3])
            conn_flag = True
            source = "pair_graph"
            unc_m = float(pos_uncertainty_by_id.get(mid, 0.01))
        elif mid == 0 and 0 in solved_state:
            sol_p = solved_state[0][:3]
            sol_yaw = 0.0
            conn_flag = True
            source = "pair_graph" if len(connected_ids) >= 2 else "arkit_world_fallback"
            unc_m = 0.0
        elif len(connected_ids) >= 2:
            xy_mapped = s_sim * (R2_sim @ raw_p[:2]) + t2_sim
            sol_p = np.array([float(xy_mapped[0]), float(xy_mapped[1]), float(raw_p[2])], dtype=np.float64)
            rot_yaw = math.atan2(float(R2_sim[1, 0]), float(R2_sim[0, 0]))
            sol_yaw = _wrap_angle(raw_yaw_yard[mid] + rot_yaw)
            conn_flag = False
            source = "vio_similarity_fallback"
            unc_m = math.hypot(0.02, 0.04 * float(np.linalg.norm(sol_p[:2])))
        else:
            sol_p = raw_p.copy()
            sol_yaw = raw_yaw_yard[mid]
            conn_flag = (mid == 0)
            source = "arkit_world_fallback"
            unc_m = 0.0 if mid == 0 else math.hypot(0.02, 0.05 * float(np.linalg.norm(sol_p[:2])))

        solved_positions_yard[mid] = sol_p
        markers_yard_list.append(
            {
                "marker_id": int(mid),
                "xyz_m": [round(float(x), 6) for x in sol_p],
                "normal_xyz": [round(float(x), 6) for x in n_xyz],
                "sightings_count": int(sightings_count_by_id[mid]),
                "yaw_yard_rad": round(float(sol_yaw), 6),
                "connected_to_origin": bool(conn_flag),
                "placement_source": source,
                "pair_degree": int(deg),
                "position_uncertainty_m": round(float(unc_m), 6),
                "xyz_arkit_open_loop_m": [round(float(x), 6) for x in raw_p],
            }
        )

    pair_graph_connected = (
        len(seen_ids) >= 2
        and len(unconnected_ids) == 0
        and 0 in connected_ids
    )

    marker_layout_dict: dict[str, Any] = {
        "solver_method": solver_method,
        "markers_seen_ids": [int(m) for m in seen_ids],
        "markers_connected_ids": [int(m) for m in connected_ids],
        "unconnected_marker_ids": [int(m) for m in unconnected_ids],
        "pair_graph_connected": bool(pair_graph_connected),
        "cycle_count": int(cycle_count),
        "pair_edges": pair_edges_json,
        "open_loop_cycle_residual_m": (
            round(open_loop_cycle_residual_m, 6) if open_loop_cycle_residual_m is not None else None
        ),
        "closed_loop_edge_rmse_m": (
            round(closed_loop_edge_rmse_m, 6) if closed_loop_edge_rmse_m is not None else None
        ),
        "max_edge_residual_m": (
            round(max_edge_residual_m, 6) if max_edge_residual_m is not None else None
        ),
        "vio_to_pair_scale_ratio": vio_to_pair_scale_ratio,
        "printed_scale_factor": (
            round(printed_scale_factor, 6) if printed_scale_factor is not None else None
        ),
        "printed_scale_se": (
            round(printed_scale_se, 6) if printed_scale_se is not None else None
        ),
        "scale_applied": round(float(scale_applied), 6),
        "trajectory_warp_applied": False,
        "trajectory_warp_method": "none",
        "per_marker_warp_corrections": [],
    }

    return {
        "markers_yard": markers_yard_list,
        "marker_layout": marker_layout_dict,
        "solved_positions_yard": solved_positions_yard,
        "raw_positions_yard": raw_xyz_yard,
        "vio_similarity": (s_sim, R2_sim, t2_sim),
        "printed_size_mismatch": printed_size_mismatch,
    }
