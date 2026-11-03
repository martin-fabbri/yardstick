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

"""Synthetic Level 1 yard recording generator and `.zip` archive helper.

Generates a deterministic, physically consistent synthetic yard recording
that exercises every requirement in `YS-00` through `YS-115` without needing
an iPhone on the test bench.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from yardstick import CONTRACT_VERSION
from yardstick.geometry import rotmat_to_quat
from yardstick.markers import (
    MARKERS_CSV_BASE_COLUMNS,
    MARKERS_CSV_CAMERA_COLUMNS,
    PAIR_CSV_COLUMNS,
    compute_single_frame_pair_measurement,
)
from yardstick.validator import sha256_file


@dataclass
class SynthOptions:
    """Configuration for generating a synthetic Level 1 capture."""

    capture_id: str = "20261102T120000Z_yardstick_synth"
    num_saved_frames: int = 24
    fps: int = 5
    rgb_width: int = 640
    rgb_height: int = 480
    depth_width: int = 256
    depth_height: int = 192
    fx: float = 480.0
    fy: float = 480.0
    cx: float = 320.0
    cy: float = 240.0
    pitch_down_deg: float = 30.0
    include_marker_0: bool = True
    include_loop_closure: bool = True
    loop_drift_xy_m: tuple[float, float] = (0.012, -0.008)
    marker_0_normal_tilt_deg: float = 0.0
    actual_marker_size_mm: float = 250.0
    declared_marker_size_mm: float = 250.0
    limited_tracking_fraction: float = 0.0
    pose_jump_m: float = 0.0
    include_tape_checks: bool = True
    primary_tape_residual_m: float = 0.015
    held_out_residuals_m: tuple[float, float, float] = (0.012, -0.014, 0.016)
    location_status: str = "ok"
    true_north_yaw_deg: float = 35.0
    include_center_void_hole: bool = True
    include_upstream_frame_drop: bool = True
    include_wall_points: bool = True
    # YS-115 options for up to 8 tossed markers, one-frame pairs, VIO shrink, and depth-scale bias
    markers_world_layout: dict[int, tuple[float, ...]] | None = None
    walk_pair_sequence: list[tuple[int, int]] | None = None
    solo_sightings_ids: list[int] | None = None
    vio_scale_shrink_ratio: float = 0.0
    depth_scale_bias_ratio: float = 0.0
    max_marker_sighting_range_m: float = 2.00
    write_pairs_csv: bool = False
    wall_pass1_and_pass2: bool = False


def _camera_rotation_from_yaw_pitch(yaw_rad: float, pitch_down_deg: float) -> np.ndarray:
    """Build camera-to-world rotation matrix `R_w_c` in ARKit world (+Y up).

    At `yaw_rad = 0, pitch_down_deg = 0`:
    - Camera optical axis `-Z_c` points along `[1, 0, 0]` (+X_w).
    - Camera `+X_c` (right) points along `[0, 0, 1]` (+Z_w).
    - Camera `+Y_c` (up) points along `[0, 1, 0]` (+Y_w, up).
    Then pitch down by `pitch_down_deg` about camera `+X_c` and rotate yaw about `+Y_w`.
    """
    r0 = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    theta = -math.radians(pitch_down_deg)
    c_p, s_p = math.cos(theta), math.sin(theta)
    r_pitch_local = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_p, -s_p],
            [0.0, s_p, c_p],
        ],
        dtype=np.float64,
    )
    c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)
    r_yaw_world = np.array(
        [
            [c_y, 0.0, -s_y],
            [0.0, 1.0, 0.0],
            [s_y, 0.0, c_y],
        ],
        dtype=np.float64,
    )
    return r_yaw_world @ (r0 @ r_pitch_local)


def _marker_rotation_in_world(marker_yaw_rad: float = 0.0) -> np.ndarray:
    """Build marker-to-world rotation matrix `R_w_m` on ground Y_w = 0.

    At `marker_yaw_rad = 0`:
    - `+X_m` = `[1, 0, 0]` (+X_yard)
    - `+Y_m` = `[0, 0, -1]` (+Y_yard, since Z_yard = [0, 1, 0] and Y_yard = Z_yard x X_yard = [0, 0, -1])
    - `+Z_m` = `[0, 1, 0]` (upward normal)
    Rotating by `marker_yaw_rad` around `+Y_w` (`+Z_yard`) rotates `+X_m` toward `+Y_yard`.
    """
    c_y, s_y = math.cos(marker_yaw_rad), math.sin(marker_yaw_rad)
    x_axis = np.array([c_y, 0.0, -s_y], dtype=np.float64)
    z_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _render_aruco_marker_patch(marker_id: int, size_px: int = 240) -> np.ndarray:
    """Render an ArUco `DICT_5X5_250` marker with a crisp white quiet zone."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    inner = int(size_px * 0.8)
    marker_gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, inner)
    canvas = np.full((size_px, size_px), 255, dtype=np.uint8)
    pad = (size_px - inner) // 2
    canvas[pad : pad + inner, pad : pad + inner] = marker_gray
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def generate_synthetic_capture(
    out_dir: Path,
    options: SynthOptions | None = None,
) -> Path:
    """Write a complete Level 1 synthetic yard capture to `out_dir`."""
    opts = options or SynthOptions()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)
    (out_dir / "stills").mkdir(parents=True, exist_ok=True)

    actual_size_m = opts.actual_marker_size_mm / 1000.0
    half_s = actual_size_m / 2.0
    depth_bias_factor = 1.0 + float(opts.depth_scale_bias_ratio)
    vio_scale_factor = 1.0 + float(opts.vio_scale_shrink_ratio)
    use_custom_layout = opts.markers_world_layout is not None

    markers_world: dict[int, np.ndarray] = {}
    markers_yaw_world: dict[int, float] = {}
    if use_custom_layout and opts.markers_world_layout is not None:
        for mid, spec in opts.markers_world_layout.items():
            markers_world[int(mid)] = np.array([float(spec[0]), float(spec[1]), float(spec[2])], dtype=np.float64)
            markers_yaw_world[int(mid)] = float(spec[3]) if len(spec) >= 4 else 0.0
    else:
        markers_world = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float64),
            1: np.array([2.40, 0.0, -3.70], dtype=np.float64),
            2: np.array([5.65, 0.0, 0.0], dtype=np.float64),
            3: np.array([2.40, 0.0, 3.70], dtype=np.float64),
        }
        markers_yaw_world = {mid: 0.0 for mid in markers_world}

    if not opts.include_marker_0:
        markers_world.pop(0, None)
        markers_yaw_world.pop(0, None)

    # Build camera waypoint schedule if using custom tossed-marker layout
    custom_cam_schedule: list[tuple[np.ndarray, float, set[int]]] = []
    if use_custom_layout:
        cam_h = 1.32
        # 1. Start at Marker 0 (2 frames)
        p0 = markers_world.get(0, np.zeros(3, dtype=np.float64))
        for _ in range(2):
            custom_cam_schedule.append((np.array([p0[0] - 1.50, cam_h, p0[2]], dtype=np.float64), 0.0, {0}))

        # 2. Pair sightings along walk_pair_sequence
        pair_seq = opts.walk_pair_sequence
        if pair_seq is None:
            mids_ordered = [m for m in markers_world.keys() if m != 0]
            seq_nodes = [0] + mids_ordered + ([0] if opts.include_loop_closure else [])
            pair_seq = [(seq_nodes[i], seq_nodes[i + 1]) for i in range(len(seq_nodes) - 1)]

        for ma, mb in pair_seq:
            if ma not in markers_world or mb not in markers_world:
                continue
            pa = markers_world[ma]
            pb = markers_world[mb]
            mid_pt = 0.5 * (pa + pb)
            chord_vec = pb - pa
            chord_len = float(np.linalg.norm(chord_vec))
            # Perpendicular horizontal direction in XZ plane
            if chord_len > 1e-6:
                perp = np.array([-chord_vec[2] / chord_len, 0.0, chord_vec[0] / chord_len], dtype=np.float64)
            else:
                perp = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            # Stand back from the pair midpoint so both markers fit comfortably in FOV, <= 3.5 m range, and < 60 deg angle
            back_dist = max(1.80, 0.68 * chord_len)
            horiz_dist = math.hypot(back_dist, 0.5 * chord_len)
            cam_h_pair = max(cam_h, 0.60 * horiz_dist)
            cam_p = np.array([mid_pt[0] - back_dist * perp[0], cam_h_pair, mid_pt[2] - back_dist * perp[2]], dtype=np.float64)
            yaw_look = math.atan2(float(mid_pt[2] - cam_p[2]), float(mid_pt[0] - cam_p[0]))
            for _ in range(2):
                custom_cam_schedule.append((cam_p.copy(), yaw_look, {ma, mb}))

        # 3. Optional solo sightings (e.g. disconnected marker)
        for m_solo in opts.solo_sightings_ids or []:
            if m_solo in markers_world:
                ps = markers_world[m_solo]
                cam_p = np.array([ps[0] - 1.50, cam_h, ps[2]], dtype=np.float64)
                for _ in range(2):
                    custom_cam_schedule.append((cam_p.copy(), 0.0, {m_solo}))

        # 4. Closing pass at Marker 0 (3 frames)
        if opts.include_loop_closure and 0 in markers_world:
            for _ in range(3):
                custom_cam_schedule.append((np.array([p0[0] - 1.50, cam_h, p0[2]], dtype=np.float64), 0.0, {0}))

    n_frames = len(custom_cam_schedule) if use_custom_layout else opts.num_saved_frames
    cam_height_m = 0.90
    cx_loop = 1.85
    radius_loop = 2.75
    total_time_steps = float(max(1, (n_frames - 2) + (1 if opts.include_upstream_frame_drop else 0)))

    frames_rows: list[dict[str, Any]] = []
    poses_rows: list[dict[str, Any]] = []
    stills_rows: list[dict[str, Any]] = []
    markers_rows: list[dict[str, Any]] = []
    pairs_rows: list[dict[str, Any]] = []
    events_rows: list[dict[str, Any]] = []
    rgb_frames_for_video: list[np.ndarray] = []
    saved_timestamps: list[float] = []
    all_timestamps: list[float] = []

    dt_step = 1.0 / float(opts.fps)
    t_cursor = 100.0

    max_mid = max(markers_world.keys(), default=3)
    marker_patches = {mid: _render_aruco_marker_patch(mid, 240) for mid in range(max(8, max_mid + 1))}
    emit_cam_and_pairs = bool(opts.write_pairs_csv or use_custom_layout)

    for s_idx in range(n_frames):
        if opts.include_upstream_frame_drop and s_idx == 3:
            t_drop = round(t_cursor, 6)
            all_timestamps.append(t_drop)
            frames_rows.append(
                {
                    "t": f"{t_drop:.6f}",
                    "saved_index": "",
                    "dropped": "true",
                    "drop_reason": "writer_queue_full",
                    "tracking_state": "normal",
                    "tracking_reason": "",
                }
            )
            t_cursor += dt_step

        t_s = round(t_cursor, 6)
        t_cursor += dt_step

        saved_timestamps.append(t_s)
        all_timestamps.append(t_s)

        if use_custom_layout:
            cam_pos_true, yaw_rad, active_mids = custom_cam_schedule[s_idx]
            frac = float(s_idx) / float(max(1, n_frames - 1))
            cam_pos_w = np.array(
                [
                    vio_scale_factor * cam_pos_true[0] + frac * opts.loop_drift_xy_m[0],
                    cam_pos_true[1],
                    vio_scale_factor * cam_pos_true[2] - frac * opts.loop_drift_xy_m[1],
                ],
                dtype=np.float64,
            )
            if opts.pose_jump_m > 0.0 and s_idx >= n_frames // 2:
                cam_pos_w[0] += opts.pose_jump_m
        else:
            step_f = (t_s - 100.0) / dt_step
            theta = (2.0 * math.pi * step_f) / total_time_steps
            wx = cx_loop - radius_loop * math.cos(theta)
            wz = -radius_loop * math.sin(theta)
            if s_idx in (0, 1) or s_idx >= n_frames - 3:
                target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            elif s_idx in (5, 6):
                target = np.array([2.40, 0.0, -3.70], dtype=np.float64)
            elif s_idx in (10, 11):
                target = np.array([5.65, 0.0, 0.0], dtype=np.float64)
            elif s_idx in (15, 16):
                target = np.array([2.40, 0.0, 3.70], dtype=np.float64)
            else:
                target = np.array([cx_loop, 0.0, 0.0], dtype=np.float64)
            yaw_rad = math.atan2(float(target[2]) - wz, float(target[0]) - wx)
            if s_idx == 0:
                yaw_rad = 0.0

            cam_pos_true = np.array([wx, cam_height_m, wz], dtype=np.float64)
            if opts.pose_jump_m > 0.0 and s_idx >= n_frames // 2:
                wx += opts.pose_jump_m
            cam_pos_w = np.array([wx, cam_height_m, wz], dtype=np.float64)
            active_mids = set(markers_world.keys())

        r_w_c = _camera_rotation_from_yaw_pitch(yaw_rad, opts.pitch_down_deg)
        qx, qy, qz, qw = rotmat_to_quat(r_w_c)

        is_limited = (s_idx / float(max(1, n_frames))) < opts.limited_tracking_fraction
        t_state = "limited" if is_limited else "normal"
        t_reason = "relocalizing" if is_limited else ""

        frames_rows.append(
            {
                "t": f"{t_s:.6f}",
                "saved_index": str(s_idx),
                "dropped": "false",
                "drop_reason": "",
                "tracking_state": t_state,
                "tracking_reason": t_reason,
            }
        )

        poses_rows.append(
            {
                "t": f"{t_s:.6f}",
                "saved_index": str(s_idx),
                "px": f"{cam_pos_w[0]:.6f}",
                "py": f"{cam_pos_w[1]:.6f}",
                "pz": f"{cam_pos_w[2]:.6f}",
                "qx": f"{qx:.8f}",
                "qy": f"{qy:.8f}",
                "qz": f"{qz:.8f}",
                "qw": f"{qw:.8f}",
                "fx": f"{opts.fx:.4f}",
                "fy": f"{opts.fy:.4f}",
                "cx": f"{opts.cx:.4f}",
                "cy": f"{opts.cy:.4f}",
                "width": str(opts.rgb_width),
                "height": str(opts.rgb_height),
            }
        )

        # Synthesize planar depth image (uint16 mm) and confidence image (uint8 0..2)
        scale_x = float(opts.depth_width) / float(opts.rgb_width)
        scale_y = float(opts.depth_height) / float(opts.rgb_height)
        fx_d = opts.fx * scale_x
        fy_d = opts.fy * scale_y
        cx_d = opts.cx * scale_x
        cy_d = opts.cy * scale_y

        vv, uu = np.meshgrid(
            np.arange(opts.depth_height, dtype=np.float64),
            np.arange(opts.depth_width, dtype=np.float64),
            indexing="ij",
        )
        ray_x_c = (uu - cx_d) / fx_d
        ray_y_c = -(vv - cy_d) / fy_d
        ray_z_c = -np.ones_like(uu)
        rays_c = np.stack([ray_x_c, ray_y_c, ray_z_c], axis=-1)
        rays_w = rays_c @ r_w_c.T

        ref_cam_for_scene = cam_pos_true if use_custom_layout else cam_pos_w
        denom = rays_w[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_depth_m = np.where(denom < -1e-4, -ref_cam_for_scene[1] / denom, 0.0)

        pts_hit_w = ref_cam_for_scene.reshape(1, 1, 3) + z_depth_m[..., None] * rays_w

        # Optional vertical wall at X_w = 5.95 m (YS-51)
        if opts.include_wall_points and (not use_custom_layout) and s_idx in (10, 11):
            with np.errstate(divide="ignore", invalid="ignore"):
                z_wall = np.where(
                    rays_w[..., 0] > 1e-4, (5.95 - ref_cam_for_scene[0]) / rays_w[..., 0], 999.0
                )
            wall_y = ref_cam_for_scene[1] + z_wall * rays_w[..., 1]
            use_wall = (
                (z_wall > 0.3)
                & (z_wall < z_depth_m)
                & (wall_y >= 0.0)
                & (wall_y <= 2.2)
            )
            z_depth_m = np.where(use_wall, z_wall, z_depth_m)
            pts_hit_w = ref_cam_for_scene.reshape(1, 1, 3) + z_depth_m[..., None] * rays_w

        # Optional Pass-1 & Pass-2 house wall behind Marker 0 at X_w_true = 1.65 m (YS-115 wall single-band test)
        if opts.wall_pass1_and_pass2 and (s_idx in (0, 1) or s_idx >= n_frames - 3):
            wall_x_true = 1.65
            with np.errstate(divide="ignore", invalid="ignore"):
                z_wall_p12 = np.where(
                    rays_w[..., 0] > 1e-4, (wall_x_true - ref_cam_for_scene[0]) / rays_w[..., 0], 999.0
                )
            wall_y_p12 = ref_cam_for_scene[1] + z_wall_p12 * rays_w[..., 1]
            wall_z_p12 = ref_cam_for_scene[2] + z_wall_p12 * rays_w[..., 2]
            use_wall_p12 = (
                (z_wall_p12 > 0.4)
                & (z_wall_p12 < z_depth_m)
                & (wall_y_p12 >= 0.05)
                & (wall_y_p12 <= 2.2)
                & (np.abs(wall_z_p12) <= 1.2)
            )
            z_depth_m = np.where(use_wall_p12, z_wall_p12, z_depth_m)
            pts_hit_w = ref_cam_for_scene.reshape(1, 1, 3) + z_depth_m[..., None] * rays_w

        # Apply optional LiDAR depth-scale bias (Condition 3)
        z_depth_biased_m = z_depth_m * depth_bias_factor
        ranges_m = z_depth_biased_m * np.sqrt(1.0 + ray_x_c * ray_x_c + ray_y_c * ray_y_c)
        valid_range = (ranges_m >= 0.25) & (ranges_m <= 4.25)

        if opts.include_center_void_hole and not use_custom_layout:
            in_void = (
                (pts_hit_w[..., 0] >= 1.0)
                & (pts_hit_w[..., 0] <= 2.6)
                & (pts_hit_w[..., 2] >= -0.8)
                & (pts_hit_w[..., 2] <= 0.8)
            )
            valid_range = valid_range & (~in_void)

        depth_mm = np.where(valid_range, np.round(z_depth_biased_m * 1000.0), 0).astype(np.uint16)
        conf_u8 = np.where(valid_range, 2, 0).astype(np.uint8)

        Image.fromarray(depth_mm).save(out_dir / "depth" / f"{s_idx:06d}.png")
        Image.fromarray(conf_u8).save(out_dir / "confidence" / f"{s_idx:06d}.png")

        # Synthesize RGB image with grid texture and visible ArUco markers
        rgb_img = np.full((opts.rgb_height, opts.rgb_width, 3), 185, dtype=np.uint8)
        rgb_img[::32, :, :] = 70
        rgb_img[:, ::32, :] = 70

        r_c_w = r_w_c.T
        max_depth_c = float(opts.max_marker_sighting_range_m) if not use_custom_layout else max(3.80, float(opts.max_marker_sighting_range_m))
        frame_detected_markers: dict[int, dict[str, Any]] = {}

        for mid, m_pos_w in sorted(markers_world.items()):
            if use_custom_layout:
                if mid not in active_mids:
                    continue
                m_center_true = m_pos_w.copy()
                cam_ref = cam_pos_true
            else:
                if mid == 0 and 2 <= s_idx < n_frames - 3:
                    continue
                is_return_pass = mid == 0 and s_idx >= n_frames - 3
                if is_return_pass and not opts.include_loop_closure:
                    continue
                m_center_true = m_pos_w.copy()
                if is_return_pass:
                    if opts.pose_jump_m > 0.0 and s_idx >= n_frames // 2:
                        m_center_true[0] += opts.pose_jump_m
                    m_center_true[0] += opts.loop_drift_xy_m[0]
                    m_center_true[2] -= opts.loop_drift_xy_m[1]
                cam_ref = cam_pos_w

            corners_local = np.array(
                [
                    [-half_s, +half_s, 0.0],
                    [+half_s, +half_s, 0.0],
                    [+half_s, -half_s, 0.0],
                    [-half_s, -half_s, 0.0],
                ],
                dtype=np.float64,
            )
            r_w_m = _marker_rotation_in_world(markers_yaw_world.get(mid, 0.0))

            corners_w_true = m_center_true.reshape(1, 3) + corners_local @ r_w_m.T
            corners_c_true = (corners_w_true - cam_ref.reshape(1, 3)) @ r_c_w.T
            depths_c = -corners_c_true[:, 2]

            if np.all((depths_c >= 0.50) & (depths_c <= max_depth_c)):
                u_px = opts.cx + opts.fx * (corners_c_true[:, 0] / depths_c)
                v_px = opts.cy - opts.fy * (corners_c_true[:, 1] / depths_c)
                if np.all(
                    (u_px >= 10)
                    & (u_px < opts.rgb_width - 10)
                    & (v_px >= 10)
                    & (v_px < opts.rgb_height - 10)
                ):
                    patch = marker_patches[mid]
                    ph, pw = patch.shape[:2]
                    pad_q = float(pw) * 0.1
                    lo_px = pad_q - 0.44
                    hi_px = float(pw) - pad_q - 0.56
                    src_pts = np.array(
                        [
                            [lo_px, lo_px],
                            [hi_px, lo_px],
                            [hi_px, hi_px],
                            [lo_px, hi_px],
                        ],
                        dtype=np.float32,
                    )
                    dst_pts = np.stack([u_px, v_px], axis=-1).astype(np.float32)
                    h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(
                        patch,
                        h_mat,
                        (opts.rgb_width, opts.rgb_height),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )
                    mask = np.zeros((opts.rgb_height, opts.rgb_width), dtype=np.uint8)
                    pad_sheet = float(pw) * 0.18
                    full_box = np.array(
                        [
                            [-pad_sheet, -pad_sheet],
                            [pw + pad_sheet, -pad_sheet],
                            [pw + pad_sheet, ph + pad_sheet],
                            [-pad_sheet, ph + pad_sheet],
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    dst_box = cv2.perspectiveTransform(full_box, h_mat).astype(np.int32)
                    cv2.fillConvexPoly(mask, dst_box, 255)
                    rgb_img[mask > 0] = warped[mask > 0]

                    # Single-frame camera-space marker center & orientation (with optional depth_scale_bias_ratio)
                    p_cam_meas = depth_bias_factor * ((m_center_true - cam_ref) @ r_c_w.T)
                    r_c_m = r_c_w @ r_w_m
                    cqx, cqy, cqz, cqw = rotmat_to_quat(r_c_m)
                    range_m = float(np.linalg.norm(p_cam_meas))
                    view_dir = -p_cam_meas / max(range_m, 1e-9)
                    cos_inc = float(np.clip(abs(float(np.dot(r_c_m[:, 2], view_dir))), -1.0, 1.0))
                    angle_deg = math.degrees(math.acos(cos_inc))

                    if use_custom_layout:
                        m_center_logged = cam_pos_w + (r_w_c @ p_cam_meas)
                    else:
                        m_center_logged = m_center_true

                    qx_m, qy_m, qz_m, qw_m = rotmat_to_quat(r_w_m)
                    if mid == 0 and abs(opts.marker_0_normal_tilt_deg) > 1e-6:
                        tilt_r = math.radians(opts.marker_0_normal_tilt_deg)
                        ct, st = math.cos(tilt_r), math.sin(tilt_r)
                        r_tilt = np.array(
                            [[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]],
                            dtype=np.float64,
                        )
                        qx_m, qy_m, qz_m, qw_m = rotmat_to_quat(r_tilt @ r_w_m)
                    jitter_mm = 0.25 * math.sin(s_idx * 1.7 + mid)
                    meas_mm = (opts.actual_marker_size_mm * depth_bias_factor) + jitter_mm

                    mrow: dict[str, Any] = {
                        "t": f"{t_s:.6f}",
                        "saved_index": str(s_idx),
                        "marker_id": str(mid),
                        "size_mm_printed": f"{opts.declared_marker_size_mm:.2f}",
                        "size_mm_measured": f"{meas_mm:.3f}",
                        "x": f"{m_center_logged[0]:.6f}",
                        "y": f"{m_center_logged[1]:.6f}",
                        "z": f"{m_center_logged[2]:.6f}",
                        "qx": f"{qx_m:.8f}",
                        "qy": f"{qy_m:.8f}",
                        "qz": f"{qz_m:.8f}",
                        "qw": f"{qw_m:.8f}",
                        "c0_u": f"{u_px[0]:.3f}",
                        "c0_v": f"{v_px[0]:.3f}",
                        "c1_u": f"{u_px[1]:.3f}",
                        "c1_v": f"{v_px[1]:.3f}",
                        "c2_u": f"{u_px[2]:.3f}",
                        "c2_v": f"{v_px[2]:.3f}",
                        "c3_u": f"{u_px[3]:.3f}",
                        "c3_v": f"{v_px[3]:.3f}",
                    }
                    if emit_cam_and_pairs:
                        mrow.update(
                            {
                                "cam_x": f"{p_cam_meas[0]:.6f}",
                                "cam_y": f"{p_cam_meas[1]:.6f}",
                                "cam_z": f"{p_cam_meas[2]:.6f}",
                                "cam_qx": f"{cqx:.8f}",
                                "cam_qy": f"{cqy:.8f}",
                                "cam_qz": f"{cqz:.8f}",
                                "cam_qw": f"{cqw:.8f}",
                                "range_m": f"{range_m:.4f}",
                                "angle_deg": f"{angle_deg:.2f}",
                                "edge_disagreement_ratio": "0.00400",
                                "plane_rmse_m": "0.00150",
                                "depth_sample_count": "24",
                            }
                        )
                    markers_rows.append(mrow)
                    frame_detected_markers[mid] = {
                        "p_cam": p_cam_meas,
                        "R_c_m": r_c_m,
                        "size_mm": meas_mm,
                        "angle_deg": angle_deg,
                    }

        if emit_cam_and_pairs and len(frame_detected_markers) >= 2:
            gravity_cam = r_c_w @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
            det_ids = sorted(frame_detected_markers.keys())
            for ia in range(len(det_ids)):
                for ib in range(ia + 1, len(det_ids)):
                    mi, mj = det_ids[ia], det_ids[ib]
                    di = frame_detected_markers[mi]
                    dj = frame_detected_markers[mj]
                    p_meas = compute_single_frame_pair_measurement(
                        t_s=t_s,
                        saved_index=s_idx,
                        marker_i=mi,
                        marker_j=mj,
                        p_i_cam=di["p_cam"],
                        R_cam_from_i=di["R_c_m"],
                        size_i_mm=di["size_mm"],
                        angle_i_deg=di["angle_deg"],
                        p_j_cam=dj["p_cam"],
                        R_cam_from_j=dj["R_c_m"],
                        size_j_mm=dj["size_mm"],
                        angle_j_deg=dj["angle_deg"],
                        gravity_up_cam=gravity_cam,
                    )
                    pairs_rows.append(
                        {
                            "t": f"{p_meas['t']:.6f}",
                            "saved_index": str(p_meas["saved_index"]),
                            "marker_i": str(p_meas["marker_i"]),
                            "marker_j": str(p_meas["marker_j"]),
                            "dist_3d_m": f"{p_meas['dist_3d_m']:.6f}",
                            "dist_horiz_m": f"{p_meas['dist_horiz_m']:.6f}",
                            "dz_gravity_m": f"{p_meas['dz_gravity_m']:.6f}",
                            "dx_grav_i_m": f"{p_meas['dx_grav_i_m']:.6f}",
                            "dy_grav_i_m": f"{p_meas['dy_grav_i_m']:.6f}",
                            "rel_yaw_ij_rad": f"{p_meas['rel_yaw_ij_rad']:.6f}",
                            "dx_cam_m": f"{p_meas['dx_cam_m']:.6f}",
                            "dy_cam_m": f"{p_meas['dy_cam_m']:.6f}",
                            "dz_cam_m": f"{p_meas['dz_cam_m']:.6f}",
                            "range_i_m": f"{p_meas['range_i_m']:.4f}",
                            "range_j_m": f"{p_meas['range_j_m']:.4f}",
                            "angle_i_deg": f"{p_meas['angle_i_deg']:.2f}",
                            "angle_j_deg": f"{p_meas['angle_j_deg']:.2f}",
                            "size_i_mm_measured": f"{p_meas['size_i_mm_measured']:.3f}",
                            "size_j_mm_measured": f"{p_meas['size_j_mm_measured']:.3f}",
                            "sigma_dist_m": f"{p_meas['sigma_dist_m']:.6f}",
                            "sigma_yaw_rad": f"{p_meas['sigma_yaw_rad']:.6f}",
                        }
                    )

        rgb_frames_for_video.append(rgb_img)

        if (
            use_custom_layout
            or s_idx % 2 == 0
            or s_idx in (0, 1, 5, 6, 10, 11, 15, 16, n_frames - 3, n_frames - 2, n_frames - 1)
        ):
            still_rel = f"stills/{s_idx:06d}.jpg"
            Image.fromarray(rgb_img, mode="RGB").save(out_dir / still_rel, quality=95)
            stills_rows.append(
                {
                    "t": f"{t_s:.6f}",
                    "saved_index": str(s_idx),
                    "file": still_rel,
                    "px": f"{cam_pos_w[0]:.6f}",
                    "py": f"{cam_pos_w[1]:.6f}",
                    "pz": f"{cam_pos_w[2]:.6f}",
                    "qx": f"{qx:.8f}",
                    "qy": f"{qy:.8f}",
                    "qz": f"{qz:.8f}",
                    "qw": f"{qw:.8f}",
                    "fx": f"{opts.fx:.4f}",
                    "fy": f"{opts.fy:.4f}",
                    "cx": f"{opts.cx:.4f}",
                    "cy": f"{opts.cy:.4f}",
                    "width": str(opts.rgb_width),
                    "height": str(opts.rgb_height),
                }
            )

    events_rows.append(
        {
            "t": f"{saved_timestamps[0]:.6f}",
            "saved_index": "0",
            "kind": "marker",
            "detail": "Marker 0 start",
        }
    )
    if opts.include_tape_checks:
        events_rows.append(
            {
                "t": f"{saved_timestamps[min(2, n_frames - 1)]:.6f}",
                "saved_index": str(min(2, n_frames - 1)),
                "kind": "tape_a",
                "detail": "Primary tape endpoint A",
            }
        )
        events_rows.append(
            {
                "t": f"{saved_timestamps[min(4, n_frames - 1)]:.6f}",
                "saved_index": str(min(4, n_frames - 1)),
                "kind": "tape_b",
                "detail": "Primary tape endpoint B (5.00m)",
            }
        )
    events_rows.append(
        {
            "t": f"{saved_timestamps[-1]:.6f}",
            "saved_index": str(n_frames - 1),
            "kind": "done",
            "detail": "Completed walk at Marker 0",
        }
    )

    _write_csv(
        out_dir / "frames.csv",
        ["t", "saved_index", "dropped", "drop_reason", "tracking_state", "tracking_reason"],
        frames_rows,
    )
    _write_csv(
        out_dir / "poses.csv",
        [
            "t", "saved_index", "px", "py", "pz", "qx", "qy", "qz", "qw",
            "fx", "fy", "cx", "cy", "width", "height",
        ],
        poses_rows,
    )
    _write_csv(
        out_dir / "stills.csv",
        [
            "t", "saved_index", "file", "px", "py", "pz", "qx", "qy", "qz", "qw",
            "fx", "fy", "cx", "cy", "width", "height",
        ],
        stills_rows,
    )
    markers_cols = (
        list(MARKERS_CSV_BASE_COLUMNS) + list(MARKERS_CSV_CAMERA_COLUMNS)
        if emit_cam_and_pairs
        else list(MARKERS_CSV_BASE_COLUMNS)
    )
    _write_csv(out_dir / "markers.csv", markers_cols, markers_rows)
    if emit_cam_and_pairs:
        _write_csv(out_dir / "pairs.csv", list(PAIR_CSV_COLUMNS), pairs_rows)
    _write_csv(
        out_dir / "events.csv",
        ["t", "saved_index", "kind", "detail"],
        events_rows,
    )

    _encode_rgb_mp4_vfr(
        out_dir / "rgb.mp4",
        rgb_frames_for_video,
        saved_timestamps,
        fps=opts.fps,
    )

    frame_span_s = max(0.5, all_timestamps[-1] - all_timestamps[0])
    _write_audio_wav(out_dir / "audio.wav", duration_s=frame_span_s, sample_rate=16000)

    north_samples: list[dict[str, Any]] = []
    if opts.location_status == "ok":
        for i in range(12):
            jitter = 0.6 * math.sin(i * 1.1)
            north_samples.append(
                {
                    "t_s": round(saved_timestamps[0] + i * 0.04, 6),
                    "camera_yaw_world_deg": 0.0,
                    "true_heading_deg": round((opts.true_north_yaw_deg + jitter) % 360.0, 3),
                    "heading_accuracy_deg": 6.0,
                }
            )

    tape_checks: list[dict[str, Any]] = []
    if opts.include_tape_checks:
        p_meas = 5.0 + opts.primary_tape_residual_m
        tape_checks.append(
            {
                "id": 0,
                "role": "primary",
                "t_a_s": round(saved_timestamps[0], 6),
                "t_b_s": round(saved_timestamps[min(4, n_frames - 1)], 6),
                "saved_index_a": 0,
                "saved_index_b": min(4, n_frames - 1),
                "point_a_world_m": [0.0, 0.0, 0.0],
                "point_b_world_m": [round(p_meas, 6), 0.0, 0.0],
                "tape_m": 5.0,
                "measured_m": round(p_meas, 6),
                "residual_m": round(abs(opts.primary_tape_residual_m), 6),
            }
        )
        held_out_specs = [
            (4.0, opts.held_out_residuals_m[0]),
            (5.0, opts.held_out_residuals_m[1]),
            (4.0, opts.held_out_residuals_m[2]),
        ]
        for idx, (tm, res) in enumerate(held_out_specs, start=1):
            hm = tm + res
            tape_checks.append(
                {
                    "id": idx,
                    "role": "held_out",
                    "t_a_s": round(saved_timestamps[0], 6),
                    "t_b_s": round(saved_timestamps[-1], 6),
                    "saved_index_a": 0,
                    "saved_index_b": n_frames - 1,
                    "point_a_world_m": [0.0, 0.0, 0.0],
                    "point_b_world_m": [round(hm, 6), 0.0, 0.0],
                    "tape_m": tm,
                    "measured_m": round(hm, 6),
                    "residual_m": round(abs(res), 6),
                }
            )

    t0_s = all_timestamps[0]
    t_end_s = all_timestamps[-1]
    span_s = t_end_s - t0_s
    sec_int = int(round(span_s))
    stop_min = sec_int // 60
    stop_sec = sec_int % 60
    stop_utc = f"2026-11-02T12:{stop_min:02d}:{stop_sec:02d}Z"

    session_data = {
        "contract": CONTRACT_VERSION,
        "capture_id": opts.capture_id,
        "device": {
            "model": "iPhone18,1",
            "name": "iPhone 17 Pro",
            "ios_version": "19.1",
            "lidar_supported": True,
        },
        "app_version": "0.1.0",
        "clock": {
            "started_at_utc": "2026-11-02T12:00:00Z",
            "t0_monotonic_s": t0_s,
            "start_clock_pair": {
                "t_s": t0_s,
                "utc": "2026-11-02T12:00:00Z",
            },
            "stop_clock_pair": {
                "t_s": t_end_s,
                "utc": stop_utc,
            },
            "audio_first_sample_t_s": t0_s,
            "audio_sample_rate_hz": 16000,
        },
        "camera_format": {
            "requested": {
                "width": opts.rgb_width,
                "height": opts.rgb_height,
                "fps": opts.fps,
            },
            "actual": {
                "width": opts.rgb_width,
                "height": opts.rgb_height,
                "fps": opts.fps,
            },
        },
        "pose_convention": "arkit_right_handed_y_up",
        "state": {
            "status": "stopped",
            "last_heartbeat_t_s": t_end_s,
            "stop_reason": "user_done",
            "failed_stream": None,
        },
        "location": {
            "status": opts.location_status,
            "latitude_deg": 37.7749 if opts.location_status == "ok" else None,
            "longitude_deg": -122.4194 if opts.location_status == "ok" else None,
            "horizontal_accuracy_m": 4.5 if opts.location_status == "ok" else None,
            "true_heading_deg": opts.true_north_yaw_deg if opts.location_status == "ok" else None,
            "magnetic_heading_deg": (opts.true_north_yaw_deg - 13.0) if opts.location_status == "ok" else None,
            "heading_accuracy_deg": 6.0 if opts.location_status == "ok" else None,
            "north_samples": north_samples,
        },
        "marker_config": {
            "dictionary": "DICT_5X5_250",
            "printed_sizes_mm": {
                str(k): float(opts.declared_marker_size_mm) for k in range(8)
            },
        },
        "guide_thresholds": {
            "pitch_min_deg": -45.0,
            "pitch_max_deg": -15.0,
            "roll_max_abs_deg": 15.0,
        },
        "loop_closure_gaps": [],
        "tape_checks": tape_checks,
        "provisional_grade": {
            "grade": "A",
            "distance_tolerance_10m_m": 0.035,
            "loop_closure_gap_m": round(math.hypot(*opts.loop_drift_xy_m), 4),
            "reasons": ["provisional_on_device"],
        },
    }
    (out_dir / "session.json").write_text(
        json.dumps(session_data, indent=2) + "\n", encoding="utf-8"
    )

    seal_manifest(out_dir, capture_id=opts.capture_id)
    return out_dir


def seal_manifest(capture_dir: Path, capture_id: str | None = None) -> Path:
    """Compute SHA-256 and byte sizes of all Level 1 files and write `manifest.json`."""
    if capture_id is None:
        sess = json.loads((capture_dir / "session.json").read_text(encoding="utf-8"))
        capture_id = str(sess["capture_id"])

    files_entries: list[dict[str, Any]] = []
    for p in sorted(capture_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(capture_dir).as_posix()
        if rel == "manifest.json" or rel.startswith("standard/") or rel.startswith("."):
            continue
        files_entries.append(
            {
                "path": rel,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )

    streams_declared = [
        "session",
        "frames",
        "rgb_video",
        "depth",
        "confidence",
        "poses",
        "audio",
        "events",
        "stills",
        "markers",
    ]
    if (capture_dir / "pairs.csv").exists():
        streams_declared.append("pairs")
    if (capture_dir / "guidance.csv").exists():
        streams_declared.append("guidance")
    manifest_data = {
        "contract": CONTRACT_VERSION,
        "capture_id": capture_id,
        "sealed_at_utc": "2026-11-02T12:00:05Z",
        "streams_declared": streams_declared,
        "files": files_entries,
    }
    manifest_path = capture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def pack_capture_zip(capture_dir: Path, zip_path: Path) -> Path:
    """Pack `capture_dir` into a store-mode (`ZIP_STORED`) `.zip` archive (`YS-22`)."""
    capture_dir = capture_dir.resolve()
    zip_path = zip_path.resolve()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(capture_dir.rglob("*")):
            if p.is_file():
                arcname = f"{capture_dir.name}/{p.relative_to(capture_dir).as_posix()}"
                zf.write(p, arcname=arcname)
    return zip_path


def unpack_capture_zip(zip_path: Path, output_parent_dir: Path) -> Path:
    """Unpack a capture `.zip` archive into `output_parent_dir` (`YS-22`)."""
    zip_path = zip_path.resolve()
    output_parent_dir = output_parent_dir.resolve()
    output_parent_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        names = zf.namelist()
        zf.extractall(output_parent_dir)
    if names:
        top = names[0].split("/")[0]
        candidate = output_parent_dir / top
        if candidate.is_dir():
            return candidate
    return output_parent_dir


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _encode_rgb_mp4_vfr(
    mp4_path: Path,
    rgb_frames: list[np.ndarray],
    timestamps_s: list[float],
    fps: int,
) -> None:
    """Encode RGB frames into an MP4 whose frame count and duration match `timestamps_s`."""
    h, w, _ = rgb_frames[0].shape
    n = len(rgb_frames)
    span_s = max(1.0 / fps, (timestamps_s[-1] - timestamps_s[0]) + (1.0 / fps))
    eff_fps = float(n) / span_s
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        f"{eff_fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        str(mp4_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None
    for frame in rgb_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait(timeout=30)
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def _write_audio_wav(wav_path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    n_samples = int(round(duration_s * sample_rate))
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    pcm = (np.sin(2.0 * math.pi * 440.0 * t) * 2000.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
