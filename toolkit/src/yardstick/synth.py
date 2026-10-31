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
that exercises every requirement in `YS-00` through `YS-53` without needing
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

    # Markers in ARKit world on ground Y_w = 0:
    markers_world: dict[int, np.ndarray] = {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        1: np.array([5.0, 0.0, 0.0], dtype=np.float64),
        2: np.array([5.0, 0.0, -4.0], dtype=np.float64),
        3: np.array([0.0, 0.0, -4.0], dtype=np.float64),
    }
    if not opts.include_marker_0:
        markers_world.pop(0, None)

    n_frames = opts.num_saved_frames
    # Camera height = 0.90 m so at 30 deg downward pitch the optical center hits ground at 1.56 m horizontal!
    cam_height_m = 0.90
    perimeter_waypoints = [
        (-1.10, 0.0, 0.0),            # 0: Looking at Marker 0 at [0,0,0]
        (-1.05, 0.0, 0.0),            # 1: Pass 1 on Marker 0
        (0.5, -0.5, 0.0),
        (1.5, -0.5, 0.0),
        (2.5, -0.5, 0.0),
        (3.90, 0.0, 0.0),             # 5: Looking at Marker 1 at [5,0,0]
        (3.95, 0.0, 0.0),             # 6: Pass on Marker 1
        (5.0, -1.0, -math.pi / 2),
        (5.0, -2.0, -math.pi / 2),
        (5.0, -2.90, -math.pi / 2),   # 9: Looking at Marker 2 at [5,0,-4]
        (5.0, -2.95, -math.pi / 2),   # 10: Pass on Marker 2
        (4.0, -4.0, math.pi),
        (3.0, -4.0, math.pi),
        (2.0, -4.0, math.pi),
        (1.10, -4.0, math.pi),        # 14: Looking at Marker 3 at [0,0,-4]
        (1.05, -4.0, math.pi),        # 15: Pass on Marker 3
        (0.0, -3.2, math.pi / 2),
        (0.0, -2.4, math.pi / 2),
        (0.0, -1.6, math.pi / 2),
        (0.0, -0.8, math.pi / 2),
        (1.5, -2.5, 0.4),
        (-1.12, 0.0, 0.0),            # 21: Return pass at Marker 0
        (-1.08, 0.0, 0.0),            # 22: Return pass at Marker 0
        (-1.02, 0.0, 0.0),            # 23: Final frame on Marker 0
    ]
    waypoints: list[tuple[float, float, float]] = []
    for i in range(n_frames):
        idx = min(i, len(perimeter_waypoints) - 1)
        waypoints.append(perimeter_waypoints[idx])

    frames_rows: list[dict[str, Any]] = []
    poses_rows: list[dict[str, Any]] = []
    stills_rows: list[dict[str, Any]] = []
    markers_rows: list[dict[str, Any]] = []
    events_rows: list[dict[str, Any]] = []
    rgb_frames_for_video: list[np.ndarray] = []
    saved_timestamps: list[float] = []
    all_timestamps: list[float] = []

    dt_step = 1.0 / float(opts.fps)
    t_cursor = 100.0

    marker_patches = {mid: _render_aruco_marker_patch(mid, 240) for mid in range(4)}

    for s_idx in range(n_frames):
        # Optional upstream dropped frame before saved_index == 3 (YS-02)
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

        # Ensure > 3.0s gap before the return pass so cluster_marker_passes separates Pass 1 and Pass 2
        if s_idx == n_frames - 2:
            t_cursor += 3.5

        # If pose_jump_m > 0, make dt <= 0.08 s on the jump frame so detect_pose_jumps triggers!
        if opts.pose_jump_m > 0.0 and s_idx == n_frames // 2:
            t_s = round(saved_timestamps[-1] + 0.05, 6)
            t_cursor = t_s + dt_step
        else:
            t_s = round(t_cursor, 6)
            t_cursor += dt_step

        saved_timestamps.append(t_s)
        all_timestamps.append(t_s)

        wx, wz, yaw_rad = waypoints[s_idx]
        if opts.pose_jump_m > 0.0 and s_idx == n_frames // 2:
            wx = waypoints[s_idx - 1][0] + opts.pose_jump_m
            wz = waypoints[s_idx - 1][1]

        cam_pos_w = np.array([wx, cam_height_m, wz], dtype=np.float64)
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

        denom = rays_w[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_depth_m = np.where(denom < -1e-4, -cam_pos_w[1] / denom, 0.0)

        pts_hit_w = cam_pos_w.reshape(1, 1, 3) + z_depth_m[..., None] * rays_w

        # Optional vertical wall at X_w = 6.2 m (YS-51)
        if opts.include_wall_points and s_idx in (4, 5):
            with np.errstate(divide="ignore", invalid="ignore"):
                z_wall = np.where(
                    rays_w[..., 0] > 1e-4, (6.2 - cam_pos_w[0]) / rays_w[..., 0], 999.0
                )
            wall_y = cam_pos_w[1] + z_wall * rays_w[..., 1]
            use_wall = (
                (z_wall > 0.3)
                & (z_wall < z_depth_m)
                & (wall_y >= 0.0)
                & (wall_y <= 2.2)
            )
            z_depth_m = np.where(use_wall, z_wall, z_depth_m)
            pts_hit_w = cam_pos_w.reshape(1, 1, 3) + z_depth_m[..., None] * rays_w

        ranges_m = z_depth_m * np.sqrt(1.0 + ray_x_c * ray_x_c + ray_y_c * ray_y_c)
        valid_range = (ranges_m >= 0.25) & (ranges_m <= 3.95)

        # Carve an unscanned center void (> 1.0 m^2) at X_w in [1.6, 3.4], Z_w in [-2.9, -1.1] (1.8m x 1.8m = 3.24 m^2)
        if opts.include_center_void_hole:
            in_void = (
                (pts_hit_w[..., 0] >= 1.6)
                & (pts_hit_w[..., 0] <= 3.4)
                & (pts_hit_w[..., 2] >= -2.9)
                & (pts_hit_w[..., 2] <= -1.1)
            )
            valid_range = valid_range & (~in_void)

        depth_mm = np.where(valid_range, np.round(z_depth_m * 1000.0), 0).astype(np.uint16)
        conf_u8 = np.where(valid_range, 2, 0).astype(np.uint8)

        Image.fromarray(depth_mm).save(out_dir / "depth" / f"{s_idx:06d}.png")
        Image.fromarray(conf_u8).save(out_dir / "confidence" / f"{s_idx:06d}.png")

        # Synthesize RGB image with grid texture and visible ArUco markers
        rgb_img = np.full((opts.rgb_height, opts.rgb_width, 3), 185, dtype=np.uint8)
        rgb_img[::32, :, :] = 70
        rgb_img[:, ::32, :] = 70

        r_c_w = r_w_c.T
        for mid, m_pos_w in markers_world.items():
            is_return_pass = mid == 0 and s_idx >= n_frames - 2
            if is_return_pass and not opts.include_loop_closure:
                continue
            m_center = m_pos_w.copy()
            if is_return_pass:
                m_center[0] += opts.loop_drift_xy_m[0]
                m_center[2] -= opts.loop_drift_xy_m[1]

            # 4 corners of marker on ground Y_w = 0: TL, TR, BR, BL
            corners_local = np.array(
                [
                    [-half_s, +half_s, 0.0],
                    [+half_s, +half_s, 0.0],
                    [+half_s, -half_s, 0.0],
                    [-half_s, -half_s, 0.0],
                ],
                dtype=np.float64,
            )
            # Marker rotation in ARKit world: +X_m = [1,0,0], +Y_m = [0,0,-1], +Z_m (normal) = [0,1,0]
            r_w_m = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=np.float64,
            )

            corners_w = m_center.reshape(1, 3) + corners_local @ r_w_m.T
            corners_c = (corners_w - cam_pos_w.reshape(1, 3)) @ r_c_w.T
            depths_c = -corners_c[:, 2]

            if np.all((depths_c >= 0.60) & (depths_c <= 2.00)):
                u_px = opts.cx + opts.fx * (corners_c[:, 0] / depths_c)
                v_px = opts.cy - opts.fy * (corners_c[:, 1] / depths_c)
                if np.all(
                    (u_px >= 12)
                    & (u_px < opts.rgb_width - 12)
                    & (v_px >= 12)
                    & (v_px < opts.rgb_height - 12)
                ):
                    patch = marker_patches[mid]
                    ph, pw = patch.shape[:2]
                    pad_q = float(pw) * 0.1
                    src_pts = np.array(
                        [
                            [pad_q, pad_q],
                            [pw - pad_q, pad_q],
                            [pw - pad_q, ph - pad_q],
                            [pad_q, ph - pad_q],
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
                        borderValue=(185, 185, 185),
                    )
                    mask = np.zeros((opts.rgb_height, opts.rgb_width), dtype=np.uint8)
                    full_box = np.array(
                        [[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32
                    ).reshape(-1, 1, 2)
                    dst_box = cv2.perspectiveTransform(full_box, h_mat).astype(np.int32)
                    cv2.fillConvexPoly(mask, dst_box, 255)
                    rgb_img[mask > 0] = warped[mask > 0]

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
                    meas_mm = opts.actual_marker_size_mm + jitter_mm

                    markers_rows.append(
                        {
                            "t": f"{t_s:.6f}",
                            "saved_index": str(s_idx),
                            "marker_id": str(mid),
                            "size_mm_printed": f"{opts.declared_marker_size_mm:.2f}",
                            "size_mm_measured": f"{meas_mm:.3f}",
                            "x": f"{m_center[0]:.6f}",
                            "y": f"{m_center[1]:.6f}",
                            "z": f"{m_center[2]:.6f}",
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
                    )

        rgb_frames_for_video.append(rgb_img)

        # Save still every 2 frames or on marker frames
        if s_idx % 2 == 0 or s_idx in (0, 1, 5, 6, 9, 10, 14, 15, n_frames - 3, n_frames - 2, n_frames - 1):
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

    # Events (`t,saved_index,kind,detail` where kind in marker|tape_a|tape_b|done|note)
    events_rows.append(
        {
            "t": f"{saved_timestamps[0]:.6f}",
            "saved_index": "0",
            "kind": "marker",
            "detail": "Marker 0 start",
        }
    )
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

    # Write CSVs
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
    _write_csv(
        out_dir / "markers.csv",
        [
            "t", "saved_index", "marker_id", "size_mm_printed", "size_mm_measured",
            "x", "y", "z", "qx", "qy", "qz", "qw",
            "c0_u", "c0_v", "c1_u", "c1_v", "c2_u", "c2_v", "c3_u", "c3_v",
        ],
        markers_rows,
    )
    _write_csv(
        out_dir / "events.csv",
        ["t", "saved_index", "kind", "detail"],
        events_rows,
    )

    # Encode `rgb.mp4` with VFR presentation timestamps matching `saved_timestamps`
    _encode_rgb_mp4_vfr(
        out_dir / "rgb.mp4",
        rgb_frames_for_video,
        saved_timestamps,
        fps=opts.fps,
    )

    # Write `audio.wav` (16 kHz mono 16-bit PCM matching frame span)
    frame_span_s = max(0.5, all_timestamps[-1] - all_timestamps[0])
    _write_audio_wav(out_dir / "audio.wav", duration_s=frame_span_s, sample_rate=16000)

    # Build north samples (12 pairs at Marker 0)
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

    # Build tape checks (1 primary + 3 held-out)
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

    # Compute UTC stop timestamp matching monotonic span
    t0_s = all_timestamps[0]
    t_end_s = all_timestamps[-1]
    span_s = t_end_s - t0_s
    sec_int = int(round(span_s))
    stop_utc = f"2026-11-02T12:00:{sec_int:02d}Z"

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
                str(k): float(opts.declared_marker_size_mm) for k in range(6)
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
