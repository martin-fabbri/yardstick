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

"""Convert a Level 1 raw phone recording into a Level 2 Standard Capture (`standard/`).

Implements YS-14, YS-21, YS-31, YS-32, YS-33, YS-41, YS-42, YS-43, YS-50, YS-51, YS-52, YS-53.
Notice (YS-43): `tape_checks` in `session.json` are NEVER consulted while building
`cloud.ply`, `frame.json`, `keyframes.json`, `trajectory.json`, or `coverage.json`.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from yardstick import CONTRACT_VERSION
from yardstick.geometry import (
    build_yard_frame,
    calibrate_north_in_yard_frame,
    cluster_marker_passes,
    compute_scanned_ground_coverage,
    detect_pose_jumps,
    fit_ground_plane_ransac,
    pose_to_matrix,
    unproject_depth_frame,
    voxel_downsample_point_cloud,
)
from yardstick.grading import compute_grade_and_tolerances
from yardstick.markers import evaluate_marker_scale_errors
from yardstick.ply import write_cloud_ply
from yardstick.validator import sha256_file, validate_recording


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def convert_recording_to_standard(
    capture_dir: Path,
    *,
    voxel_size_m: float = 0.025,
) -> dict[str, Any]:
    """Convert a Level 1 capture folder into `<capture_dir>/standard/`.

    Returns a dictionary containing `frame`, `grade`, `capture`, and `coverage`.
    """
    capture_dir = capture_dir.resolve()
    integrity_report = validate_recording(capture_dir)
    integrity_status = str(integrity_report["status"])

    session = _load_json(capture_dir / "session.json")
    capture_id = str(session["capture_id"])

    frames_rows = _read_csv(capture_dir / "frames.csv")
    poses_rows = _read_csv(capture_dir / "poses.csv")
    stills_rows = _read_csv(capture_dir / "stills.csv")
    markers_rows = _read_csv(capture_dir / "markers.csv")
    events_rows = _read_csv(capture_dir / "events.csv")

    # 1. Index camera poses by saved_index and compute tracking stats
    poses_by_index: dict[int, dict[str, Any]] = {}
    camera_poses_world: dict[int, np.ndarray] = {}
    camera_positions_list: list[np.ndarray] = []

    for row in poses_rows:
        s_idx = int(row["saved_index"])
        poses_by_index[s_idx] = row
        T_wc = pose_to_matrix(
            float(row["px"]),
            float(row["py"]),
            float(row["pz"]),
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"]),
        )
        camera_poses_world[s_idx] = T_wc
        camera_positions_list.append(T_wc[:3, 3])

    camera_positions_world = (
        np.stack(camera_positions_list, axis=0)
        if camera_positions_list
        else np.zeros((0, 3), dtype=np.float64)
    )

    saved_frame_rows = [r for r in frames_rows if r.get("dropped", "").strip().lower() != "true"]
    n_saved_frames = len(saved_frame_rows)
    n_limited = sum(
        1 for r in saved_frame_rows if r.get("tracking_state", "").strip() != "normal"
    )
    limited_tracking_share = (
        float(n_limited) / float(n_saved_frames) if n_saved_frames > 0 else 0.0
    )

    # Map stills by saved_index for RGB lookup during depth unprojection
    stills_by_saved: dict[int, dict[str, str]] = {}
    for srow in stills_rows:
        stills_by_saved[int(srow["saved_index"])] = srow

    # 2. Unproject LiDAR depth frames into ARKit world coordinates
    all_pts_w: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_ranges: list[np.ndarray] = []

    for s_idx in sorted(poses_by_index.keys()):
        prow = poses_by_index[s_idx]
        T_wc = camera_poses_world[s_idx]
        depth_path = capture_dir / "depth" / f"{s_idx:06d}.png"
        conf_path = capture_dir / "confidence" / f"{s_idx:06d}.png"
        if not depth_path.exists() or not conf_path.exists():
            continue

        with Image.open(depth_path) as d_img:
            depth_mm = np.array(d_img, dtype=np.uint16)
        with Image.open(conf_path) as c_img:
            conf_img = np.array(c_img, dtype=np.uint8)

        rgb_img: np.ndarray | None = None
        if s_idx in stills_by_saved:
            still_path = capture_dir / stills_by_saved[s_idx]["file"]
            if still_path.exists():
                with Image.open(still_path) as s_img:
                    rgb_img = np.array(s_img.convert("RGB"), dtype=np.uint8)

        pts_w, colors_u8, confs_u8, ranges_m = unproject_depth_frame(
            depth_mm=depth_mm,
            conf_img=conf_img,
            rgb_img=rgb_img,
            fx=float(prow["fx"]),
            fy=float(prow["fy"]),
            cx=float(prow["cx"]),
            cy=float(prow["cy"]),
            rgb_width=int(prow["width"]),
            rgb_height=int(prow["height"]),
            T_target_from_cam=T_wc,
            min_conf=1,
            min_range_m=0.20,
            max_range_m=4.50,
        )
        if pts_w.shape[0] > 0:
            all_pts_w.append(pts_w)
            all_rgb.append(colors_u8)
            all_conf.append(confs_u8)
            all_ranges.append(ranges_m)

    if all_pts_w:
        pts_world = np.vstack(all_pts_w)
        rgb_world = np.vstack(all_rgb)
        conf_world = np.concatenate(all_conf)
        ranges_world = np.concatenate(all_ranges)
    else:
        pts_world = np.zeros((0, 3), dtype=np.float64)
        rgb_world = np.zeros((0, 3), dtype=np.uint8)
        conf_world = np.zeros((0,), dtype=np.uint8)
        ranges_world = np.zeros((0,), dtype=np.float64)

    # 3. Cluster marker passes & detect pose jumps (YS-30, YS-31, YS-32)
    passes_by_marker = cluster_marker_passes(markers_rows, camera_poses_world)
    pose_jumps = detect_pose_jumps(poses_rows, jump_threshold_m=0.08, max_dt_s=0.10)

    marker_0_pos_world: np.ndarray | None = None
    if 0 in passes_by_marker and len(passes_by_marker[0]) >= 1:
        first_m0 = passes_by_marker[0][0]
        marker_0_pos_world = np.median(
            np.array(
                [[float(r["x"]), float(r["y"]), float(r["z"])] for r in first_m0],
                dtype=np.float64,
            ),
            axis=0,
        )

    # 4. Fit ground plane via gravity-gated RANSAC (YS-51)
    ground_normal_w, ground_d_w, ground_residual_m = fit_ground_plane_ransac(
        points_world=pts_world,
        camera_positions_world=camera_positions_world,
        marker_0_pos_world=marker_0_pos_world,
        max_tilt_deg=15.0,
        inlier_threshold_m=0.03,
    )

    first_cam_pose_w = (
        camera_poses_world[min(camera_poses_world.keys())]
        if camera_poses_world
        else np.eye(4, dtype=np.float64)
    )

    walk_start_t_s: float | None = float(poses_rows[0]["t"]) if poses_rows else None
    walk_end_t_s: float | None = float(poses_rows[-1]["t"]) if poses_rows else None
    non_done_tap_times = [
        float(er["t"]) for er in events_rows if er.get("kind", "").strip() not in ("", "done")
    ]
    last_tap_t_s: float | None = max(non_done_tap_times) if non_done_tap_times else None

    # 5. Build Yard Frame transform & calibrate North (YS-31, YS-32, YS-14)
    yard_frame_dict = build_yard_frame(
        ground_normal_world=ground_normal_w,
        ground_d_world=ground_d_w,
        ground_residual_m=ground_residual_m,
        passes_by_marker=passes_by_marker,
        first_camera_pose_world=first_cam_pose_w,
        pose_jumps=pose_jumps,
        walk_start_t_s=walk_start_t_s,
        walk_end_t_s=walk_end_t_s,
        last_tap_t_s=last_tap_t_s,
    )
    T_yard_from_world: np.ndarray = yard_frame_dict["T_matrix"]
    R_yw = T_yard_from_world[:3, :3]
    t_yw = T_yard_from_world[:3, 3]

    north_cal = calibrate_north_in_yard_frame(
        location_obj=session.get("location", {}),
        T_yard_from_world=T_yard_from_world,
    )

    frame_json_data = {
        "contract": CONTRACT_VERSION,
        "method": yard_frame_dict["method"],
        "marker_id": yard_frame_dict["marker_id"],
        "marker_0_normal_ok": yard_frame_dict["marker_0_normal_ok"],
        "pose_convention": "yard_right_handed_z_up",
        "world_to_yard_transform": yard_frame_dict["world_to_yard_transform"],
        "origin_height_m": yard_frame_dict["origin_height_m"],
        "origin_residual_m": yard_frame_dict["origin_residual_m"],
        "ground_plane_residual_m": yard_frame_dict["ground_plane_residual_m"],
        "north_calibration": north_cal,
        "loop_closure_gaps": yard_frame_dict["loop_closure_gaps"],
        "pose_jumps": yard_frame_dict["pose_jumps"],
    }

    # 6. Transform point cloud into Yard Frame & voxel downsample (YS-50)
    if pts_world.shape[0] > 0:
        pts_yard = (pts_world @ R_yw.T) + t_yw
    else:
        pts_yard = np.zeros((0, 3), dtype=np.float64)

    ds_xyz, ds_rgb, ds_conf, ds_ranges = voxel_downsample_point_cloud(
        points_xyz=pts_yard,
        colors_rgb=rgb_world,
        confidences=conf_world,
        ranges_m=ranges_world,
        voxel_size_m=voxel_size_m,
    )

    # 7. Compute scanned ground coverage polygon & holes (YS-52)
    coverage_json_data = compute_scanned_ground_coverage(
        points_xyz=ds_xyz.astype(np.float64),
        confidences=ds_conf,
        ranges_m=ds_ranges.astype(np.float64),
    )

    # 8. Build `trajectory.json` and compute total walk length
    traj_poses_list: list[dict[str, Any]] = []
    walk_length_m = 0.0
    prev_pos_yard: np.ndarray | None = None

    for s_idx in sorted(poses_by_index.keys()):
        prow = poses_by_index[s_idx]
        T_wc = camera_poses_world[s_idx]
        T_yc = T_yard_from_world @ T_wc
        pos_y = T_yc[:3, 3]
        if prev_pos_yard is not None:
            walk_length_m += float(np.linalg.norm(pos_y - prev_pos_yard))
        prev_pos_yard = pos_y
        traj_poses_list.append(
            {
                "t": round(float(prow["t"]), 6),
                "saved_index": s_idx,
                "pose": [round(float(x), 8) for x in T_yc.flatten()],
            }
        )

    trajectory_json_data = {
        "contract": CONTRACT_VERSION,
        "poses": traj_poses_list,
    }

    # 9. Build `standard/` directory and write geometry files
    std_dir = capture_dir / "standard"
    std_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = std_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    # 9a. `standard/cloud.ply`
    write_cloud_ply(std_dir / "cloud.ply", ds_xyz, ds_conf, ds_rgb, ds_ranges)

    # 9b. `standard/frame.json`
    _write_json(std_dir / "frame.json", frame_json_data)

    # 9c. `standard/trajectory.json`
    _write_json(std_dir / "trajectory.json", trajectory_json_data)

    # 9d. `standard/coverage.json`
    _write_json(std_dir / "coverage.json", coverage_json_data)

    # 9e. Copy keyframes and write `standard/keyframes.json`
    keyframes_list: list[dict[str, Any]] = []
    for kf_id, srow in enumerate(stills_rows):
        s_idx = int(srow["saved_index"])
        t_val = float(srow["t"])
        src_rel = srow["file"]
        src_path = capture_dir / src_rel
        dst_rel = f"keyframes/{s_idx:06d}.jpg"
        dst_path = std_dir / dst_rel
        if src_path.exists():
            shutil.copy2(src_path, dst_path)

        T_wc = pose_to_matrix(
            float(srow["px"]),
            float(srow["py"]),
            float(srow["pz"]),
            float(srow["qx"]),
            float(srow["qy"]),
            float(srow["qz"]),
            float(srow["qw"]),
        )
        T_yc = T_yard_from_world @ T_wc
        keyframes_list.append(
            {
                "id": kf_id,
                "t": round(t_val, 6),
                "saved_index": s_idx,
                "file": dst_rel,
                "pose": [round(float(x), 8) for x in T_yc.flatten()],
                "intrinsics": {
                    "fx": round(float(srow["fx"]), 4),
                    "fy": round(float(srow["fy"]), 4),
                    "cx": round(float(srow["cx"]), 4),
                    "cy": round(float(srow["cy"]), 4),
                    "width": int(srow["width"]),
                    "height": int(srow["height"]),
                },
            }
        )

    keyframes_json_data = {
        "contract": CONTRACT_VERSION,
        "keyframes": keyframes_list,
    }
    _write_json(std_dir / "keyframes.json", keyframes_json_data)

    # 9f. Copy `audio.wav` and write `standard/events.json`
    src_audio = capture_dir / "audio.wav"
    dst_audio = std_dir / "audio.wav"
    if src_audio.exists():
        shutil.copy2(src_audio, dst_audio)

    std_events: list[dict[str, Any]] = []
    for erow in events_rows:
        std_events.append(
            {
                "t": round(float(erow["t"]), 6),
                "saved_index": int(erow["saved_index"]),
                "kind": erow["kind"],
                "detail": erow["detail"],
            }
        )
    events_json_data = {
        "contract": CONTRACT_VERSION,
        "audio_file": "audio.wav",
        "audio_first_sample_t_s": float(session["clock"]["audio_first_sample_t_s"]),
        "audio_sample_rate_hz": int(session["clock"]["audio_sample_rate_hz"]),
        "events": std_events,
    }
    _write_json(std_dir / "events.json", events_json_data)

    # 10. Compute marker scale error & `standard/grade.json` (YS-33, YS-41, YS-42, YS-43)
    # Notice (YS-43): `tape_checks` are ONLY read here, AFTER all geometry files have been written!
    marker_scale_info = evaluate_marker_scale_errors(
        capture_dir=capture_dir,
        markers_rows=markers_rows,
        poses_by_index=poses_by_index,
        passes_by_marker=passes_by_marker,
    )
    tape_checks_world = session.get("tape_checks", [])
    grade_json_data = compute_grade_and_tolerances(
        integrity_status=integrity_status,
        frame_info=yard_frame_dict,
        marker_scale_info=marker_scale_info,
        tape_checks_world=tape_checks_world,
        T_yard_from_world=T_yard_from_world,
        walk_length_m=walk_length_m,
        limited_tracking_share=limited_tracking_share,
    )
    _write_json(std_dir / "grade.json", grade_json_data)

    # 11. Seal `standard/capture.json` (YS-21, Q023)
    manifest_path = capture_dir / "manifest.json"
    rec_manifest_sha256 = (
        sha256_file(manifest_path) if manifest_path.exists() else "0" * 64
    )

    std_files_entries: list[dict[str, Any]] = []
    for fpath in sorted(std_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(std_dir).as_posix()
        if rel == "capture.json" or rel.startswith("."):
            continue
        std_files_entries.append(
            {
                "path": rel,
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
            }
        )

    dev = session["device"]
    capture_json_data = {
        "contract": CONTRACT_VERSION,
        "capture_id": capture_id,
        "recording_manifest_sha256": rec_manifest_sha256,
        "captured_at_utc": session["clock"]["started_at_utc"],
        "source": {
            "app": "Yardstick",
            "app_version": session["app_version"],
            "device_model": dev["model"],
            "device_name": dev["name"],
            "ios_version": dev["ios_version"],
            "sensor": "lidar",
        },
        "location": session["location"],
        "voxel_size_m": voxel_size_m,
        "integrity": integrity_report,
        "files": std_files_entries,
    }
    _write_json(std_dir / "capture.json", capture_json_data)

    return {
        "frame": frame_json_data,
        "grade": grade_json_data,
        "capture": capture_json_data,
        "coverage": coverage_json_data,
    }
