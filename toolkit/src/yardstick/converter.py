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

Implements YS-14, YS-21, YS-31, YS-32, YS-33, YS-41, YS-42, YS-43, YS-50, YS-51, YS-52, YS-53, YS-115.
Notice (YS-43): `tape_checks` in `session.json` are NEVER consulted while building
`cloud.ply`, `cloud.raw.ply`, `frame.json`, `keyframes.json`, `trajectory.json`, `trajectory.raw.json`, or `coverage.json`.
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
    warp_trajectory_to_marker_layout,
)
from yardstick.grading import compute_grade_and_tolerances
from yardstick.markers import (
    detect_aruco_markers_opencv,
    evaluate_marker_scale_errors,
    extract_one_frame_pairs,
    solve_marker_layout_graph,
)
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


def _is_operator_tap_kind(kind: str) -> bool:
    """Return True only for operator tap event kinds (`marker`, `note`, `tape_*`) (YS-67)."""
    k = kind.strip()
    return k in ("marker", "note") or k.startswith("tape_")


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
    guidance_rows = _read_csv(capture_dir / "guidance.csv")

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

    # 2. Unproject LiDAR depth frames in camera coordinates & ARKit world coordinates
    eye4 = np.eye(4, dtype=np.float64)
    per_frame_cam_clouds: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
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

        try:
            with Image.open(depth_path) as d_img:
                depth_mm = np.array(d_img, dtype=np.uint16)
            with Image.open(conf_path) as c_img:
                conf_img = np.array(c_img, dtype=np.uint8)
        except Exception:
            integrity_status = "invalid"
            continue

        if depth_mm.ndim != 2 or conf_img.ndim != 2 or depth_mm.shape != conf_img.shape:
            integrity_status = "invalid"
            continue

        rgb_img: np.ndarray | None = None
        if s_idx in stills_by_saved:
            still_path = capture_dir / stills_by_saved[s_idx]["file"]
            if still_path.exists():
                try:
                    with Image.open(still_path) as s_img:
                        rgb_img = np.array(s_img.convert("RGB"), dtype=np.uint8)
                except Exception:
                    integrity_status = "invalid"
                    rgb_img = None

        pts_cam, colors_u8, confs_u8, ranges_m = unproject_depth_frame(
            depth_mm=depth_mm,
            conf_img=conf_img,
            rgb_img=rgb_img,
            fx=float(prow["fx"]),
            fy=float(prow["fy"]),
            cx=float(prow["cx"]),
            cy=float(prow["cy"]),
            rgb_width=int(prow["width"]),
            rgb_height=int(prow["height"]),
            T_target_from_cam=eye4,
            min_conf=1,
            min_range_m=0.20,
            max_range_m=4.50,
        )
        if pts_cam.shape[0] > 0:
            per_frame_cam_clouds.append((s_idx, pts_cam, colors_u8, confs_u8, ranges_m))
            pts_w = (pts_cam @ T_wc[:3, :3].T) + T_wc[:3, 3]
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
    res_ransac = fit_ground_plane_ransac(
        points_world=pts_world,
        camera_positions_world=camera_positions_world,
        marker_0_pos_world=marker_0_pos_world,
        max_tilt_deg=15.0,
        inlier_threshold_m=0.03,
        return_p95=True,
    )
    assert len(res_ransac) == 4
    ground_normal_w, ground_d_w, ground_residual_m, ground_residual_p95_m = res_ransac

    first_cam_pose_w = (
        camera_poses_world[min(camera_poses_world.keys())]
        if camera_poses_world
        else np.eye(4, dtype=np.float64)
    )

    walk_start_t_s: float | None = float(poses_rows[0]["t"]) if poses_rows else None
    walk_end_t_s: float | None = float(poses_rows[-1]["t"]) if poses_rows else None
    loop_closed_times = [
        float(r["t"])
        for r in (events_rows + guidance_rows)
        if (r.get("kind") or r.get("event") or "").strip() == "loop_closed"
    ]
    loop_closed_t_s: float | None = max(loop_closed_times) if loop_closed_times else None
    operator_tap_times = [
        float(er["t"])
        for er in events_rows
        if _is_operator_tap_kind(er.get("kind", ""))
        and (loop_closed_t_s is None or float(er["t"]) <= loop_closed_t_s + 1e-6)
    ]
    last_tap_t_s: float | None = max(operator_tap_times) if operator_tap_times else None

    # 5. Build Yard Frame transform & calibrate North (YS-31, YS-32, YS-14, YS-67)
    confirmed_marker0_still_indices: set[int] = set()
    m0_candidate_indices = {
        int(mr["saved_index"]) for mr in markers_rows if int(mr["marker_id"]) == 0
    }
    for s_idx in m0_candidate_indices:
        if s_idx not in stills_by_saved:
            continue
        still_path = capture_dir / stills_by_saved[s_idx]["file"]
        if not still_path.exists():
            continue
        try:
            with Image.open(still_path) as s_img:
                gray = np.array(s_img.convert("L"), dtype=np.uint8)
            dets = detect_aruco_markers_opencv(gray)
            if 0 in dets:
                confirmed_marker0_still_indices.add(s_idx)
        except Exception:
            continue

    yard_frame_dict = build_yard_frame(
        ground_normal_world=ground_normal_w,
        ground_d_world=ground_d_w,
        ground_residual_m=ground_residual_m,
        ground_residual_p95_m=ground_residual_p95_m,
        passes_by_marker=passes_by_marker,
        first_camera_pose_world=first_cam_pose_w,
        pose_jumps=pose_jumps,
        walk_start_t_s=walk_start_t_s,
        walk_end_t_s=walk_end_t_s,
        last_tap_t_s=last_tap_t_s,
        loop_closed_t_s=loop_closed_t_s,
        camera_poses_by_index=camera_poses_world,
        confirmed_marker0_still_indices=confirmed_marker0_still_indices,
        max_closing_distance_m=1.50,
    )
    T_yard_from_world: np.ndarray = yard_frame_dict["T_matrix"]
    R_yw = T_yard_from_world[:3, :3]
    t_yw = T_yard_from_world[:3, 3]

    north_cal = calibrate_north_in_yard_frame(
        location_obj=session.get("location", {}),
        T_yard_from_world=T_yard_from_world,
    )

    # 5b. Evaluate printed-marker scale (never reads tape_checks) & solve one-frame pair layout graph (YS-115)
    marker_scale_info = evaluate_marker_scale_errors(
        capture_dir=capture_dir,
        markers_rows=markers_rows,
        poses_by_index=poses_by_index,
        passes_by_marker=passes_by_marker,
    )
    pair_rows = extract_one_frame_pairs(
        capture_dir=capture_dir,
        markers_rows=markers_rows,
        poses_by_index=poses_by_index,
        ground_normal_world=ground_normal_w,
    )
    layout_res = solve_marker_layout_graph(
        markers_rows=markers_rows,
        pair_rows=pair_rows,
        passes_by_marker=passes_by_marker,
        T_yard_from_world=T_yard_from_world,
        origin_height_m=float(yard_frame_dict["origin_height_m"]),
        marker_scale_info=marker_scale_info,
    )
    scale_applied = float(layout_res["marker_layout"]["scale_applied"])

    warp_res = warp_trajectory_to_marker_layout(
        camera_poses_world=camera_poses_world,
        poses_by_index=poses_by_index,
        T_yard_from_world=T_yard_from_world,
        markers_rows=markers_rows,
        passes_by_marker=passes_by_marker,
        solved_positions_yard=layout_res["solved_positions_yard"],
        vio_similarity=layout_res["vio_similarity"],
        scale_applied=scale_applied,
    )
    warped_poses_yard: dict[int, np.ndarray] = warp_res["warped_poses_yard"]
    raw_poses_yard: dict[int, np.ndarray] = warp_res["raw_poses_yard"]

    layout_res["marker_layout"]["trajectory_warp_applied"] = bool(warp_res["trajectory_warp_applied"])
    layout_res["marker_layout"]["trajectory_warp_method"] = str(warp_res["trajectory_warp_method"])
    layout_res["marker_layout"]["per_marker_warp_corrections"] = warp_res["per_marker_warp_corrections"]

    yard_frame_dict["markers_yard"] = layout_res["markers_yard"]
    yard_frame_dict["marker_layout"] = layout_res["marker_layout"]

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
        "ground_plane_residual_p95_m": yard_frame_dict["ground_plane_residual_p95_m"],
        "north_calibration": north_cal,
        "loop_closure_gaps": yard_frame_dict["loop_closure_gaps"],
        "pose_jumps": yard_frame_dict["pose_jumps"],
        "pose_jump_window_t_s": yard_frame_dict["pose_jump_window_t_s"],
        "markers_yard": layout_res["markers_yard"],
        "marker_layout": layout_res["marker_layout"],
    }

    # 6. Transform point cloud into Yard Frame (both raw unwarped and default warped) & voxel downsample (YS-50, YS-115 Condition 4)
    if pts_world.shape[0] > 0:
        pts_yard_raw = (pts_world @ R_yw.T) + t_yw
    else:
        pts_yard_raw = np.zeros((0, 3), dtype=np.float64)

    ds_raw_xyz, ds_raw_rgb, ds_raw_conf, ds_raw_ranges = voxel_downsample_point_cloud(
        points_xyz=pts_yard_raw,
        colors_rgb=rgb_world,
        confidences=conf_world,
        ranges_m=ranges_world,
        voxel_size_m=voxel_size_m,
    )

    if per_frame_cam_clouds and warp_res["trajectory_warp_applied"]:
        warped_pts_list: list[np.ndarray] = []
        warped_ranges_list: list[np.ndarray] = []
        for s_idx, pts_cam, _, _, ranges_m in per_frame_cam_clouds:
            T_yc_w = warped_poses_yard.get(s_idx, raw_poses_yard[s_idx])
            pts_cam_scaled = scale_applied * pts_cam
            pts_y_w = (pts_cam_scaled @ T_yc_w[:3, :3].T) + T_yc_w[:3, 3]
            warped_pts_list.append(pts_y_w)
            warped_ranges_list.append(scale_applied * ranges_m)
        pts_yard_warped = np.vstack(warped_pts_list)
        ranges_warped = np.concatenate(warped_ranges_list)
        ds_xyz, ds_rgb, ds_conf, ds_ranges = voxel_downsample_point_cloud(
            points_xyz=pts_yard_warped,
            colors_rgb=rgb_world,
            confidences=conf_world,
            ranges_m=ranges_warped,
            voxel_size_m=voxel_size_m,
        )
    else:
        ds_xyz, ds_rgb, ds_conf, ds_ranges = ds_raw_xyz, ds_raw_rgb, ds_raw_conf, ds_raw_ranges

    # 7. Compute scanned ground coverage polygon & holes (YS-52)
    try:
        coverage_json_data = compute_scanned_ground_coverage(
            points_xyz=ds_xyz.astype(np.float64),
            confidences=ds_conf,
            ranges_m=ds_ranges.astype(np.float64),
        )
    except Exception:
        integrity_status = "invalid"
        coverage_json_data = {
            "contract": CONTRACT_VERSION,
            "polygon_xy_m": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
            "holes_xy_m": [],
            "area_m2": 0.01,
            "high_confidence_share": 0.0,
            "parameters": {
                "z_max_abs_m": 0.25,
                "range_max_m": 4.0,
                "grid_cell_m": 0.20,
                "closing_cells": 1,
                "max_filled_hole_m2": 1.0,
                "simplify_epsilon_m": 0.10,
            },
        }

    # 8. Build `trajectory.json` (warped default) and `trajectory.raw.json` (unwarped original)
    traj_poses_list: list[dict[str, Any]] = []
    traj_raw_poses_list: list[dict[str, Any]] = []
    walk_length_m = 0.0
    prev_pos_yard: np.ndarray | None = None

    for s_idx in sorted(poses_by_index.keys()):
        prow = poses_by_index[s_idx]
        T_yc_raw = raw_poses_yard[s_idx]
        T_yc_warped = warped_poses_yard.get(s_idx, T_yc_raw)
        pos_y = T_yc_warped[:3, 3]
        if prev_pos_yard is not None:
            walk_length_m += float(np.linalg.norm(pos_y - prev_pos_yard))
        prev_pos_yard = pos_y
        traj_poses_list.append(
            {
                "t": round(float(prow["t"]), 6),
                "saved_index": s_idx,
                "pose": [round(float(x), 8) for x in T_yc_warped.flatten()],
            }
        )
        traj_raw_poses_list.append(
            {
                "t": round(float(prow["t"]), 6),
                "saved_index": s_idx,
                "pose": [round(float(x), 8) for x in T_yc_raw.flatten()],
            }
        )

    trajectory_json_data = {
        "contract": CONTRACT_VERSION,
        "poses": traj_poses_list,
    }
    trajectory_raw_json_data = {
        "contract": CONTRACT_VERSION,
        "poses": traj_raw_poses_list,
    }

    # 9. Build `standard/` directory and write geometry files
    std_dir = capture_dir / "standard"
    std_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = std_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    # 9a. `standard/cloud.ply` (warped default) and `standard/cloud.raw.ply` (unwarped original, Condition 4)
    write_cloud_ply(std_dir / "cloud.ply", ds_xyz, ds_conf, ds_rgb, ds_ranges)
    write_cloud_ply(std_dir / "cloud.raw.ply", ds_raw_xyz, ds_raw_conf, ds_raw_rgb, ds_raw_ranges)

    # 9b. `standard/frame.json`
    _write_json(std_dir / "frame.json", frame_json_data)

    # 9c. `standard/trajectory.json` and `standard/trajectory.raw.json`
    _write_json(std_dir / "trajectory.json", trajectory_json_data)
    _write_json(std_dir / "trajectory.raw.json", trajectory_raw_json_data)

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

        if s_idx in warped_poses_yard:
            T_yc = warped_poses_yard[s_idx]
        else:
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

    # 10. Compute `standard/grade.json` (YS-33, YS-41, YS-42, YS-43, YS-115)
    # Notice (YS-43): `tape_checks` are ONLY read here, AFTER all geometry files have been written!
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
