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

"""Validator for Level 1 recordings and Level 2 standard captures (YS-01..05, YS-20..23, YS-53)."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np
from PIL import Image

from yardstick.geometry import is_orthonormal_se3, is_simple_polygon, is_valid_unit_quaternion, pose_to_matrix
from yardstick.grading import recompute_tape_check
from yardstick.markers import (
    MARKERS_CSV_BASE_COLUMNS,
    MARKERS_CSV_CAMERA_COLUMNS,
    PAIR_CSV_COLUMNS,
    detect_aruco_markers_opencv,
)
from yardstick.ply import read_cloud_ply

FORBIDDEN_INTERPRETATION_KEYS = {
    "objects",
    "object_labels",
    "labels",
    "lot_lines",
    "setbacks",
    "constraints",
    "verdict",
    "verdicts",
    "feasibility",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_schema(schema_name: str) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = repo_root / "contract" / schema_name
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _check_no_forbidden_keys(obj: Any, path_prefix: str = "") -> list[str]:
    """Recursively check JSON structure for forbidden interpretation keys (YS-53)."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_k = f"{path_prefix}.{k}" if path_prefix else k
            if k.lower() in FORBIDDEN_INTERPRETATION_KEYS:
                hits.append(full_k)
            hits.extend(_check_no_forbidden_keys(v, full_k))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            hits.extend(_check_no_forbidden_keys(item, f"{path_prefix}[{idx}]"))
    return hits


def probe_video_frame_count_and_duration(video_path: Path) -> tuple[int, float]:
    """Return (decoded_frame_count, duration_s) for rgb.mp4 using ffprobe or OpenCV."""
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,duration:format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        info = json.loads(res.stdout)
        streams = info.get("streams", [])
        fmt = info.get("format", {})
        nb_frames = int(streams[0].get("nb_read_frames", 0)) if streams else 0
        dur_str = streams[0].get("duration") if streams else None
        if dur_str is None or dur_str == "N/A":
            dur_str = fmt.get("duration", "0.0")
        return nb_frames, float(dur_str)
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0, 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()
        return count, (count / fps) if fps > 0 else 0.0


def validate_recording(capture_dir: Path, verify_opencv_markers: bool = True) -> dict[str, Any]:
    """Validate Level 1 recording folder and return integrity {"status": ..., "findings": [...]} (YS-23)."""
    findings: list[dict[str, str]] = []

    def add_finding(code: str, severity: str, message: str) -> None:
        findings.append({"code": code, "severity": severity, "message": message})

    if not capture_dir.exists() or not capture_dir.is_dir():
        add_finding("capture_dir_missing", "error", f"Capture directory does not exist: {capture_dir}")
        return {"status": "invalid", "findings": findings}

    session_path = capture_dir / "session.json"
    manifest_path = capture_dir / "manifest.json"

    if not session_path.exists():
        add_finding("missing_session_json", "error", "session.json is missing")
        return {"status": "invalid", "findings": findings}

    try:
        session_data = json.loads(session_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=session_data, schema=_load_schema("session.schema.json"))
    except Exception as exc:
        add_finding("session_schema_error", "error", f"session.json failed schema validation: {exc}")
        return {"status": "invalid", "findings": findings}

    # Check if recording finished cleanly or is partial (YS-08, YS-10)
    state_status = session_data["state"]["status"]
    stop_reason = session_data["state"]["stop_reason"]
    failed_stream = session_data["state"]["failed_stream"]
    is_partial = False

    if failed_stream is not None or state_status == "failed":
        is_partial = True
        add_finding(
            "stream_write_failed",
            "warning",
            f"Recording stopped due to failed stream: {failed_stream} (reason={stop_reason})",
        )
    elif state_status != "stopped" or stop_reason != "user_done":
        is_partial = True
        add_finding(
            "recording_incomplete",
            "warning",
            f"Recording did not finish normally: status={state_status}, stop_reason={stop_reason}",
        )

    # Check YS-00 LiDAR support
    if not session_data["device"].get("lidar_supported", False):
        add_finding("unsupported_device", "error", "Device does not support LiDAR (YS-00)")

    # Check YS-03 requested vs actual camera format
    req_fmt = session_data["camera_format"]["requested"]
    act_fmt = session_data["camera_format"]["actual"]
    if (
        req_fmt["width"] != act_fmt["width"]
        or req_fmt["height"] != act_fmt["height"]
        or req_fmt["fps"] != act_fmt["fps"]
    ):
        add_finding(
            "camera_format_mismatch",
            "error",
            f"Requested camera format {req_fmt} != actual {act_fmt} (YS-03)",
        )

    # Check YS-01 UTC clock mapping if stop_clock_pair exists
    clock_obj = session_data["clock"]
    if clock_obj.get("stop_clock_pair") is not None:
        try:
            start_t = float(clock_obj["start_clock_pair"]["t_s"])
            stop_t = float(clock_obj["stop_clock_pair"]["t_s"])
            start_utc = datetime.fromisoformat(clock_obj["start_clock_pair"]["utc"].replace("Z", "+00:00"))
            stop_utc = datetime.fromisoformat(clock_obj["stop_clock_pair"]["utc"].replace("Z", "+00:00"))
            utc_span = (stop_utc - start_utc).total_seconds()
            mono_span = stop_t - start_t
            if abs(utc_span - mono_span) > 1.0:
                add_finding(
                    "clock_utc_drift",
                    "error",
                    f"Monotonic span ({mono_span:.3f}s) vs UTC span ({utc_span:.3f}s) differs by > 1.0s (YS-01)",
                )
        except Exception as exc:
            add_finding("clock_parse_error", "error", f"Invalid UTC timestamp in session.json: {exc}")

    # Check YS-14 location / north status
    loc_obj = session_data["location"]
    if loc_obj["status"] in ("denied", "accuracy_exceeded", "unavailable"):
        add_finding(
            f"location_{loc_obj['status']}",
            "warning",
            f"Location/heading calibration status is {loc_obj['status']} (YS-14)",
        )

    # Check YS-41 tape check endpoints vs phone-written measured_m / residual_m
    for tc in session_data.get("tape_checks", []):
        rtc = recompute_tape_check(tc)
        if rtc.get("phone_mismatch", False):
            add_finding(
                "tape_check_residual_mismatch",
                "warning",
                f"session.json tape_check id={tc.get('id')} phone measured={rtc['phone_measured_m']:.4f}m/res={rtc['phone_residual_m']:.4f}m "
                f"!= recomputed measured={rtc['measured_m']:.4f}m/res={rtc['residual_m']:.4f}m (YS-41)",
            )

    # Check manifest.json (YS-20, YS-21): for a stopped capture, manifest.json must exist and match all files
    if not manifest_path.exists():
        if is_partial:
            return {"status": "partial", "findings": findings}
        add_finding("missing_manifest_json", "error", "manifest.json is missing on a stopped capture")
        return {"status": "invalid", "findings": findings}

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=manifest_data, schema=_load_schema("manifest.schema.json"))
    except Exception as exc:
        add_finding("manifest_schema_error", "error", f"manifest.json failed schema validation: {exc}")
        return {"status": "invalid", "findings": findings}

    required_streams = {
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
    }
    declared_streams = set(manifest_data.get("streams_declared", []))
    missing_streams = required_streams - declared_streams
    if missing_streams and not is_partial:
        add_finding("missing_declared_stream", "error", f"Missing required streams in manifest: {sorted(missing_streams)}")

    # Verify bidirectional file list & SHA-256 hashes (YS-20, YS-21, Q023: ignore standard/ and manifest.json)
    declared_files: dict[str, dict[str, Any]] = {}
    for entry in manifest_data["files"]:
        rel_p = entry["path"]
        if rel_p == "manifest.json" or rel_p.startswith("standard/"):
            add_finding("invalid_manifest_entry", "error", f"manifest.json must not list {rel_p}")
        declared_files[rel_p] = entry

    actual_files: set[str] = set()
    for p in sorted(capture_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(capture_dir).as_posix()
            if rel == "manifest.json" or rel.startswith("standard/") or rel.startswith("."):
                continue
            actual_files.add(rel)

    undeclared = sorted(actual_files - set(declared_files.keys()))
    if undeclared:
        add_finding("undeclared_file", "error", f"Files present on disk but not declared in manifest.json: {undeclared}")

    missing_on_disk = sorted(set(declared_files.keys()) - actual_files)
    if missing_on_disk:
        add_finding("missing_declared_file", "error", f"Files declared in manifest.json but missing on disk: {missing_on_disk}")

    for rel_p, meta in declared_files.items():
        fp = capture_dir / rel_p
        if not fp.exists():
            continue
        actual_size = fp.stat().st_size
        if actual_size != int(meta["size_bytes"]):
            add_finding(
                "file_size_mismatch",
                "error",
                f"File {rel_p} size {actual_size} != manifest size {meta['size_bytes']} (truncated or altered)",
            )
            continue
        actual_hash = sha256_file(fp)
        if actual_hash != meta["sha256"]:
            add_finding(
                "bad_hash",
                "error",
                f"SHA-256 mismatch on {rel_p}: expected {meta['sha256']}, got {actual_hash}",
            )

    # Validate frames.csv (YS-01, YS-02, YS-11)
    frames_path = capture_dir / "frames.csv"
    saved_indices: list[int] = []
    frame_times_saved: list[float] = []
    all_frame_times: list[float] = []
    dropped_count = 0
    limited_count = 0

    if not frames_path.exists():
        add_finding("missing_frames_csv", "error", "frames.csv is missing")
    else:
        with open(frames_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_cols = ["t", "saved_index", "dropped", "drop_reason", "tracking_state", "tracking_reason"]
            if reader.fieldnames != expected_cols:
                add_finding("frames_csv_header", "error", f"frames.csv header {reader.fieldnames} != {expected_cols}")
            else:
                prev_t = -1.0
                for row_num, row in enumerate(reader):
                    t_val = float(row["t"])
                    if t_val <= prev_t and row_num > 0:
                        add_finding("non_monotonic_timestamp", "error", f"frames.csv row {row_num} timestamp {t_val} <= {prev_t}")
                    prev_t = t_val
                    all_frame_times.append(t_val)

                    dropped_flag = row["dropped"].strip().lower() == "true"
                    s_idx_str = row["saved_index"].strip()
                    reason_str = row["drop_reason"].strip()
                    tr_state = row["tracking_state"].strip()
                    tr_reason = row["tracking_reason"].strip()

                    if not tr_state:
                        add_finding("missing_tracking_state", "error", f"frames.csv row {row_num} missing tracking_state")
                    if tr_state == "limited" and not tr_reason:
                        add_finding("missing_tracking_reason", "error", f"frames.csv row {row_num} limited tracking without reason")

                    if dropped_flag:
                        dropped_count += 1
                        if not reason_str:
                            add_finding("drop_without_reason", "error", f"frames.csv row {row_num} dropped=true without drop_reason (YS-02)")
                        if s_idx_str != "":
                            add_finding("dropped_has_saved_index", "error", f"frames.csv row {row_num} dropped=true but has saved_index={s_idx_str}")
                    else:
                        if s_idx_str == "":
                            add_finding("missing_saved_index", "error", f"frames.csv row {row_num} dropped=false but empty saved_index")
                        else:
                            saved_indices.append(int(s_idx_str))
                            frame_times_saved.append(t_val)
                        if tr_state != "normal":
                            limited_count += 1

    expected_saved = list(range(len(saved_indices)))
    if saved_indices != expected_saved:
        add_finding("saved_index_gap", "error", f"saved_index sequence in frames.csv has gaps or is not 0..n-1")

    n_saved = len(saved_indices)
    total_delivered = len(all_frame_times)
    if total_delivered > 0:
        drop_ratio = float(dropped_count) / float(total_delivered)
        if drop_ratio > 0.05:
            add_finding("unplanned_drops_exceeded", "warning", f"Unplanned frame drops {drop_ratio*100:.1f}% > 5% floor (YS-02)")
    if n_saved > 0:
        limited_ratio = float(limited_count) / float(n_saved)
        if limited_ratio > 0.10:
            add_finding("limited_tracking_exceeded", "warning", f"Limited tracking on {limited_ratio*100:.1f}% of saved frames (> 10%, YS-11)")

    # Validate poses.csv (YS-04, YS-05, YS-23)
    poses_path = capture_dir / "poses.csv"
    poses_by_idx: dict[int, dict[str, Any]] = {}
    if not poses_path.exists():
        add_finding("missing_poses_csv", "error", "poses.csv is missing")
    else:
        with open(poses_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_cols = ["t", "saved_index", "px", "py", "pz", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy", "width", "height"]
            if reader.fieldnames != expected_cols:
                add_finding("poses_csv_header", "error", f"poses.csv header {reader.fieldnames} != {expected_cols}")
            else:
                prev_t = -1.0
                for row_num, row in enumerate(reader):
                    t_val = float(row["t"])
                    if t_val <= prev_t and row_num > 0:
                        add_finding("poses_non_monotonic", "error", f"poses.csv non-monotonic t at row {row_num}")
                    prev_t = t_val
                    s_idx = int(row["saved_index"])
                    poses_by_idx[s_idx] = row
                    qx, qy, qz, qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
                    T = pose_to_matrix(
                        float(row["px"]),
                        float(row["py"]),
                        float(row["pz"]),
                        qx,
                        qy,
                        qz,
                        qw,
                        normalize=False,
                    )
                    if not is_valid_unit_quaternion(qx, qy, qz, qw, tol=1e-4) or not is_orthonormal_se3(T, tol=1e-4):
                        add_finding("non_orthonormal_pose", "error", f"Pose at saved_index={s_idx} is not orthonormal SE(3)")
                    fx, fy = float(row["fx"]), float(row["fy"])
                    w, h = int(row["width"]), int(row["height"])
                    if fx <= 0 or fy <= 0 or w != act_fmt["width"] or h != act_fmt["height"]:
                        add_finding("invalid_intrinsics", "error", f"Invalid intrinsics at saved_index={s_idx}")

    if sorted(poses_by_idx.keys()) != expected_saved:
        add_finding("poses_count_mismatch", "error", f"poses.csv saved_index set does not match 0..{n_saved-1}")

    # Validate depth/ and confidence/ PNGs (YS-04, YS-23: open EVERY image in the recording)
    depth_dir = capture_dir / "depth"
    conf_dir = capture_dir / "confidence"
    if not depth_dir.exists() or not conf_dir.exists():
        add_finding("missing_depth_or_conf_dir", "error", "depth/ or confidence/ directory missing")
    else:
        depth_files = sorted(depth_dir.glob("*.png"))
        conf_files = sorted(conf_dir.glob("*.png"))
        if len(depth_files) != n_saved or len(conf_files) != n_saved:
            add_finding(
                "depth_conf_count_mismatch",
                "error",
                f"Expected {n_saved} depth & confidence PNGs, found {len(depth_files)} depth and {len(conf_files)} confidence",
            )
        else:
            for idx in range(n_saved):
                dp = depth_dir / f"{idx:06d}.png"
                cp = conf_dir / f"{idx:06d}.png"
                try:
                    with Image.open(dp) as d_img:
                        d_arr = np.array(d_img)
                    with Image.open(cp) as c_img:
                        c_arr = np.array(c_img)
                except Exception as exc:
                    add_finding("corrupt_depth_or_conf_png", "error", f"Corrupt or truncated PNG at index {idx:06d}: {exc}")
                    continue
                if d_arr.dtype != np.uint16:
                    add_finding("depth_bit_depth_error", "error", f"{dp.name} is {d_arr.dtype}, expected uint16")
                if c_arr.dtype != np.uint8 or np.any(c_arr > 2):
                    add_finding("confidence_value_error", "error", f"{cp.name} has invalid dtype {c_arr.dtype} or values > 2")
                if d_arr.shape != c_arr.shape:
                    add_finding("depth_conf_size_mismatch", "error", f"Depth shape {d_arr.shape} != confidence shape {c_arr.shape}")

    # Validate rgb.mp4 frame count and duration (YS-02, YS-04)
    video_path = capture_dir / "rgb.mp4"
    if not video_path.exists():
        add_finding("missing_rgb_mp4", "error", "rgb.mp4 is missing")
    elif n_saved > 0 and not any(f["code"] == "file_size_mismatch" for f in findings):
        v_frames, v_dur = probe_video_frame_count_and_duration(video_path)
        if v_frames != n_saved:
            add_finding("video_frame_count_mismatch", "error", f"rgb.mp4 decoded frame count {v_frames} != {n_saved} saved frames (YS-02)")
        fps = float(act_fmt["fps"])
        expected_span = (frame_times_saved[-1] - frame_times_saved[0]) + (1.0 / fps)
        if abs(v_dur - expected_span) > (1.5 / fps) and abs(v_dur - (frame_times_saved[-1] - frame_times_saved[0])) > (1.5 / fps):
            add_finding(
                "video_duration_mismatch",
                "error",
                f"rgb.mp4 duration {v_dur:.3f}s does not match capture span {expected_span:.3f}s within 1 frame period (YS-02)",
            )

    # Validate audio.wav (YS-01, YS-06)
    audio_path = capture_dir / "audio.wav"
    if not audio_path.exists():
        add_finding("missing_audio_wav", "error", "audio.wav is missing")
    elif n_saved > 0:
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                n_samples = wf.getnframes()
                audio_dur = float(n_samples) / float(sr) if sr > 0 else 0.0
                if sr != 16000 or ch != 1:
                    add_finding("audio_format_error", "error", f"audio.wav is {sr}Hz {ch}ch, expected 16000Hz mono")
                frame_span = all_frame_times[-1] - all_frame_times[0] if all_frame_times else 0.0
                if abs(audio_dur - frame_span) > 0.5:
                    add_finding(
                        "audio_span_mismatch",
                        "error",
                        f"audio.wav duration {audio_dur:.3f}s differs from frame span {frame_span:.3f}s by > 0.5s (YS-01)",
                    )
        except Exception as exc:
            add_finding("audio_read_error", "error", f"Failed to read audio.wav: {exc}")

    # Validate events.csv (YS-07)
    events_path = capture_dir / "events.csv"
    if not events_path.exists():
        add_finding("missing_events_csv", "error", "events.csv is missing")
    else:
        with open(events_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["t", "saved_index", "kind", "detail"]:
                add_finding("events_csv_header", "error", f"Invalid events.csv header: {reader.fieldnames}")
            else:
                prev_t = -1.0
                for r in reader:
                    t_val = float(r["t"])
                    if t_val < prev_t:
                        add_finding("events_non_monotonic", "error", "events.csv timestamps not monotonic")
                    prev_t = t_val
                    if int(r["saved_index"]) not in poses_by_idx:
                        add_finding("event_invalid_saved_index", "error", f"Event references unknown saved_index {r['saved_index']}")

    # Validate optional guidance.csv stream (YS-67)
    guidance_path = capture_dir / "guidance.csv"
    if "guidance" in declared_streams and not guidance_path.exists():
        add_finding("missing_guidance_csv", "error", "guidance stream declared in manifest.json but guidance.csv is missing (YS-67)")
    if guidance_path.exists():
        with open(guidance_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or reader.fieldnames[0] != "t":
                add_finding("guidance_csv_header", "error", f"Invalid guidance.csv header: {reader.fieldnames} (YS-67)")
            else:
                prev_t = -1.0
                for r in reader:
                    try:
                        t_val = float(r["t"])
                    except Exception:
                        add_finding("guidance_invalid_timestamp", "error", f"Invalid guidance.csv timestamp: {r.get('t')} (YS-67)")
                        continue
                    if t_val < prev_t:
                        add_finding("guidance_non_monotonic", "error", "guidance.csv timestamps not monotonic (YS-67)")
                    prev_t = t_val
                    s_idx_raw = (r.get("saved_index") or "").strip()
                    if s_idx_raw != "" and int(s_idx_raw) not in poses_by_idx:
                        add_finding("guidance_invalid_saved_index", "error", f"Guidance row references unknown saved_index {s_idx_raw} (YS-67)")

    # Validate stills.csv & stills/*.jpg (YS-13, YS-67)
    stills_path = capture_dir / "stills.csv"
    stills_by_idx: dict[int, str] = {}
    if not stills_path.exists():
        add_finding("missing_stills_csv", "error", "stills.csv is missing")
    else:
        with open(stills_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_cols = ["t", "saved_index", "file", "px", "py", "pz", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy", "width", "height"]
            expected_cols_with_exposure = expected_cols + ["exposure_duration_s", "iso"]
            if reader.fieldnames not in (expected_cols, expected_cols_with_exposure):
                add_finding("stills_csv_header", "error", f"stills.csv header {reader.fieldnames} != {expected_cols} (or + ['exposure_duration_s', 'iso'])")
            else:
                prev_t = -1.0
                for r in reader:
                    t_val = float(r["t"])
                    if t_val <= prev_t and prev_t >= 0.0:
                        add_finding("stills_non_monotonic", "error", "stills.csv timestamps not strictly increasing")
                    prev_t = t_val
                    s_idx = int(r["saved_index"])
                    rel_f = r["file"]
                    stills_by_idx[s_idx] = rel_f
                    for opt_col in ("exposure_duration_s", "iso"):
                        raw_opt = (r.get(opt_col) or "").strip()
                        if raw_opt != "":
                            try:
                                val_opt = float(raw_opt)
                                if not np.isfinite(val_opt) or val_opt < 0.0:
                                    add_finding("still_invalid_exposure_or_iso", "error", f"stills.csv {opt_col}={raw_opt} at saved_index={s_idx} must be >= 0 (YS-67)")
                            except ValueError:
                                add_finding("still_invalid_exposure_or_iso", "error", f"stills.csv {opt_col}={raw_opt} at saved_index={s_idx} is not a number (YS-67)")
                    sqx, sqy, sqz, sqw = float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])
                    T_s = pose_to_matrix(
                        float(r["px"]),
                        float(r["py"]),
                        float(r["pz"]),
                        sqx,
                        sqy,
                        sqz,
                        sqw,
                        normalize=False,
                    )
                    if not is_valid_unit_quaternion(sqx, sqy, sqz, sqw, tol=1e-4) or not is_orthonormal_se3(T_s, tol=1e-4):
                        add_finding("still_pose_not_orthonormal", "error", f"Still pose at saved_index={s_idx} not orthonormal SE(3)")
                    if s_idx not in poses_by_idx:
                        add_finding("still_invalid_saved_index", "error", f"Still references unknown saved_index {s_idx}")
                    still_fp = capture_dir / rel_f
                    if not still_fp.exists():
                        add_finding("missing_still_file", "error", f"Still file {rel_f} missing")
                    else:
                        try:
                            with Image.open(still_fp) as s_img:
                                s_img.load()
                        except Exception as exc:
                            add_finding("corrupt_still_file", "error", f"Corrupt still image {rel_f}: {exc}")

    # Validate markers.csv & OpenCV agreement on still-backed sightings (YS-30, YS-115)
    markers_path = capture_dir / "markers.csv"
    if not markers_path.exists():
        add_finding("missing_markers_csv", "error", "markers.csv is missing")
    else:
        with open(markers_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_cols = list(MARKERS_CSV_BASE_COLUMNS)
            expected_cols_with_cam = expected_cols + list(MARKERS_CSV_CAMERA_COLUMNS)
            if reader.fieldnames not in (expected_cols, expected_cols_with_cam):
                add_finding(
                    "markers_csv_header",
                    "error",
                    f"markers.csv header {reader.fieldnames} != {expected_cols} (or + camera columns)",
                )
            else:
                has_cam_cols = reader.fieldnames == expected_cols_with_cam
                prev_t = -1.0
                for r in reader:
                    t_val = float(r["t"])
                    if t_val < prev_t:
                        add_finding("markers_non_monotonic", "error", "markers.csv timestamps not monotonic")
                    prev_t = t_val
                    s_idx = int(r["saved_index"])
                    mid = int(r["marker_id"])
                    if not (0 <= mid <= 7):
                        add_finding("marker_id_out_of_range", "error", f"Marker ID {mid} at {s_idx} not in 0..7")
                    mqx, mqy, mqz, mqw = float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])
                    T_m = pose_to_matrix(
                        float(r["x"]),
                        float(r["y"]),
                        float(r["z"]),
                        mqx,
                        mqy,
                        mqz,
                        mqw,
                        normalize=False,
                    )
                    if not is_valid_unit_quaternion(mqx, mqy, mqz, mqw, tol=1e-4) or not is_orthonormal_se3(T_m, tol=1e-4):
                        add_finding("marker_pose_not_orthonormal", "error", f"Marker {mid} pose at {s_idx} not orthonormal")
                    if has_cam_cols and r.get("cam_qx", "") != "":
                        cqx, cqy, cqz, cqw = (
                            float(r["cam_qx"]),
                            float(r["cam_qy"]),
                            float(r["cam_qz"]),
                            float(r["cam_qw"]),
                        )
                        T_cm = pose_to_matrix(
                            float(r["cam_x"]),
                            float(r["cam_y"]),
                            float(r["cam_z"]),
                            cqx,
                            cqy,
                            cqz,
                            cqw,
                            normalize=False,
                        )
                        if not is_valid_unit_quaternion(cqx, cqy, cqz, cqw, tol=1e-4) or not is_orthonormal_se3(T_cm, tol=1e-4):
                            add_finding(
                                "marker_cam_pose_not_orthonormal",
                                "error",
                                f"Marker {mid} camera-space pose at {s_idx} not orthonormal",
                            )
                    if verify_opencv_markers and s_idx in stills_by_idx:
                        still_fp = capture_dir / stills_by_idx[s_idx]
                        if still_fp.exists():
                            img_gray = np.array(Image.open(still_fp).convert("L"))
                            det = detect_aruco_markers_opencv(img_gray)
                            if mid not in det:
                                add_finding(
                                    "marker_opencv_disagreement",
                                    "error",
                                    f"Phone logged marker_id={mid} at saved_index={s_idx}, but laptop OpenCV did not detect it in {stills_by_idx[s_idx]} (YS-30)",
                                )

    # Validate optional pairs.csv stream (YS-115)
    pairs_path = capture_dir / "pairs.csv"
    if "pairs" in declared_streams and not pairs_path.exists():
        add_finding("missing_pairs_csv", "error", "pairs stream declared in manifest.json but pairs.csv is missing (YS-115)")
    if pairs_path.exists():
        with open(pairs_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(PAIR_CSV_COLUMNS):
                add_finding("pairs_csv_header", "error", f"pairs.csv header {reader.fieldnames} != {PAIR_CSV_COLUMNS}")
            else:
                prev_t = -1.0
                for row_num, r in enumerate(reader):
                    try:
                        t_val = float(r["t"])
                        s_idx = int(r["saved_index"])
                        mi = int(r["marker_i"])
                        mj = int(r["marker_j"])
                        d3 = float(r["dist_3d_m"])
                        dh = float(r["dist_horiz_m"])
                        sd = float(r["sigma_dist_m"])
                        sy = float(r["sigma_yaw_rad"])
                    except Exception as exc:
                        add_finding("pairs_csv_parse_error", "error", f"Invalid numeric value in pairs.csv row {row_num}: {exc}")
                        continue
                    if t_val < prev_t:
                        add_finding("pairs_non_monotonic", "error", f"pairs.csv timestamps not monotonic at row {row_num}")
                    prev_t = t_val
                    if s_idx not in poses_by_idx:
                        add_finding("pair_invalid_saved_index", "error", f"pairs.csv row {row_num} references unknown saved_index {s_idx}")
                    if not (0 <= mi < mj <= 7):
                        add_finding("pair_invalid_marker_ids", "error", f"pairs.csv row {row_num} has invalid pair ({mi}, {mj}); expected 0 <= marker_i < marker_j <= 7")
                    if d3 < 0.0 or dh < 0.0 or sd <= 0.0 or sy <= 0.0:
                        add_finding("pair_invalid_distance_or_sigma", "error", f"pairs.csv row {row_num} has negative distance or non-positive sigma")

    has_errors = any(f["severity"] == "error" for f in findings)
    has_warnings = any(f["severity"] == "warning" for f in findings)

    if has_errors:
        status = "invalid"
    elif is_partial:
        status = "partial"
    elif has_warnings:
        status = "warnings"
    else:
        status = "clean"

    return {"status": status, "findings": findings}


def validate_standard_capture(capture_dir: Path) -> dict[str, Any]:
    """Validate Level 2 <capture-id>/standard/ folder against schemas and contract invariants (YS-20, 21, 23, 50..53)."""
    findings: list[dict[str, str]] = []

    def add_finding(code: str, severity: str, message: str) -> None:
        findings.append({"code": code, "severity": severity, "message": message})

    std_dir = capture_dir / "standard" if (capture_dir / "standard").exists() else capture_dir
    rec_dir = std_dir.parent if std_dir.name == "standard" else None

    schema_map = {
        "capture.json": "capture.schema.json",
        "frame.json": "frame.schema.json",
        "keyframes.json": "keyframes.schema.json",
        "trajectory.json": "trajectory.schema.json",
        "coverage.json": "coverage.schema.json",
        "events.json": "events.schema.json",
        "grade.json": "grade.schema.json",
    }

    loaded_json: dict[str, Any] = {}
    for fname, sname in schema_map.items():
        fp = std_dir / fname
        if not fp.exists():
            add_finding("missing_standard_file", "error", f"standard/{fname} is missing")
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            loaded_json[fname] = data
            # Check YS-53 forbidden interpretation keys before schema check
            forbidden = _check_no_forbidden_keys(data, fname)
            if forbidden:
                add_finding("forbidden_interpretation_field", "error", f"Forbidden interpretation fields in {fname}: {forbidden} (YS-53)")
            jsonschema.validate(instance=data, schema=_load_schema(sname))
        except Exception as exc:
            add_finding("standard_schema_error", "error", f"standard/{fname} failed schema validation: {exc}")

    # Optional standard/visit.json (present when capture was guided by a lotfit.visit-plan/1)
    visit_fp = std_dir / "visit.json"
    if visit_fp.exists():
        try:
            visit_data = json.loads(visit_fp.read_text(encoding="utf-8"))
            loaded_json["visit.json"] = visit_data
            forbidden_visit = _check_no_forbidden_keys(visit_data, "visit.json")
            if forbidden_visit:
                add_finding(
                    "forbidden_interpretation_field",
                    "error",
                    f"Forbidden interpretation fields in visit.json: {forbidden_visit} (YS-53)",
                )
            jsonschema.validate(instance=visit_data, schema=_load_schema("visit.schema.json"))
            if rec_dir is not None and (rec_dir / "visit_plan.json").exists():
                expected_plan_sha = sha256_file(rec_dir / "visit_plan.json")
                if visit_data.get("plan_sha256") != expected_plan_sha:
                    add_finding(
                        "visit_plan_hash_mismatch",
                        "error",
                        f"visit.json plan_sha256 {visit_data.get('plan_sha256')} != visit_plan.json sha256 {expected_plan_sha}",
                    )
        except Exception as exc:
            add_finding("standard_schema_error", "error", f"standard/visit.json failed schema validation: {exc}")

    # Check YS-53 forbidden files in standard/ (allowing YS-115 raw unwarped trajectory & cloud)
    allowed_top_files = set(schema_map.keys()) | {
        "visit.json",
        "cloud.ply",
        "cloud.raw.ply",
        "trajectory.raw.json",
        "audio.wav",
    }
    actual_std_files: set[str] = set()
    for p in sorted(std_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(std_dir).as_posix()
            if rel.startswith("."):
                continue
            actual_std_files.add(rel)
            if rel != "capture.json" and not (rel in allowed_top_files or rel.startswith("keyframes/")):
                add_finding("undeclared_standard_file", "error", f"Unexpected or interpretation file in standard/: {rel} (YS-53)")

    # Verify standard/capture.json file hashes and recording_manifest_sha256 (YS-21, Q023)
    if "capture.json" in loaded_json:
        cap_data = loaded_json["capture.json"]
        if rec_dir is not None and (rec_dir / "manifest.json").exists():
            expected_manifest_hash = sha256_file(rec_dir / "manifest.json")
            if cap_data["recording_manifest_sha256"] != expected_manifest_hash:
                add_finding(
                    "recording_manifest_hash_mismatch",
                    "error",
                    "capture.json recording_manifest_sha256 does not match recording manifest.json",
                )
        declared_std_files = {f["path"]: f for f in cap_data.get("files", [])}
        expected_siblings = actual_std_files - {"capture.json"}
        if set(declared_std_files.keys()) != expected_siblings:
            add_finding(
                "standard_manifest_mismatch",
                "error",
                f"capture.json files {sorted(declared_std_files.keys())} != actual {sorted(expected_siblings)}",
            )
        for rel_p, meta in declared_std_files.items():
            fp = std_dir / rel_p
            if fp.exists():
                if fp.stat().st_size != int(meta["size_bytes"]) or sha256_file(fp) != meta["sha256"]:
                    add_finding("standard_bad_hash", "error", f"Hash or size mismatch on standard/{rel_p}")

    # Verify frame.json SE(3) transform (YS-23, YS-31)
    if "frame.json" in loaded_json:
        T_yw = np.array(loaded_json["frame.json"]["world_to_yard_transform"], dtype=np.float64).reshape(4, 4)
        if not is_orthonormal_se3(T_yw, tol=1e-4):
            add_finding("frame_transform_not_se3", "error", "frame.json world_to_yard_transform is not orthonormal SE(3)")

    # Verify trajectory.json SE(3) poses (and optional trajectory.raw.json)
    for traj_fname in ("trajectory.json", "trajectory.raw.json"):
        traj_fp = std_dir / traj_fname
        if traj_fname == "trajectory.raw.json" and traj_fp.exists():
            try:
                raw_t_data = json.loads(traj_fp.read_text(encoding="utf-8"))
                forbidden_raw_t = _check_no_forbidden_keys(raw_t_data, traj_fname)
                if forbidden_raw_t:
                    add_finding(
                        "forbidden_interpretation_field",
                        "error",
                        f"Forbidden interpretation fields in {traj_fname}: {forbidden_raw_t} (YS-53)",
                    )
                jsonschema.validate(instance=raw_t_data, schema=_load_schema("trajectory.schema.json"))
                loaded_json[traj_fname] = raw_t_data
            except Exception as exc:
                add_finding("standard_schema_error", "error", f"standard/{traj_fname} failed schema validation: {exc}")
        if traj_fname in loaded_json:
            prev_t = -1.0
            for p_item in loaded_json[traj_fname]["poses"]:
                t_val = float(p_item["t"])
                if t_val <= prev_t and prev_t >= 0.0:
                    add_finding("trajectory_non_monotonic", "error", f"{traj_fname} timestamps not strictly increasing")
                prev_t = t_val
                T_p = np.array(p_item["pose"], dtype=np.float64).reshape(4, 4)
                if not is_orthonormal_se3(T_p, tol=1e-4):
                    add_finding("trajectory_pose_not_se3", "error", f"{traj_fname} pose at saved_index={p_item['saved_index']} not orthonormal SE(3)")

    # Verify keyframes.json poses and existence in recording
    if "keyframes.json" in loaded_json:
        for kf in loaded_json["keyframes.json"]["keyframes"]:
            T_k = np.array(kf["pose"], dtype=np.float64).reshape(4, 4)
            if not is_orthonormal_se3(T_k, tol=1e-4):
                add_finding("keyframe_pose_not_se3", "error", f"Keyframe {kf['id']} pose is not orthonormal SE(3)")
            if not (std_dir / kf["file"]).exists():
                add_finding("missing_keyframe_image", "error", f"Keyframe file {kf['file']} missing in standard/")

    # Verify coverage.json is a simple CCW polygon with area > 0 (YS-52)
    if "coverage.json" in loaded_json:
        cov = loaded_json["coverage.json"]
        poly = cov["polygon_xy_m"]
        if not is_simple_polygon(poly):
            add_finding("coverage_polygon_not_simple", "error", "coverage.json polygon_xy_m is not a simple polygon (YS-52)")
        for h_poly in cov.get("holes_xy_m", []):
            if not is_simple_polygon(h_poly):
                add_finding("coverage_hole_not_simple", "error", "coverage.json hole polygon is not simple")

    # Verify grade.json floor & recomputed tape checks (YS-41, YS-42)
    if "grade.json" in loaded_json:
        gr = loaded_json["grade.json"]
        dist_tol = float(gr["tolerances_95pct"]["distance_m"])
        primary_residuals: list[float] = []
        failed_held_out_count = 0
        for tc in gr.get("tape_checks", []):
            pa_y = np.array(tc["point_a_yard_m"], dtype=np.float64)
            pb_y = np.array(tc["point_b_yard_m"], dtype=np.float64)
            recomp_meas = float(np.linalg.norm(pb_y - pa_y))
            recomp_res = abs(recomp_meas - float(tc["tape_m"]))
            if abs(float(tc["measured_m"]) - recomp_meas) > 1e-3 or abs(float(tc["residual_m"]) - recomp_res) > 1e-3:
                add_finding(
                    "grade_tape_residual_not_recomputed",
                    "error",
                    f"grade.json tape_check id={tc['id']} measured_m/residual_m does not match 3D endpoints (YS-41)",
                )
            if tc.get("role", "primary") == "primary":
                primary_residuals.append(recomp_res)
            elif tc.get("role") == "held_out":
                if recomp_res > float(tc["tolerance_at_tape_m"]) + 1e-6:
                    failed_held_out_count += 1
        if primary_residuals and dist_tol + 1e-9 < max(primary_residuals):
            add_finding(
                "tolerance_below_tape_residual",
                "error",
                f"grade.json distance_m ({dist_tol}) < largest primary tape residual ({max(primary_residuals)}) (YS-41 floor)",
            )
        if failed_held_out_count >= 2 and gr["grade"] != "fail":
            add_finding(
                "held_out_tape_cap_violated",
                "error",
                f"grade.json has {failed_held_out_count} failing held-out tape checks but grade is {gr['grade']} != fail (YS-42)",
            )
        elif failed_held_out_count == 1 and gr["grade"] in ("A", "B"):
            add_finding(
                "held_out_tape_cap_violated",
                "error",
                f"grade.json has 1 failing held-out tape check but grade is {gr['grade']} (must be capped at C, YS-42)",
            )

    # Verify cloud.ply (and optional cloud.raw.ply) binary little-endian format (YS-50, YS-115)
    for ply_name in ("cloud.ply", "cloud.raw.ply"):
        ply_path = std_dir / ply_name
        if not ply_path.exists():
            if ply_name == "cloud.ply":
                add_finding("missing_cloud_ply", "error", "standard/cloud.ply is missing")
        else:
            try:
                verts = read_cloud_ply(ply_path)
                if len(verts) == 0:
                    add_finding("empty_cloud_ply", "error", f"standard/{ply_name} has 0 vertices")
            except Exception as exc:
                add_finding("invalid_cloud_ply", "error", f"standard/{ply_name} failed validation: {exc}")

    return {
        "status": "invalid" if any(f["severity"] == "error" for f in findings) else "clean",
        "findings": findings,
    }
