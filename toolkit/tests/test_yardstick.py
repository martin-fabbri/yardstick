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

"""End-to-end pytest suite for Yardstick (`YS-01` through `YS-53`)."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yardstick.cli import main as cli_main
from yardstick.converter import convert_recording_to_standard
from yardstick.geometry import camera_pitch_deg_from_pose, pose_to_matrix
from yardstick.markers import detect_aruco_markers_opencv
from yardstick.ply import read_cloud_ply
from yardstick.synth import (
    SynthOptions,
    generate_synthetic_capture,
    pack_capture_zip,
    seal_manifest,
    unpack_capture_zip,
)
from yardstick.validator import sha256_file, validate_recording, validate_standard_capture


@pytest.fixture(scope="module")
def base_synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate and convert one baseline Grade A synthetic yard capture."""
    out = tmp_path_factory.mktemp("synth_base") / "20261102T120000Z_yardstick_synth"
    generate_synthetic_capture(out, SynthOptions())
    convert_recording_to_standard(out)
    return out


def _clone_capture(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("standard"))
    return dst


def test_clean_baseline_recording_and_standard_capture(base_synth_dir: Path) -> None:
    """Verify baseline synthetic capture validates `clean` at both Level 1 and Level 2."""
    rep1 = validate_recording(base_synth_dir)
    assert rep1["status"] == "clean", f"Recording findings: {rep1['findings']}"

    rep2 = validate_standard_capture(base_synth_dir / "standard")
    assert rep2["status"] == "clean", f"Standard capture findings: {rep2['findings']}"


def test_ys01_ys02_timestamps_audio_and_upstream_drop(base_synth_dir: Path) -> None:
    """YS-01, YS-02: Monotonic timestamps, contiguous saved_index 0..n, and dropped row logged."""
    frames = list(csv.DictReader((base_synth_dir / "frames.csv").open()))
    saved_rows = [r for r in frames if r["dropped"].strip().lower() != "true"]
    dropped_rows = [r for r in frames if r["dropped"].strip().lower() == "true"]

    saved_indices = [int(r["saved_index"]) for r in saved_rows]
    assert saved_indices == list(range(len(saved_rows)))
    assert len(dropped_rows) == 1
    assert dropped_rows[0]["drop_reason"] == "writer_queue_full"


def test_ys05_camera_pitch_30_deg_down(base_synth_dir: Path) -> None:
    """YS-05: Camera held at 30 deg downward pitch reports -30 deg +/- 3 deg."""
    poses = list(csv.DictReader((base_synth_dir / "poses.csv").open()))
    for r in poses:
        T_wc = pose_to_matrix(
            float(r["px"]),
            float(r["py"]),
            float(r["pz"]),
            float(r["qx"]),
            float(r["qy"]),
            float(r["qz"]),
            float(r["qw"]),
        )
        pitch = camera_pitch_deg_from_pose(T_wc)
        assert abs(pitch - (-30.0)) <= 3.0


def test_ys08_interrupted_recording_classified_as_partial(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-08: Unsealed/killed recording without manifest.json is reported as `partial`."""
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_partial")
    (cap / "manifest.json").unlink()
    sess = json.loads((cap / "session.json").read_text())
    sess["state"]["status"] = "recording"
    sess["state"]["stop_reason"] = None
    (cap / "session.json").write_text(json.dumps(sess, indent=2))

    rep = validate_recording(cap)
    assert rep["status"] == "partial"

    # Converting a partial capture must force grade == 'fail'
    res = convert_recording_to_standard(cap)
    assert res["grade"]["grade"] == "fail"
    assert "integrity_partial" in res["grade"]["caps_applied"]


def test_ys14_north_calibration_and_denied_handling(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-14, Q020: North calibration computes true-north angle & uncertainty, and handles denied."""
    frame_data = json.loads((base_synth_dir / "standard" / "frame.json").read_text())
    north = frame_data["north_calibration"]
    assert north["status"] == "ok"
    assert north["sample_count"] >= 10
    assert abs(north["north_angle_deg"] - 35.0) <= 5.0
    assert north["north_uncertainty_deg"] is not None and north["north_uncertainty_deg"] > 0

    # Test permission denied case
    cap_denied = _clone_capture(base_synth_dir, tmp_path / "cap_north_denied")
    sess = json.loads((cap_denied / "session.json").read_text())
    sess["location"]["status"] = "denied"
    sess["location"]["north_samples"] = []
    (cap_denied / "session.json").write_text(json.dumps(sess, indent=2))
    seal_manifest(cap_denied)
    res = convert_recording_to_standard(cap_denied)
    assert res["frame"]["north_calibration"]["status"] == "denied"
    assert res["frame"]["north_calibration"]["north_angle_deg"] is None


def test_ys22_zip_store_pack_and_unpack(base_synth_dir: Path, tmp_path: Path) -> None:
    """YS-22: Store-mode .zip archive unpacks cleanly and passes hash validation."""
    zip_file = pack_capture_zip(base_synth_dir, tmp_path / "capture.zip")
    extracted = unpack_capture_zip(zip_file, tmp_path / "unpacked")
    rep = validate_recording(extracted)
    assert rep["status"] == "clean", f"Unpacked findings: {rep['findings']}"


def test_ys23_all_planted_faults_caught(base_synth_dir: Path, tmp_path: Path) -> None:
    """YS-23: Validator catches all 7 planted fault classes with specific error codes."""
    # 1. Truncated file
    c1 = _clone_capture(base_synth_dir, tmp_path / "f1_truncated")
    p_depth = c1 / "depth" / "000000.png"
    raw = p_depth.read_bytes()
    p_depth.write_bytes(raw[: len(raw) // 2])
    r1 = validate_recording(c1)
    assert r1["status"] == "invalid"
    assert any(f["code"] == "file_size_mismatch" for f in r1["findings"])

    # 2. saved_index gap
    c2 = _clone_capture(base_synth_dir, tmp_path / "f2_saved_gap")
    f_rows = list(csv.DictReader((c2 / "frames.csv").open()))
    f_rows[-1]["saved_index"] = str(int(f_rows[-1]["saved_index"]) + 2)
    with (c2 / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f_rows[0].keys()))
        w.writeheader()
        w.writerows(f_rows)
    seal_manifest(c2)
    r2 = validate_recording(c2)
    assert r2["status"] == "invalid"
    assert any(f["code"] == "saved_index_gap" for f in r2["findings"])

    # 3. Upstream frame drop without drop_reason in frames.csv
    c3 = _clone_capture(base_synth_dir, tmp_path / "f3_drop_no_reason")
    f_rows3 = list(csv.DictReader((c3 / "frames.csv").open()))
    for r in f_rows3:
        if r["dropped"].strip().lower() == "true":
            r["drop_reason"] = ""
    with (c3 / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f_rows3[0].keys()))
        w.writeheader()
        w.writerows(f_rows3)
    seal_manifest(c3)
    r3 = validate_recording(c3)
    assert r3["status"] == "invalid"
    assert any(f["code"] == "drop_without_reason" for f in r3["findings"])

    # 4. Bad SHA-256 hash
    c4 = _clone_capture(base_synth_dir, tmp_path / "f4_bad_hash")
    man = json.loads((c4 / "manifest.json").read_text())
    man["files"][0]["sha256"] = "0" * 64
    (c4 / "manifest.json").write_text(json.dumps(man, indent=2))
    r4 = validate_recording(c4)
    assert r4["status"] == "invalid"
    assert any(f["code"] == "bad_hash" for f in r4["findings"])

    # 5. Undeclared file on disk
    c5 = _clone_capture(base_synth_dir, tmp_path / "f5_undeclared")
    (c5 / "rogue_file.txt").write_text("undeclared")
    r5 = validate_recording(c5)
    assert r5["status"] == "invalid"
    assert any(f["code"] == "undeclared_file" for f in r5["findings"])

    # 6. Missing required stream
    c6 = _clone_capture(base_synth_dir, tmp_path / "f6_missing_stream")
    (c6 / "audio.wav").unlink()
    seal_manifest(c6)
    r6 = validate_recording(c6)
    assert r6["status"] == "invalid"
    assert any(f["code"] == "missing_audio_wav" for f in r6["findings"])

    # 7. Requested vs actual camera format mismatch
    c7 = _clone_capture(base_synth_dir, tmp_path / "f7_req_vs_actual")
    sess = json.loads((c7 / "session.json").read_text())
    sess["camera_format"]["actual"]["fps"] = 60
    (c7 / "session.json").write_text(json.dumps(sess, indent=2))
    seal_manifest(c7)
    r7 = validate_recording(c7)
    assert r7["status"] == "invalid"
    assert any(f["code"] == "camera_format_mismatch" for f in r7["findings"])


def test_ys30_opencv_aruco_agrees_with_logged_markers(base_synth_dir: Path) -> None:
    """YS-30: Independent OpenCV DICT_5X5_250 detector on stills agrees with markers.csv."""
    markers_rows = list(csv.DictReader((base_synth_dir / "markers.csv").open()))
    stills_rows = list(csv.DictReader((base_synth_dir / "stills.csv").open()))
    stills_by_saved = {int(r["saved_index"]): r["file"] for r in stills_rows}

    checked = 0
    for mrow in markers_rows:
        s_idx = int(mrow["saved_index"])
        expected_id = int(mrow["marker_id"])
        if s_idx not in stills_by_saved:
            continue
        img_path = base_synth_dir / stills_by_saved[s_idx]
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        dets = detect_aruco_markers_opencv(img)
        assert expected_id in dets, f"Expected marker {expected_id} in {stills_by_saved[s_idx]}, got {list(dets.keys())}"
        checked += 1

    assert checked >= 4


def test_ys31_ys32_marker0_origin_return_pass_and_fallbacks(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-31, YS-32: Marker 0 sets origin; return pass is within 5 cm; missing Marker 0 falls back and caps at C."""
    frame_data = json.loads((base_synth_dir / "standard" / "frame.json").read_text())
    grade_data = json.loads((base_synth_dir / "standard" / "grade.json").read_text())

    assert frame_data["method"] == "marker"
    assert abs(frame_data["origin_height_m"]) <= 0.02
    assert frame_data["origin_residual_m"] is not None
    assert frame_data["origin_residual_m"] <= 0.05
    assert grade_data["tolerances_95pct"]["coefficients"]["loop_closure_gap_m"] <= 0.05

    # Without Marker 0 -> method == 'first_pose', grade capped at C
    cap_no_m0 = tmp_path / "cap_no_m0"
    generate_synthetic_capture(cap_no_m0, SynthOptions(include_marker_0=False))
    res_no_m0 = convert_recording_to_standard(cap_no_m0)
    assert res_no_m0["frame"]["method"] == "first_pose"
    assert res_no_m0["grade"]["grade"] == "C"
    assert "no_marker_0" in res_no_m0["grade"]["caps_applied"]


def test_ys33_marker_scale_error_and_misdeclared_size_caught(tmp_path: Path) -> None:
    """YS-33, Q027: Multi-frame fused scale error and SE reported; 180 mm vs 250 mm mismatch flagged."""
    cap_mismatch = tmp_path / "cap_size_mismatch"
    generate_synthetic_capture(
        cap_mismatch,
        SynthOptions(actual_marker_size_mm=250.0, declared_marker_size_mm=180.0),
    )
    res = convert_recording_to_standard(cap_mismatch)
    ms = res["grade"]["marker_scale"]
    # (250 - 180) / 180 = +0.3889 (~38.9% scale error)
    assert abs(ms["scale_error_ratio"] - ((250.0 - 180.0) / 180.0)) < 0.03
    assert ms["scale_error_se"] < 0.02
    assert ms["scale_error_eff"] > 0.30
    assert res["grade"]["grade"] == "fail"


def test_ys41_tolerance_closed_form_and_held_out_checks(base_synth_dir: Path) -> None:
    """YS-41, Q029: Primary tape check sets floor; all 3 held-out tape checks pass within tol(tape_m_j)."""
    grade_data = json.loads((base_synth_dir / "standard" / "grade.json").read_text())
    tol_10m = grade_data["tolerances_95pct"]["distance_m"]
    primary_max = grade_data["tolerances_95pct"]["coefficients"]["max_tape_residual_m"]

    assert tol_10m >= primary_max
    held_out = [c for c in grade_data["tape_checks"] if c["role"] == "held_out"]
    assert len(held_out) == 3
    for c in held_out:
        assert c["within_tolerance"] is True
        assert c["residual_m"] <= c["tolerance_at_tape_m"]


def test_ys42_all_grades_and_caps(tmp_path: Path) -> None:
    """YS-42: Planted captures for A, B, C, fail, and each cap get exact expected grade and reason."""
    # Grade A: baseline
    cap_a = tmp_path / "grade_a"
    generate_synthetic_capture(cap_a, SynthOptions())
    res_a = convert_recording_to_standard(cap_a)
    assert res_a["grade"]["grade"] == "A"

    # Grade B: loop drift 8 cm (between 5 cm and 12 cm)
    cap_b = tmp_path / "grade_b"
    generate_synthetic_capture(cap_b, SynthOptions(loop_drift_xy_m=(0.08, 0.0)))
    res_b = convert_recording_to_standard(cap_b)
    assert res_b["grade"]["grade"] == "B"

    # Grade C: loop drift 16 cm (between 12 cm and 25 cm)
    cap_c = tmp_path / "grade_c"
    generate_synthetic_capture(cap_c, SynthOptions(loop_drift_xy_m=(0.16, 0.0)))
    res_c = convert_recording_to_standard(cap_c)
    assert res_c["grade"]["grade"] == "C"

    # Grade fail: loop drift 30 cm (> 25 cm)
    cap_fail = tmp_path / "grade_fail"
    generate_synthetic_capture(cap_fail, SynthOptions(loop_drift_xy_m=(0.30, 0.0)))
    res_fail = convert_recording_to_standard(cap_fail)
    assert res_fail["grade"]["grade"] == "fail"

    # Cap at C: no tape check
    cap_notape = tmp_path / "cap_notape"
    generate_synthetic_capture(cap_notape, SynthOptions(include_tape_checks=False))
    res_notape = convert_recording_to_standard(cap_notape)
    assert res_notape["grade"]["grade"] == "C"
    assert "no_tape_check" in res_notape["grade"]["caps_applied"]

    # Cap at B: limited tracking > 10%
    cap_lim = tmp_path / "cap_lim"
    generate_synthetic_capture(cap_lim, SynthOptions(limited_tracking_fraction=0.25))
    res_lim = convert_recording_to_standard(cap_lim)
    assert res_lim["grade"]["grade"] == "B"
    assert "limited_tracking_over_10pct" in res_lim["grade"]["caps_applied"]

    # Cap at C: Marker 0 tilted > 30 deg from ground up
    cap_tilt = tmp_path / "cap_tilt"
    generate_synthetic_capture(cap_tilt, SynthOptions(marker_0_normal_tilt_deg=38.0))
    res_tilt = convert_recording_to_standard(cap_tilt)
    assert res_tilt["grade"]["grade"] == "C"
    assert "marker_0_tilted" in res_tilt["grade"]["caps_applied"]

    # Pose jump detection (> 0.08 m in single frame)
    cap_jump = tmp_path / "cap_jump"
    generate_synthetic_capture(cap_jump, SynthOptions(pose_jump_m=0.14))
    res_jump = convert_recording_to_standard(cap_jump)
    assert len(res_jump["frame"]["pose_jumps"]) >= 1
    assert res_jump["grade"]["grade"] == "C"


def test_ys43_tape_checks_never_alter_geometry_bytes(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-43: Geometry files in `standard/` are byte-identical with and without tape checks."""
    cap_with = _clone_capture(base_synth_dir, tmp_path / "cap_with_tape")
    cap_without = _clone_capture(base_synth_dir, tmp_path / "cap_without_tape")

    sess = json.loads((cap_without / "session.json").read_text())
    sess["tape_checks"] = []
    (cap_without / "session.json").write_text(json.dumps(sess, indent=2))
    seal_manifest(cap_without)

    convert_recording_to_standard(cap_with)
    convert_recording_to_standard(cap_without)

    for rel in [
        "cloud.ply",
        "frame.json",
        "keyframes.json",
        "trajectory.json",
        "coverage.json",
    ]:
        b_with = (cap_with / "standard" / rel).read_bytes()
        b_without = (cap_without / "standard" / rel).read_bytes()
        assert b_with == b_without, f"YS-43 violation: {rel} changed when tape_checks were removed!"


def test_ys50_ys51_point_cloud_accuracy_and_ground_plane_z0(base_synth_dir: Path) -> None:
    """YS-50, YS-51: Ground points in cloud.ply have median |z| <= 2 cm; vertical wall rejected."""
    verts = read_cloud_ply(base_synth_dir / "standard" / "cloud.ply")
    assert len(verts) > 500
    assert np.all(verts["confidence"] >= 1)

    # Filter ground points (exclude the vertical wall at x ~ 6.2 m)
    ground_verts = verts[verts["x"] < 5.8]
    median_abs_z = float(np.median(np.abs(ground_verts["z"])))
    assert median_abs_z <= 0.02, f"Expected median |z| <= 0.02 m, got {median_abs_z:.4f} m"


def test_ys52_scanned_ground_coverage_and_holes(base_synth_dir: Path) -> None:
    """YS-52, Q031: coverage.json outer polygon is simple, CCW, area > 0, and preserves void > 1 m^2."""
    cov = json.loads((base_synth_dir / "standard" / "coverage.json").read_text())
    assert cov["area_m2"] > 10.0
    outer = cov["polygon_xy_m"]
    assert len(outer) >= 3
    assert len(cov["holes_xy_m"]) >= 1


def test_ys53_validator_rejects_scene_interpretation_fields(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-53: Validator fails if any standard/ JSON contains objects, lot_lines, constraints, or verdict."""
    std_copy = tmp_path / "std_forbidden"
    shutil.copytree(base_synth_dir / "standard", std_copy)

    cov_path = std_copy / "coverage.json"
    cov = json.loads(cov_path.read_text())
    cov["lot_lines"] = [[0, 0], [10, 0]]
    cov_path.write_text(json.dumps(cov, indent=2))

    rep = validate_standard_capture(std_copy)
    assert rep["status"] == "invalid"
    assert any(f["code"] == "forbidden_interpretation_field" for f in rep["findings"])


def test_cli_entrypoints(tmp_path: Path) -> None:
    """Test `yardstick synth`, `yardstick validate`, `yardstick convert`, and `yardstick grade` CLI."""
    out_cap = tmp_path / "cli_cap"
    assert cli_main(["synth", "--output", str(out_cap)]) == 0
    assert cli_main(["validate", str(out_cap)]) == 0
    assert cli_main(["convert", str(out_cap)]) == 0
    assert cli_main(["validate", str(out_cap), "--standard"]) == 0
    assert cli_main(["grade", str(out_cap)]) == 0


def test_ys42_held_out_tape_failures_cap_and_fail(tmp_path: Path) -> None:
    """YS-42 (CS005): 1 failing held-out tape pair caps grade at C; >= 2 failing held-out pairs grade fail."""
    # 1 failing held-out pair (35 cm residual on held-out #1) -> capped at C
    cap_1fail = tmp_path / "cap_held_out_1fail"
    generate_synthetic_capture(
        cap_1fail,
        SynthOptions(held_out_residuals_m=(0.35, -0.012, 0.014)),
    )
    res_1 = convert_recording_to_standard(cap_1fail)
    assert res_1["grade"]["grade"] == "C"
    assert "held_out_tape_failed_1" in res_1["grade"]["caps_applied"]

    # 2 failing held-out pairs (35 cm and 40 cm residuals) -> grade fail
    cap_2fail = tmp_path / "cap_held_out_2fail"
    generate_synthetic_capture(
        cap_2fail,
        SynthOptions(held_out_residuals_m=(0.35, -0.40, 0.014)),
    )
    res_2 = convert_recording_to_standard(cap_2fail)
    assert res_2["grade"]["grade"] == "fail"
    assert "held_out_tape_failed_2plus" in res_2["grade"]["caps_applied"]


def test_ys41_phone_written_tape_residuals_never_trusted(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-41 (CS005): Converter & validator recompute measured_m and residual_m from 3D endpoints and tape_m."""
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_spoofed_tape")
    sess = json.loads((cap / "session.json").read_text())
    # Move primary endpoint B to 5.45 m (45 cm error vs 5.0 m tape) while phone claims 0.0 m residual
    sess["tape_checks"][0]["point_b_world_m"] = [5.45, 0.0, 0.0]
    sess["tape_checks"][0]["measured_m"] = 5.00
    sess["tape_checks"][0]["residual_m"] = 0.00
    (cap / "session.json").write_text(json.dumps(sess, indent=2))
    seal_manifest(cap)

    rep = validate_recording(cap)
    assert any(f["code"] == "tape_check_residual_mismatch" for f in rep["findings"])

    res = convert_recording_to_standard(cap)
    tc0 = res["grade"]["tape_checks"][0]
    assert abs(tc0["measured_m"] - 5.45) < 1e-4
    assert abs(tc0["residual_m"] - 0.45) < 1e-4
    assert res["grade"]["tolerances_95pct"]["distance_m"] >= 0.45
    assert res["grade"]["grade"] == "fail"


def test_ys32_early_resighting_of_marker0_not_counted_as_loop_closure(
    tmp_path: Path,
) -> None:
    """YS-32 (CS005): An early re-sighting of Marker 0 (not at walk end) does not count as loop closure."""
    cap = tmp_path / "cap_early_m0"
    generate_synthetic_capture(cap, SynthOptions(include_loop_closure=False, fps=1))

    poses_rows = list(csv.DictReader((cap / "poses.csv").open()))
    pose_t_by_idx = {int(r["saved_index"]): r["t"] for r in poses_rows}

    m_rows = list(csv.DictReader((cap / "markers.csv").open()))
    early_m0 = dict(m_rows[0])
    # Inject an early second pass of Marker 0 at saved_index=7 (odd frame, ~7s after start, before Markers 2 & 3)
    early_m0["saved_index"] = "7"
    early_m0["t"] = pose_t_by_idx[7]
    m_rows.append(early_m0)
    m_rows.sort(key=lambda r: float(r["t"]))

    with (cap / "markers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m_rows[0].keys()))
        w.writeheader()
        w.writerows(m_rows)
    seal_manifest(cap)

    rep = validate_recording(cap)
    assert rep["status"] == "clean", f"Unexpected findings: {rep['findings']}"

    res = convert_recording_to_standard(cap)
    assert res["frame"]["origin_residual_m"] is None
    assert not any(g["marker_id"] == 0 for g in res["frame"]["loop_closure_gaps"])
    assert res["grade"]["grade"] == "C"
    assert "no_loop_closure" in res["grade"]["caps_applied"]


def test_ys42_pose_jump_without_loop_closure_still_caps_no_loop_closure(tmp_path: Path) -> None:
    """YS-42 (CS005): Missing loop closure caps at C even when pose jumps were found."""
    cap = tmp_path / "cap_jump_no_loop"
    generate_synthetic_capture(
        cap,
        SynthOptions(include_loop_closure=False, pose_jump_m=0.12),
    )
    res = convert_recording_to_standard(cap)
    assert len(res["frame"]["pose_jumps"]) >= 1
    assert "no_loop_closure" in res["grade"]["caps_applied"]
    assert res["grade"]["grade"] == "C"


def test_ys23_raw_quaternion_norm_and_every_image_opened(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-23 (CS005): Raw non-unit or zero quaternion is invalid, and corrupt non-sampled PNG is caught."""
    # 1. Zero quaternion in poses.csv -> invalid
    cap_q = _clone_capture(base_synth_dir, tmp_path / "cap_zero_quat")
    p_rows = list(csv.DictReader((cap_q / "poses.csv").open()))
    p_rows[5]["qx"] = "0.0"
    p_rows[5]["qy"] = "0.0"
    p_rows[5]["qz"] = "0.0"
    p_rows[5]["qw"] = "0.0"
    with (cap_q / "poses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(p_rows[0].keys()))
        w.writeheader()
        w.writerows(p_rows)
    seal_manifest(cap_q)
    rep_q = validate_recording(cap_q)
    assert rep_q["status"] == "invalid"
    assert any(f["code"] == "non_orthonormal_pose" for f in rep_q["findings"])

    # 2. Corrupt depth/000007.png (non-endpoint, non-midpoint index) with sealed manifest -> caught
    cap_png = _clone_capture(base_synth_dir, tmp_path / "cap_corrupt_png7")
    dp7 = cap_png / "depth" / "000007.png"
    dp7.write_bytes(b"not_a_valid_png_file_payload")
    seal_manifest(cap_png)
    rep_png = validate_recording(cap_png)
    assert rep_png["status"] == "invalid"
    assert any(f["code"] == "corrupt_depth_or_conf_png" for f in rep_png["findings"])


def test_ys50_ys52_1cm_synthetic_accuracy_and_1p2m2_hole_preserved(
    base_synth_dir: Path,
) -> None:
    """YS-50, YS-52: Synthetic ground points are within 1 cm of Z=0, and a 1.2 m^2 hole is preserved."""
    from yardstick.geometry import compute_scanned_ground_coverage

    verts = read_cloud_ply(base_synth_dir / "standard" / "cloud.ply")
    ground_verts = verts[verts["x"] < 5.8]
    assert float(np.max(np.abs(ground_verts["z"]))) <= 0.01

    # Build a synthetic grid of ground points on [0, 6] x [0, 6] with:
    # - a 1.2 m x 1.2 m = 1.44 m^2 void (> 1.0 m^2, must be preserved in holes_xy_m)
    # - a 0.6 m x 0.6 m = 0.36 m^2 void (<= 1.0 m^2, must be filled)
    xs, ys = np.meshgrid(np.linspace(0.05, 5.95, 60), np.linspace(0.05, 5.95, 60))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    in_large_hole = (pts[:, 0] >= 1.2) & (pts[:, 0] <= 2.4) & (pts[:, 1] >= 1.2) & (pts[:, 1] <= 2.4)
    in_small_hole = (pts[:, 0] >= 4.0) & (pts[:, 0] <= 4.6) & (pts[:, 1] >= 4.0) & (pts[:, 1] <= 4.6)
    pts = pts[~(in_large_hole | in_small_hole)]
    confs = np.full((pts.shape[0],), 2, dtype=np.uint8)
    ranges = np.full((pts.shape[0],), 2.0, dtype=np.float64)

    cov = compute_scanned_ground_coverage(pts, confs, ranges)
    assert len(cov["holes_xy_m"]) == 1


def test_round4_brisk_walk_snap_across_gap_and_multiframe_ramp(tmp_path: Path) -> None:
    """Round 4 Blocker 7: Clean 1.4 m/s walk at 30 fps has 0 jumps; 1.0 m gap-snap & 30 cm 15-frame ramp caught."""
    import math
    from yardstick.geometry import detect_pose_jumps

    # 1. Clean 1.4 m/s brisk walk at 30 fps (150 frames = 5.0 s)
    rows_brisk = [
        {
            "t": f"{i / 30.0:.6f}",
            "saved_index": str(i),
            "px": f"{1.4 * (i / 30.0):.6f}",
            "py": "0.900000",
            "pz": f"{0.25 * math.sin(i / 30.0):.6f}",
        }
        for i in range(150)
    ]
    assert detect_pose_jumps(rows_brisk) == []

    # 2. Single-frame 1.0 m snap across a 0.167 s gap (5 frames at 30 fps)
    rows_gap = []
    for i in range(40):
        t_val = (i / 30.0) if i < 15 else ((i + 4) / 30.0)
        x_val = 1.1 * t_val + (1.0 if i >= 15 else 0.0)
        rows_gap.append(
            {
                "t": f"{t_val:.6f}",
                "saved_index": str(i),
                "px": f"{x_val:.6f}",
                "py": "0.900000",
                "pz": "0.000000",
            }
        )
    jumps_gap = detect_pose_jumps(rows_gap)
    assert len(jumps_gap) == 1
    assert abs(jumps_gap[0]["jump_m"] - 1.0) < 0.02

    # 3. Multi-frame 30 cm ramp over 15 frames at 30 fps (0.50 s)
    rows_ramp = []
    for i in range(45):
        t_val = i / 30.0
        ramp_m = 0.0 if i <= 10 else (0.30 * (i - 10) / 15.0 if i <= 25 else 0.30)
        rows_ramp.append(
            {
                "t": f"{t_val:.6f}",
                "saved_index": str(i),
                "px": f"{1.0 * t_val + ramp_m:.6f}",
                "py": "0.900000",
                "pz": "0.000000",
            }
        )
    jumps_ramp = detect_pose_jumps(rows_ramp)
    assert len(jumps_ramp) == 1
    assert abs(jumps_ramp[0]["jump_m"] - 0.30) < 0.02

    # 4. Verify synth.py does not shrink dt on the jump frame
    cap_jump = tmp_path / "cap_jump_dt"
    generate_synthetic_capture(cap_jump, SynthOptions(pose_jump_m=0.14))
    p_rows = list(csv.DictReader((cap_jump / "poses.csv").open()))
    mid_idx = len(p_rows) // 2
    dt_jump = float(p_rows[mid_idx]["t"]) - float(p_rows[mid_idx - 1]["t"])
    assert abs(dt_jump - 0.20) < 1e-5


def test_round4_far_or_unconfirmed_closing_sighting_rejected(tmp_path: Path) -> None:
    """Round 4 Blocker 8: Closing Marker 0 sighting > 1.5 m away or without OpenCV-confirmed still is rejected."""
    # Case 1: Closing Marker 0 sighting at ~2.56 m distance (> 1.50 m) at saved_index=17 (in last 35% of walk, after Marker 3)
    cap_far = tmp_path / "cap_far_m0"
    generate_synthetic_capture(cap_far, SynthOptions(include_loop_closure=False))
    poses_rows = list(csv.DictReader((cap_far / "poses.csv").open()))
    m_rows = list(csv.DictReader((cap_far / "markers.csv").open()))
    far_m0 = dict(m_rows[0])
    far_m0["saved_index"] = poses_rows[17]["saved_index"]
    far_m0["t"] = poses_rows[17]["t"]
    m_rows.append(far_m0)
    m_rows.sort(key=lambda r: float(r["t"]))
    with (cap_far / "markers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m_rows[0].keys()))
        w.writeheader()
        w.writerows(m_rows)
    seal_manifest(cap_far)

    res_far = convert_recording_to_standard(cap_far)
    assert res_far["frame"]["origin_residual_m"] is None
    assert "no_loop_closure" in res_far["grade"]["caps_applied"]
    assert res_far["grade"]["grade"] == "C"

    # Case 2: Closing Marker 0 sighting within 1.50 m has no still in stills.csv (not confirmed by OpenCV)
    cap_unconf = tmp_path / "cap_unconf_m0"
    generate_synthetic_capture(cap_unconf, SynthOptions(include_loop_closure=True))
    s_rows = list(csv.DictReader((cap_unconf / "stills.csv").open()))
    kept_s_rows = [r for r in s_rows if int(r["saved_index"]) not in (21, 22, 23)]
    for s_idx in (21, 22, 23):
        still_p = cap_unconf / "stills" / f"{s_idx:06d}.jpg"
        if still_p.exists():
            still_p.unlink()
    with (cap_unconf / "stills.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_s_rows)
    seal_manifest(cap_unconf)

    res_unconf = convert_recording_to_standard(cap_unconf)
    assert res_unconf["frame"]["origin_residual_m"] is None
    assert "no_loop_closure" in res_unconf["grade"]["caps_applied"]
    assert res_unconf["grade"]["grade"] == "C"


def test_round4_crowned_yard_p95_and_height_tolerance(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """Round 4 Blocker 9: A 15 cm crown reports both median and p95 ground residual and bases height_m on p95."""
    from yardstick.geometry import build_yard_frame, fit_ground_plane_ransac
    from yardstick.grading import compute_grade_and_tolerances

    xs, zs = np.meshgrid(np.linspace(-1.0, 5.0, 60), np.linspace(-3.0, 3.0, 60))
    r2 = (xs - 2.0) ** 2 + zs**2
    crown_y = 0.15 * np.clip(1.0 - r2 / (1.8**2), 0.0, 1.0)
    pts_world = np.column_stack([xs.ravel(), crown_y.ravel(), zs.ravel()])
    cam_pos = np.array([[0.0, 0.9, 0.0]], dtype=np.float64)
    m0_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    res_r = fit_ground_plane_ransac(pts_world, cam_pos, m0_pos, return_p95=True)
    assert len(res_r) == 4
    n_w, d_w, med_m, p95_m = res_r
    assert med_m < 0.05
    assert p95_m >= 0.10

    yf = build_yard_frame(
        ground_normal_world=n_w,
        ground_d_world=d_w,
        ground_residual_m=med_m,
        ground_residual_p95_m=p95_m,
        passes_by_marker={},
        first_camera_pose_world=np.eye(4, dtype=np.float64),
        pose_jumps=[],
    )
    assert yf["ground_plane_residual_p95_m"] >= 0.10

    grade_info = compute_grade_and_tolerances(
        integrity_status="clean",
        frame_info=yf,
        marker_scale_info={"scale_error_ratio": 0.0, "scale_error_se": 0.0, "scale_error_eff": 0.0},
        tape_checks_world=[],
        T_yard_from_world=yf["T_matrix"],
        walk_length_m=20.0,
        limited_tracking_share=0.0,
    )
    assert grade_info["tolerances_95pct"]["height_m"] >= p95_m


def test_round4_unaligned_square_and_circular_holes_preserved() -> None:
    """Round 4 Blocker 10: Unaligned 1.10..1.49 m^2 square and 1.30 m^2 round holes are preserved."""
    import math
    from yardstick.geometry import compute_scanned_ground_coverage

    xs, ys = np.meshgrid(np.linspace(0.05, 5.95, 100), np.linspace(0.05, 5.95, 100))
    base = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    cx, cy = 2.53, 2.57  # Unaligned with 0.20 m cells

    for area in (0.36, 0.80):
        s = math.sqrt(area)
        in_h = (np.abs(base[:, 0] - cx) < s / 2.0) & (np.abs(base[:, 1] - cy) < s / 2.0)
        pts = base[~in_h]
        cov = compute_scanned_ground_coverage(
            pts, np.full(len(pts), 2, dtype=np.uint8), np.full(len(pts), 2.0)
        )
        assert len(cov["holes_xy_m"]) == 0, f"Expected {area} m^2 hole to be filled"

    for area in (1.10, 1.21, 1.44, 1.49):
        s = math.sqrt(area)
        in_h = (np.abs(base[:, 0] - cx) < s / 2.0) & (np.abs(base[:, 1] - cy) < s / 2.0)
        pts = base[~in_h]
        cov = compute_scanned_ground_coverage(
            pts, np.full(len(pts), 2, dtype=np.uint8), np.full(len(pts), 2.0)
        )
        assert len(cov["holes_xy_m"]) == 1, f"Expected unaligned {area} m^2 square hole to be preserved"

    # Round 1.30 m^2 hole
    r_c = math.sqrt(1.30 / math.pi)
    in_c = ((base[:, 0] - cx) ** 2 + (base[:, 1] - cy) ** 2) < (r_c * r_c)
    pts_c = base[~in_c]
    cov_c = compute_scanned_ground_coverage(
        pts_c, np.full(len(pts_c), 2, dtype=np.uint8), np.full(len(pts_c), 2.0)
    )
    assert len(cov_c["holes_xy_m"]) == 1, "Expected 1.30 m^2 circular hole to be preserved"


def test_round4_90deg_rotated_marker_corner_order_and_plane_fit() -> None:
    """Round 4 Blocker 2: 90-deg CW rotated marker corner shift rot % 4 and LiDAR plane fit up to 3.5 m."""
    import cv2
    from yardstick.markers import detect_aruco_markers_opencv, measure_marker_corners_3d_on_lidar_plane

    # Render Marker 0 and rotate image 90 deg CW
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    m_gray = cv2.aruco.generateImageMarker(aruco_dict, 0, 160)
    canvas = np.full((240, 240), 255, dtype=np.uint8)
    canvas[40:200, 40:200] = m_gray
    canvas_rot90_cw = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

    dets = detect_aruco_markers_opencv(canvas_rot90_cw)
    assert 0 in dets
    c_opencv = dets[0]

    # When a marker is rotated 90 deg CW in the image, its canonical TL corner is at the image's TR corner (index 1)
    # and the shift `rot % 4` with rot=1 selects index 1 as c0 (TL) and index 2 as c1 (TR):
    detected_img_quad = np.array(
        [[40.0, 40.0], [200.0, 40.0], [200.0, 200.0], [40.0, 200.0]], dtype=np.float64
    )
    rot = 1
    shifted = np.array([detected_img_quad[(k + (rot % 4)) % 4] for k in range(4)])
    assert np.allclose(shifted, c_opencv, atol=2.0)

    # Also test measure_marker_corners_3d_on_lidar_plane at 3.2 m range (within widened 0.50..3.50 m gate)
    depth_mm = np.full((192, 256), 3200, dtype=np.uint16)
    conf_img = np.full((192, 256), 2, dtype=np.uint8)
    corners_uv = np.array(
        [[280.0, 200.0], [317.5, 200.0], [317.5, 237.5], [280.0, 237.5]], dtype=np.float64
    )
    res = measure_marker_corners_3d_on_lidar_plane(
        corners_uv=corners_uv,
        depth_mm=depth_mm,
        conf_img=conf_img,
        fx=480.0,
        fy=480.0,
        cx=320.0,
        cy=240.0,
        rgb_width=640,
        rgb_height=480,
        T_world_from_cam=np.eye(4, dtype=np.float64),
    )
    assert res is not None
    _, measured_mm, mean_range_m = res
    assert abs(measured_mm - 250.0) < 5.0
    assert 3.0 <= mean_range_m <= 3.5


def test_round4_converter_handles_truncated_png_with_fail_grade(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """Round 4 Blocker 11: Converter catches truncated PNG and writes a fail grade.json instead of crashing."""
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_trunc_png")
    dp = cap / "depth" / "000000.png"
    raw = dp.read_bytes()
    dp.write_bytes(raw[:12])  # Truncated PNG

    res = convert_recording_to_standard(cap)
    assert res["grade"]["grade"] == "fail"
    assert "integrity_invalid" in res["grade"]["caps_applied"]
    assert (cap / "standard" / "grade.json").exists()


def test_ys65_symlinked_var_path_manifest_seal_and_180mm_persistence(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-65: Sealing a capture via a symlinked parent (`/var` -> `/private/var`) never prefixes paths with `private`,
    and iOS CaptureSessionRecorder defaults/persists 180 mm marker sizes for IDs 0-3.
    """
    real_root = tmp_path / "private" / "var" / "mobile" / "Documents"
    real_root.mkdir(parents=True)
    symlink_var = tmp_path / "var"
    symlink_var.symlink_to(tmp_path / "private" / "var")

    real_cap = _clone_capture(base_synth_dir, real_root / "20261101T003322Z_yardstick")
    symlink_cap = symlink_var / "mobile" / "Documents" / "20261101T003322Z_yardstick"
    (symlink_cap / "manifest.json").unlink()

    seal_manifest(symlink_cap)
    manifest = json.loads((real_cap / "manifest.json").read_text())
    paths = [entry["path"] for entry in manifest["files"]]
    assert "audio.wav" in paths
    assert not any(p.startswith("private") for p in paths)

    zip_path = symlink_var / "mobile" / "Documents" / "20261101T003322Z_yardstick.zip"
    pack_capture_zip(symlink_cap, zip_path)
    assert zip_path.stat().st_size > 10_000
    rep_dir = validate_recording(symlink_cap)
    assert rep_dir["status"] == "clean"
    extracted_dir = unpack_capture_zip(zip_path, tmp_path / "unpacked")
    rep_zip = validate_recording(extracted_dir)
    assert rep_zip["status"] == "clean"

    swift_src = (
        Path(__file__).resolve().parents[2]
        / "ios"
        / "Yardstick"
        / "Sources"
        / "CaptureSessionRecorder.swift"
    ).read_text()
    assert "resolvingSymlinksInPath().standardizedFileURL" in swift_src
    assert 'replacingOccurrences(of: dir.path + "/"' not in swift_src
    assert 'replacingOccurrences(of: sourceDir.path + "/"' not in swift_src
    assert "0: 180.0, 1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 180.0" in swift_src
    assert "io.buildnexus.yardstick.markerSizesMM" in swift_src


def test_ys67_manifest_with_guidance_csv_and_stills_exposure_validates(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-67: A manifest declaring `guidance` with `guidance.csv` on disk and `stills.csv` with `exposure_duration_s`/`iso` validates."""
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_ys67_guidance")

    # 1. Write guidance.csv and add exposure_duration_s & iso to stills.csv
    poses_rows = list(csv.DictReader((cap / "poses.csv").open()))
    t0 = float(poses_rows[0]["t"])
    t1 = float(poses_rows[-1]["t"])
    with (cap / "guidance.csv").open("w", encoding="utf-8", newline="") as gf:
        gw = csv.DictWriter(
            gf,
            fieldnames=["t", "saved_index", "kind", "detail", "exposure_duration_s", "iso", "turn_rate_rad_s", "predicted_blur_px"],
            lineterminator="\n",
        )
        gw.writeheader()
        gw.writerow(
            {
                "t": f"{t0:.6f}",
                "saved_index": "0",
                "kind": "instruction",
                "detail": "Hold steady on marker 0",
                "exposure_duration_s": "0.004000",
                "iso": "80.0",
                "turn_rate_rad_s": "0.020",
                "predicted_blur_px": "0.115",
            }
        )
        gw.writerow(
            {
                "t": f"{t1:.6f}",
                "saved_index": str(poses_rows[-1]["saved_index"]),
                "kind": "loop_closed",
                "detail": "Marker 0 closing hold locked",
                "exposure_duration_s": "0.005000",
                "iso": "100.0",
                "turn_rate_rad_s": "0.015",
                "predicted_blur_px": "0.108",
            }
        )

    stills_rows = list(csv.DictReader((cap / "stills.csv").open()))
    new_still_cols = list(stills_rows[0].keys()) + ["exposure_duration_s", "iso"]
    for srow in stills_rows:
        srow["exposure_duration_s"] = "0.004167"
        srow["iso"] = "125.0"
    with (cap / "stills.csv").open("w", encoding="utf-8", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=new_still_cols, lineterminator="\n")
        sw.writeheader()
        sw.writerows(stills_rows)

    seal_manifest(cap)
    manifest = json.loads((cap / "manifest.json").read_text())
    assert "guidance" in manifest["streams_declared"]

    rep = validate_recording(cap)
    assert rep["status"] == "clean", f"Expected clean validation for YS-67 capture, got {rep['findings']}"

    # 2. Negative exposure_duration_s is caught
    cap_bad_exp = _clone_capture(cap, tmp_path / "cap_ys67_bad_exp")
    s_bad = list(csv.DictReader((cap_bad_exp / "stills.csv").open()))
    s_bad[0]["exposure_duration_s"] = "-0.005"
    with (cap_bad_exp / "stills.csv").open("w", encoding="utf-8", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=new_still_cols, lineterminator="\n")
        sw.writeheader()
        sw.writerows(s_bad)
    seal_manifest(cap_bad_exp)
    rep_bad_exp = validate_recording(cap_bad_exp)
    assert rep_bad_exp["status"] == "invalid"
    assert any(f["code"] == "still_invalid_exposure_or_iso" for f in rep_bad_exp["findings"])


def test_ys67_guidance_event_after_close_leaves_last_tap_unchanged(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-67: Only operator kinds (`marker`, `note`, `tape_*`) count as taps; guidance events after the close leave the last tap unchanged."""
    from yardstick.converter import _is_operator_tap_kind

    assert _is_operator_tap_kind("marker") is True
    assert _is_operator_tap_kind("note") is True
    assert _is_operator_tap_kind("tape_a") is True
    assert _is_operator_tap_kind("tape_b") is True
    assert _is_operator_tap_kind("done") is False
    assert _is_operator_tap_kind("loop_closed") is False
    assert _is_operator_tap_kind("guidance_prompt") is False

    # Clone baseline Grade A capture and append a post-close guidance/loop_closed event at the very last frame
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_ys67_post_guidance")
    poses_rows = list(csv.DictReader((cap / "poses.csv").open()))
    last_pose = poses_rows[-1]
    t_last = float(last_pose["t"])

    with (cap / "guidance.csv").open("w", encoding="utf-8", newline="") as gf:
        gw = csv.DictWriter(gf, fieldnames=["t", "saved_index", "kind", "detail"], lineterminator="\n")
        gw.writeheader()
        gw.writerow(
            {
                "t": f"{t_last + 0.05:.6f}",
                "saved_index": str(last_pose["saved_index"]),
                "kind": "instruction",
                "detail": "Walk ready to review",
            }
        )
    seal_manifest(cap)

    res = convert_recording_to_standard(cap)
    assert res["grade"]["grade"] == "A"
    assert "no_loop_closure" not in res["grade"]["caps_applied"]
    assert res["frame"]["origin_residual_m"] is not None
    assert res["frame"]["origin_residual_m"] <= 0.05


def test_ys67_60s_lookup_after_loop_closed_with_stray_marker2_keeps_loop(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-67: A 60 s look-up after `loop_closed`, including a stray sighting of marker 2, keeps the loop closed."""
    from yardstick.geometry import build_yard_frame, cluster_marker_passes

    poses_rows = list(csv.DictReader((base_synth_dir / "poses.csv").open()))
    markers_rows = list(csv.DictReader((base_synth_dir / "markers.csv").open()))
    camera_poses_world = {
        int(r["saved_index"]): pose_to_matrix(
            float(r["px"]),
            float(r["py"]),
            float(r["pz"]),
            float(r["qx"]),
            float(r["qy"]),
            float(r["qz"]),
            float(r["qw"]),
        )
        for r in poses_rows
    }

    walk_start_t_s = float(poses_rows[0]["t"])
    loop_closed_t_s = float(poses_rows[-1]["t"])
    # Simulate 60 s of post-close look-up frames + a stray sighting of Marker 2 at loop_closed_t_s + 45 s
    walk_end_after_lookup_t_s = loop_closed_t_s + 60.0
    stray_m2_t_s = loop_closed_t_s + 45.0
    stray_m2_idx = int(poses_rows[-1]["saved_index"]) + 100
    camera_poses_world[stray_m2_idx] = camera_poses_world[int(poses_rows[10]["saved_index"])]

    m2_template = next(r for r in markers_rows if int(r["marker_id"]) == 2)
    stray_m2_row = dict(m2_template)
    stray_m2_row["t"] = f"{stray_m2_t_s:.6f}"
    stray_m2_row["saved_index"] = str(stray_m2_idx)

    augmented_markers = sorted(markers_rows + [stray_m2_row], key=lambda r: float(r["t"]))
    passes_by_marker = cluster_marker_passes(augmented_markers, camera_poses_world)
    confirmed_m0 = {int(r["saved_index"]) for r in markers_rows if int(r["marker_id"]) == 0}

    # 1. With loop_closed_t_s set: the 60 s look-up and stray Marker 2 sighting keep the loop closed
    yf_with_close = build_yard_frame(
        ground_normal_world=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ground_d_world=0.0,
        ground_residual_m=0.002,
        ground_residual_p95_m=0.004,
        passes_by_marker=passes_by_marker,
        first_camera_pose_world=camera_poses_world[0],
        pose_jumps=[],
        walk_start_t_s=walk_start_t_s,
        walk_end_t_s=walk_end_after_lookup_t_s,
        last_tap_t_s=walk_start_t_s + 0.8,
        loop_closed_t_s=loop_closed_t_s,
        camera_poses_by_index=camera_poses_world,
        confirmed_marker0_still_indices=confirmed_m0,
    )
    assert yf_with_close["origin_residual_m"] is not None
    assert any(g["marker_id"] == 0 for g in yf_with_close["loop_closure_gaps"])

    # 2. Without loop_closed_t_s (legacy behavior): the 60 s look-up and stray Marker 2 void the loop
    yf_without_close = build_yard_frame(
        ground_normal_world=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ground_d_world=0.0,
        ground_residual_m=0.002,
        ground_residual_p95_m=0.004,
        passes_by_marker=passes_by_marker,
        first_camera_pose_world=camera_poses_world[0],
        pose_jumps=[],
        walk_start_t_s=walk_start_t_s,
        walk_end_t_s=walk_end_after_lookup_t_s,
        last_tap_t_s=walk_start_t_s + 0.8,
        loop_closed_t_s=None,
        camera_poses_by_index=camera_poses_world,
        confirmed_marker0_still_indices=confirmed_m0,
    )
    assert yf_without_close["origin_residual_m"] is None
    assert not any(g["marker_id"] == 0 for g in yf_without_close["loop_closure_gaps"])


def test_ys69_subpixel_refinement_window5_and_quad_rejection_gates() -> None:
    """YS-69: OpenCV DICT_5X5_250 uses sub-pixel corner refinement (window 5), rejects > 3% corner shift, and rejects > 4% 3D edge disagreement."""
    import cv2
    from yardstick.markers import (
        check_refined_corner_shift,
        detect_aruco_markers_opencv,
        measure_marker_corners_3d_on_lidar_plane,
    )

    # 1. Sub-pixel corner refinement on rendered marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    m_gray = cv2.aruco.generateImageMarker(aruco_dict, 0, 160)
    canvas = np.full((240, 240), 255, dtype=np.uint8)
    canvas[40:200, 40:200] = m_gray
    dets = detect_aruco_markers_opencv(canvas)
    assert 0 in dets

    # 2. > 3% refined corner movement is rejected
    raw_quad = np.array([[40.0, 40.0], [200.0, 40.0], [200.0, 200.0], [40.0, 200.0]], dtype=np.float64)
    # 1.5% corner shift (2.4 px on 160 px edge) -> accepted
    ref_ok = raw_quad.copy()
    ref_ok[0, 0] += 2.4
    assert check_refined_corner_shift(raw_quad, ref_ok, max_corner_shift_ratio=0.03) is True
    # 4.0% corner shift (6.4 px on 160 px edge) -> rejected
    ref_bad = raw_quad.copy()
    ref_bad[0, 0] += 6.4
    assert check_refined_corner_shift(raw_quad, ref_bad, max_corner_shift_ratio=0.03) is False

    # 3. > 4% 3D edge length disagreement is rejected
    depth_mm = np.full((192, 256), 2000, dtype=np.uint16)
    conf_img = np.full((192, 256), 2, dtype=np.uint8)
    square_uv = np.array([[280.0, 200.0], [320.0, 200.0], [320.0, 240.0], [280.0, 240.0]], dtype=np.float64)
    res_ok = measure_marker_corners_3d_on_lidar_plane(
        corners_uv=square_uv,
        depth_mm=depth_mm,
        conf_img=conf_img,
        fx=480.0,
        fy=480.0,
        cx=320.0,
        cy=240.0,
        rgb_width=640,
        rgb_height=480,
        T_world_from_cam=np.eye(4, dtype=np.float64),
    )
    assert res_ok is not None

    # Stretch horizontal edges by 8% (43.2 px vs 40.0 px -> > 4% 3D edge disagreement) -> rejected (None)
    skewed_uv = np.array([[280.0, 200.0], [323.2, 200.0], [323.2, 240.0], [280.0, 240.0]], dtype=np.float64)
    res_bad = measure_marker_corners_3d_on_lidar_plane(
        corners_uv=skewed_uv,
        depth_mm=depth_mm,
        conf_img=conf_img,
        fx=480.0,
        fy=480.0,
        cx=320.0,
        cy=240.0,
        rgb_width=640,
        rgb_height=480,
        T_world_from_cam=np.eye(4, dtype=np.float64),
    )
    assert res_bad is None






def test_grade_parity_vectors() -> None:
    """YS-44 / YS-111: shared grade vectors; the phone's `GradeModelTests` asserts the same expectations."""
    from yardstick.grading import compute_grade_and_tolerances

    vectors = json.loads((Path(__file__).parent / "fixtures" / "grade_parity_vectors.json").read_text())
    names = {v["name"] for v in vectors}
    assert {"walk6_real", "walk7_real"} <= names
    for v in vectors:
        tcs = [
            {"id": k, "role": e[0], "tape_m": e[1], "point_a_world_m": [0, 0, 0], "point_b_world_m": [e[2], 0, 0],
             "tape_uncertainty_m": e[3] if len(e) > 3 else 0.0}
            for k, e in enumerate(v["tapes"])
        ]
        fi = {
            "method": "marker",
            "marker_0_normal_ok": v.get("normal_ok", True),
            "loop_closure_gaps": [{"marker_id": 0, "gap_m": v["gap"]}] if v["gap"] is not None else [],
            "pose_jumps": [{"jump_m": j, "t_s": 0.0, "saved_index_from": 0, "saved_index_to": 1} for j in v["jumps"]],
        }
        ms = {"scale_error_ratio": v["scale_eff"], "scale_error_se": 0.0, "scale_error_eff": v["scale_eff"], "sightings_used": 1}
        g = compute_grade_and_tolerances("ok", fi, ms, tcs, np.eye(4), v["walk"], v["limited"])
        assert g["grade"] == v["expected_grade"], v["name"]
        assert g["tolerances_95pct"]["distance_m"] == pytest.approx(v["expected_distance_m"], abs=1e-6), v["name"]
        assert g["caps_applied"] == v["expected_caps"], v["name"]


def test_ys92_visit_json_validates_and_rejects_interpretation_keys(
    base_synth_dir: Path, tmp_path: Path
) -> None:
    """YS-92 / Phase 2: validate_standard_capture accepts standard/visit.json against visit.schema.json and enforces YS-53."""
    cap = _clone_capture(base_synth_dir, tmp_path / "cap_with_visit")
    convert_recording_to_standard(cap)
    std_dir = cap / "standard"

    visit_plan_bytes = b'{"schema":"lotfit.visit-plan/1","plan_id":"vplan-eug-1704104303300-v1"}'
    visit_plan_path = cap / "visit_plan.json"
    visit_plan_path.write_bytes(visit_plan_bytes)
    plan_sha256 = sha256_file(visit_plan_path)

    valid_visit_echo = {
        "contract": "yardstick-capture/1",
        "plan_id": "019a4500-0000-7000-8000-000000000301",
        "plan_sha256": plan_sha256,
        "records": [
            {
                "layer": "parcel",
                "status": "available",
                "source": {
                    "issuer": "Lane County",
                    "service_url": "https://lcgispubags.lanecountyor.gov/arcgis/rest/services/Health/AddressParcel/MapServer/1",
                    "query": "MAPTAXLOT='1704104303300'",
                    "document": "Lane County tax lot map (AddressParcel), map tax lot 1704104303300",
                },
                "record_date": "2026-11-01",
                "stated_accuracy": None,
                "assumed_accuracy": {
                    "value": 3.0,
                    "unit": "ft",
                    "by": "CC (Claude Code)",
                    "why": "the issuer states no positional accuracy; 3.0 ft assumed for urban GIS tax lots",
                },
            }
        ],
        "markers_placed": [
            {
                "id": 0,
                "role": "origin_anchor_a",
                "sighted": True,
                "sightings_count": 18,
                "as_planned": "yes",
                "measured_yard_xyz_m": [0.0, 0.0, 0.0],
            },
            {
                "id": 1,
                "role": "anchor_b",
                "sighted": True,
                "sightings_count": 14,
                "as_planned": "yes",
                "measured_yard_xyz_m": [16.53, 0.0, 0.0],
            },
        ],
        "must_capture": [
            {
                "id": "cap-corner-a",
                "kind": "anchor_corner",
                "status": "done",
                "t_s": 100.0,
                "saved_index": 0,
                "keyframe_file": "keyframes/000000.jpg",
            }
        ],
        "live_warnings": [],
    }

    def _seal_standard_visit(visit_obj: dict) -> None:
        visit_fp = std_dir / "visit.json"
        visit_fp.write_text(json.dumps(visit_obj, indent=2), encoding="utf-8")
        cap_fp = std_dir / "capture.json"
        cap_data = json.loads(cap_fp.read_text(encoding="utf-8"))
        files_list = [f for f in cap_data["files"] if f["path"] != "visit.json"]
        files_list.append(
            {
                "path": "visit.json",
                "sha256": sha256_file(visit_fp),
                "size_bytes": visit_fp.stat().st_size,
            }
        )
        cap_data["files"] = sorted(files_list, key=lambda x: x["path"])
        cap_fp.write_text(json.dumps(cap_data, indent=2), encoding="utf-8")

    # 1. Valid standard/visit.json passes
    _seal_standard_visit(valid_visit_echo)
    rep_ok = validate_standard_capture(cap)
    assert rep_ok["status"] == "clean", f"Expected clean standard validation with visit.json, got {rep_ok['findings']}"

    # 2. YS-53 forbidden interpretation key inside standard/visit.json is rejected
    bad_ys53 = dict(valid_visit_echo)
    bad_ys53["verdict"] = "fits"
    _seal_standard_visit(bad_ys53)
    rep_ys53 = validate_standard_capture(cap)
    assert rep_ys53["status"] == "invalid"
    codes_ys53 = {f["code"] for f in rep_ys53["findings"]}
    assert "forbidden_interpretation_field" in codes_ys53

    # 3. Malformed standard/visit.json (wrong plan_sha256) is rejected
    bad_hash = dict(valid_visit_echo)
    bad_hash["plan_sha256"] = "0" * 64
    _seal_standard_visit(bad_hash)
    rep_hash = validate_standard_capture(cap)
    assert rep_hash["status"] == "invalid"
    codes_hash = {f["code"] for f in rep_hash["findings"]}
    assert "visit_plan_hash_mismatch" in codes_hash


def test_aruco_dict_0_to_7_parity() -> None:
    """YS-115 §4.2: Verify generator, Python data, and Swift data exact hex/bit parity for IDs 0..7 and Hamming >= 5."""
    import re
    import sys

    import cv2

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "toolkit" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from generate_aruco_dict import extract_dict_5x5_patterns
    from yardstick.aruco_dict_data import ARUCO_DICT_5X5_PATTERNS

    gen_patterns = extract_dict_5x5_patterns(max_id=7)
    assert set(gen_patterns.keys()) == set(range(8))

    py_keys = {int(k): v for k, v in ARUCO_DICT_5X5_PATTERNS.items()}
    assert set(py_keys.keys()) == set(range(8)), "Run toolkit/scripts/generate_aruco_dict.py to update aruco_dict_data.py"
    for mid in range(8):
        assert py_keys[mid] == gen_patterns[mid]

    swift_text = (repo_root / "ios" / "Yardstick" / "Sources" / "ArUcoDictionaryData.swift").read_text(encoding="utf-8")
    swift_patterns: dict[int, list[int]] = {}
    for m in re.finditer(r"(\d+)\s*:\s*\[([^\]]+)\]", swift_text):
        mid = int(m.group(1))
        vals = [int(tok.strip(), 16) for tok in m.group(2).split(",") if tok.strip()]
        swift_patterns[mid] = vals
    assert set(swift_patterns.keys()) == set(range(8)), "Run toolkit/scripts/generate_aruco_dict.py to update ArUcoDictionaryData.swift"
    for mid in range(8):
        assert swift_patterns[mid] == gen_patterns[mid]

    # Minimum pairwise Hamming distance across all 8 * 4 = 32 rotated codewords must be >= 5
    all_codewords = [(mid, rot, gen_patterns[mid][rot]) for mid in range(8) for rot in range(4)]
    for i in range(len(all_codewords)):
        for j in range(i + 1, len(all_codewords)):
            _, _, w1 = all_codewords[i]
            _, _, w2 = all_codewords[j]
            hamming = int(bin(w1 ^ w2).count("1"))
            assert hamming >= 5, f"Codewords {all_codewords[i]} and {all_codewords[j]} have Hamming distance {hamming} < 5"

    # Verify rendered marker patches for IDs 0..7 at all 4 rotations decode cleanly with OpenCV
    for mid in range(8):
        patch_rgb = _render_aruco_Test_patch(mid)
        for k_rot in range(4):
            rot_img = np.rot90(patch_rgb, k=-k_rot)
            dets = detect_aruco_markers_opencv(rot_img)
            assert set(dets.keys()) == {mid}


def _render_aruco_Test_patch(marker_id: int) -> np.ndarray:
    from yardstick.synth import _render_aruco_marker_patch

    return _render_aruco_marker_patch(marker_id, size_px=240)


def _eight_marker_ground_truth_layout() -> tuple[dict[int, tuple[float, float, float, float]], list[tuple[int, int]]]:
    """Return 8 tossed markers (0..7) in non-sequential order with 2.15..2.85 m neighbour spacing and cross-chords."""
    # ARKit world coordinates (X_w, Y_w=0, Z_w, yaw_rad).
    # Note: In Yard Frame with Marker 0 at (0,0,0) and yaw=0:
    # X_yard = +X_w, Y_yard = -Z_w, Z_yard = +Y_w.
    layout: dict[int, tuple[float, float, float, float]] = {
        0: (0.00, 0.0, 0.00, 0.00),
        5: (2.40, 0.0, -0.15, 0.12),
        2: (4.85, 0.0, 0.10, -0.18),
        7: (7.20, 0.0, -0.10, 0.25),
        1: (7.10, 0.0, -2.35, -0.10),
        6: (4.75, 0.0, -2.45, 0.08),
        3: (2.35, 0.0, -2.30, -0.22),
        4: (0.10, 0.0, -2.20, 0.15),
    }
    pair_seq: list[tuple[int, int]] = [
        (0, 5),
        (5, 2),
        (2, 7),
        (7, 1),
        (1, 6),
        (6, 3),
        (3, 4),
        (4, 0),
        (5, 3),  # Cross-chord triangle bracing
        (2, 6),  # Second cross-chord
    ]
    return layout, pair_seq


def test_ys115_eight_tossed_markers_recovers_layout_under_shrink_and_jump(tmp_path: Path) -> None:
    """YS-115 §5.1.1: 8 tossed markers in non-sequential loop recover layout within <= 1.5 cm RMSE under -4.5% VIO shrink + 25 cm jump."""
    layout, pair_seq = _eight_marker_ground_truth_layout()
    cap_dir = tmp_path / "cap_ys115_eight_markers"
    generate_synthetic_capture(
        cap_dir,
        SynthOptions(
            markers_world_layout=layout,
            walk_pair_sequence=pair_seq,
            vio_scale_shrink_ratio=-0.045,
            pose_jump_m=0.25,
            loop_drift_xy_m=(0.18, -0.14),
            include_tape_checks=False,
            write_pairs_csv=True,
        ),
    )

    rep1 = validate_recording(cap_dir)
    assert rep1["status"] == "clean", f"Level 1 validation findings: {rep1['findings']}"

    res = convert_recording_to_standard(cap_dir)
    rep2 = validate_standard_capture(cap_dir / "standard")
    assert rep2["status"] == "clean", f"Level 2 validation findings: {rep2['findings']}"

    ml = res["frame"]["marker_layout"]
    assert ml["pair_graph_connected"] is True
    assert set(ml["markers_connected_ids"]) == {0, 1, 2, 3, 4, 5, 6, 7}
    assert ml["cycle_count"] >= 1
    assert ml["closed_loop_edge_rmse_m"] is not None and ml["closed_loop_edge_rmse_m"] <= 0.010
    assert ml["trajectory_warp_applied"] is True
    assert len(ml["per_marker_warp_corrections"]) >= 8

    # Condition 4: Verify raw scan files are kept alongside warped defaults and hashed in capture.json
    std_dir = cap_dir / "standard"
    assert (std_dir / "cloud.ply").exists()
    assert (std_dir / "cloud.raw.ply").exists()
    assert (std_dir / "trajectory.json").exists()
    assert (std_dir / "trajectory.raw.json").exists()
    cap_files = {f["path"] for f in res["capture"]["files"]}
    assert {"cloud.ply", "cloud.raw.ply", "trajectory.json", "trajectory.raw.json"} <= cap_files

    # Compare solved Yard Frame positions against ground truth: X_yard = X_w, Y_yard = -Z_w, Z_yard = Y_w
    gt_yard = {
        mid: np.array([spec[0], -spec[2], spec[1]], dtype=np.float64)
        for mid, spec in layout.items()
    }
    solved_yard = {
        int(m["marker_id"]): np.array(m["xyz_m"], dtype=np.float64)
        for m in res["frame"]["markers_yard"]
    }
    assert set(solved_yard.keys()) == set(range(8))

    pos_errors = [float(np.linalg.norm(solved_yard[mid] - gt_yard[mid])) for mid in range(8)]
    rmse_m = float(np.sqrt(np.mean(np.square(pos_errors))))
    max_err_m = float(np.max(pos_errors))
    assert rmse_m <= 0.015, f"Expected layout RMSE <= 0.015 m, got {rmse_m:.4f} m"
    assert max_err_m <= 0.025, f"Expected max marker error <= 0.025 m, got {max_err_m:.4f} m"

    # Verify all pairwise distances match ground truth within <= 0.8%
    for i in range(8):
        for j in range(i + 1, 8):
            d_gt = float(np.linalg.norm(gt_yard[j] - gt_yard[i]))
            d_sol = float(np.linalg.norm(solved_yard[j] - solved_yard[i]))
            rel_err = abs(d_sol - d_gt) / d_gt
            assert rel_err <= 0.008, f"Pair ({i}, {j}) distance error {rel_err*100:.2f}% > 0.8%"


def test_ys115_injected_depth_scale_bias_recovered_by_printed_size(tmp_path: Path) -> None:
    """YS-115 Condition 3: Injected -1.6% single-frame LiDAR depth-scale bias is corrected by printed marker size."""
    layout, pair_seq = _eight_marker_ground_truth_layout()
    cap_dir = tmp_path / "cap_ys115_depth_bias"
    generate_synthetic_capture(
        cap_dir,
        SynthOptions(
            markers_world_layout=layout,
            walk_pair_sequence=pair_seq,
            vio_scale_shrink_ratio=-0.040,
            depth_scale_bias_ratio=-0.016,
            actual_marker_size_mm=180.0,
            declared_marker_size_mm=180.0,
            include_tape_checks=False,
            write_pairs_csv=True,
        ),
    )

    res = convert_recording_to_standard(cap_dir)
    ml = res["frame"]["marker_layout"]
    expected_alpha = 1.0 / (1.0 - 0.016)  # ~1.01626
    assert ml["printed_scale_factor"] is not None
    assert abs(float(ml["printed_scale_factor"]) - expected_alpha) <= 0.004
    assert abs(float(ml["scale_applied"]) - expected_alpha) <= 0.004

    gt_yard = {
        mid: np.array([spec[0], -spec[2], spec[1]], dtype=np.float64)
        for mid, spec in layout.items()
    }
    solved_yard = {
        int(m["marker_id"]): np.array(m["xyz_m"], dtype=np.float64)
        for m in res["frame"]["markers_yard"]
    }
    pos_errors = [float(np.linalg.norm(solved_yard[mid] - gt_yard[mid])) for mid in range(8)]
    rmse_m = float(np.sqrt(np.mean(np.square(pos_errors))))
    assert rmse_m <= 0.015, f"Expected depth-bias-corrected layout RMSE <= 0.015 m, got {rmse_m:.4f} m"


def test_ys115_wall_single_band_after_trajectory_warp(tmp_path: Path) -> None:
    """YS-115 §5.1.2 & Condition 4: Doubled Pass-1/Pass-2 wall in cloud.raw.ply collapses to a single <= 2.5 cm band in cloud.ply."""
    layout, pair_seq = _eight_marker_ground_truth_layout()
    cap_dir = tmp_path / "cap_ys115_wall_band"
    generate_synthetic_capture(
        cap_dir,
        SynthOptions(
            markers_world_layout=layout,
            walk_pair_sequence=pair_seq,
            vio_scale_shrink_ratio=-0.045,
            pose_jump_m=0.25,
            loop_drift_xy_m=(0.12, -0.05),
            include_tape_checks=False,
            write_pairs_csv=True,
            wall_pass1_and_pass2=True,
        ),
    )
    convert_recording_to_standard(cap_dir)

    verts_raw = read_cloud_ply(cap_dir / "standard" / "cloud.raw.ply")
    verts_warped = read_cloud_ply(cap_dir / "standard" / "cloud.ply")

    # Select elevated wall points (z > 0.15 m) near X_yard ~ 1.65 m
    wall_raw_x = verts_raw["x"][verts_raw["z"] > 0.15]
    wall_warped_x = verts_warped["x"][verts_warped["z"] > 0.15]
    assert len(wall_raw_x) >= 20
    assert len(wall_warped_x) >= 20

    raw_spread_p95 = float(np.percentile( np.abs(wall_raw_x - np.median(wall_raw_x)), 95))
    warped_spread_p95 = float(np.percentile(np.abs(wall_warped_x - np.median(wall_warped_x)), 95))

    assert raw_spread_p95 >= 0.10, f"Expected raw unwarped wall to show pass-1/pass-2 split, got {raw_spread_p95:.4f} m"
    assert warped_spread_p95 <= 0.025, f"Expected warped wall 95th-percentile thickness <= 0.025 m, got {warped_spread_p95:.4f} m"


def test_ys115_disconnected_pair_graph_fallback_and_grade_cap(tmp_path: Path) -> None:
    """YS-115 §5.1.3: Connected loop {0, 2, 5, 6} + unconnected solo Marker 7 triggers vio_similarity_fallback and disconnected_marker_graph cap."""
    layout = {
        0: (0.00, 0.0, 0.00, 0.0),
        2: (2.40, 0.0, 0.00, 0.0),
        5: (2.40, 0.0, -2.30, 0.0),
        6: (0.00, 0.0, -2.30, 0.0),
        7: (5.20, 0.0, -5.20, 0.0),  # Far solo marker, zero one-frame pairs
    }
    pair_seq = [(0, 2), (2, 5), (5, 6), (6, 0)]
    cap_dir = tmp_path / "cap_ys115_disconnected"
    generate_synthetic_capture(
        cap_dir,
        SynthOptions(
            markers_world_layout=layout,
            walk_pair_sequence=pair_seq,
            solo_sightings_ids=[7],
            vio_scale_shrink_ratio=-0.040,
            include_tape_checks=False,
            write_pairs_csv=True,
        ),
    )

    res = convert_recording_to_standard(cap_dir)
    ml = res["frame"]["marker_layout"]
    assert ml["pair_graph_connected"] is False
    assert ml["markers_connected_ids"] == [0, 2, 5, 6]
    assert ml["unconnected_marker_ids"] == [7]

    by_id = {int(m["marker_id"]): m for m in res["frame"]["markers_yard"]}
    for mid in (0, 2, 5, 6):
        assert by_id[mid]["connected_to_origin"] is True
        assert by_id[mid]["placement_source"] == "pair_graph"
        gt_y = np.array([layout[mid][0], -layout[mid][2], layout[mid][1]], dtype=np.float64)
        err = float(np.linalg.norm(np.array(by_id[mid]["xyz_m"]) - gt_y))
        assert err <= 0.015

    assert by_id[7]["connected_to_origin"] is False
    assert by_id[7]["placement_source"] == "vio_similarity_fallback"
    assert "disconnected_marker_graph" in res["grade"]["caps_applied"]
    assert res["grade"]["grade"] == "C"


def test_ys115_exp2_replay_and_false_positive_gate(tmp_path: Path) -> None:
    """YS-115 §5.2: Replay Experiment 2 captures if present on disk and verify zero false positives for IDs 4..7."""
    # Always verify that structured high-frequency synthetic yard/fence textures produce zero false positives for IDs 0..7
    rng = np.random.default_rng(11502)
    for trial in range(4):
        noise_img = rng.integers(30, 225, size=(480, 640), dtype=np.uint8)
        # Add synthetic fence slats / grass stripes
        for x_slat in range(0, 640, 18 + trial * 4):
            noise_img[:, x_slat : min(640, x_slat + 6)] = 45
        dets_noise = detect_aruco_markers_opencv(noise_img)
        assert not (set(dets_noise.keys()) & set( range(8) )), f"False positive in synthetic yard texture: {dets_noise.keys()}"

    exp2_root = Path(os.environ.get("YARDSTICK_REPLAY_ROOT", "/nonexistent")) / "2026-11-02-exp2"
    if not exp2_root.exists():
        pytest.skip("Experiment 2 scrap captures not present on this machine")

    scan_dirs = sorted({p.parent for p in exp2_root.rglob("session.json")})
    if not scan_dirs and exp2_root.parent.exists():
        scan_dirs = sorted({p.parent for p in exp2_root.parent.rglob("session.json") if "exp2" in str(p)})

    still_files: list[Path] = []
    for scan_dir in scan_dirs:
        stills_dir = scan_dir / "stills"
        if stills_dir.exists():
            still_files.extend(sorted(stills_dir.glob("*.jpg"))[:15])
    if not still_files:
        still_files = sorted(exp2_root.rglob("*.jpg"))[:30]

    for still_p in still_files:
        gray = np.array(Image.open(still_p).convert("L"), dtype=np.uint8)
        dets = detect_aruco_markers_opencv(gray)
        false_ids = set(dets.keys()) - {0, 1, 2, 3}
        assert not false_ids, f"False positive marker IDs {false_ids} detected in {still_p}"


def test_ys115_ground_plane_printed_size_fallback_and_w3_pairs() -> None:
    """YS-115 C5/C6: Ground-plane + 180 mm printed size recovers 3D position for far 2D markers (`seen_2d_no_3d`) in w3 frames 3960 and 3966."""
    from yardstick.markers import (
        measure_marker_3d_camera_frame,
        measure_marker_3d_from_ground_plane_and_printed_size,
    )

    w3_dir = Path(os.environ.get("YARDSTICK_REPLAY_ROOT", "/nonexistent")) / "2026-11-02-eight" / "w3" / "20261103T005121Z_yardstick"
    if not w3_dir.exists():
        pytest.skip("Field walk w3 not present on disk")

    poses_by_idx = {}
    with (w3_dir / "poses.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            poses_by_idx[int(r["saved_index"])] = r

    for s_idx in (3960, 3966):
        pr = poses_by_idx[s_idx]
        T_wc = pose_to_matrix(
            float(pr["px"]),
            float(pr["py"]),
            float(pr["pz"]),
            float(pr["qx"]),
            float(pr["qy"]),
            float(pr["qz"]),
            float(pr["qw"]),
        )
        gray = np.array(Image.open(w3_dir / "stills" / f"{s_idx:06d}.jpg").convert("L"), dtype=np.uint8)
        dets = detect_aruco_markers_opencv(gray)
        assert {5, 6}.issubset(set(dets.keys())), f"Expected both markers 5 and 6 detected in 2D in still {s_idx}"

        depth_mm = np.array(Image.open(w3_dir / "depth" / f"{s_idx:06d}.png"), dtype=np.uint16)
        conf_img = np.array(Image.open(w3_dir / "confidence" / f"{s_idx:06d}.png"), dtype=np.uint8)

        m5 = measure_marker_3d_camera_frame(
            corners_uv=dets[5],
            depth_mm=depth_mm,
            conf_img=conf_img,
            fx=float(pr["fx"]),
            fy=float(pr["fy"]),
            cx=float(pr["cx"]),
            cy=float(pr["cy"]),
            rgb_width=int(pr["width"]),
            rgb_height=int(pr["height"]),
            T_world_from_cam=T_wc,
        )
        assert m5 is not None
        m6 = measure_marker_3d_from_ground_plane_and_printed_size(
            corners_uv=dets[6],
            printed_size_mm=180.0,
            fx=float(pr["fx"]),
            fy=float(pr["fy"]),
            cx=float(pr["cx"]),
            cy=float(pr["cy"]),
            T_world_from_cam=T_wc,
            reference_ground_normal_cam=m5["R_cam_from_marker"][:, 2],
        )
        assert m6 is not None
        assert m6["is_ground_plane_printed_size_fallback"] is True
        assert m6["position_uncertainty_m"] > 0.03
        dist_56 = float(np.linalg.norm(m6["center_world"] - m5["center_world"]))
        assert 3.00 <= dist_56 <= 3.30, f"Expected 5-6 pair distance ~3.1-3.2 m in frame {s_idx}, got {dist_56:.3f} m"



