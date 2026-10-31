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
from yardstick.validator import validate_recording, validate_standard_capture


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


