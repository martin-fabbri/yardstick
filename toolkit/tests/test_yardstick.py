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
