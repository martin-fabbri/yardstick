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

"""Closed-form 95% tolerance model and grading engine (YS-40, YS-41, YS-42, YS-43, YS-44, YS-45)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from yardstick import CONTRACT_VERSION, GRADING_VERSION

SIGMA_LIDAR_M = 0.015
SIGMA_TRACKING_BASE_M = 0.005
SIGMA_TRACKING_SLOPE_M = 0.050
REFERENCE_RANGE_M = 10.0

FORMULA_STRING = (
    "max(primary_tape_floor(r), 1.96 * sqrt(sigma_lidar_m^2 + "
    "(scale_error_eff * r)^2 + (loop_gap_per_m * r)^2 + sigma_tracking_m^2))"
)


def recompute_tape_check(tc: dict[str, Any]) -> dict[str, Any]:
    """Recompute measured distance and residual from recorded 3D endpoints and typed tape_m (YS-41).

    Never trusts phone-written `measured_m` or `residual_m`; flags mismatches > 1 mm.
    """
    pa_w = np.array(tc["point_a_world_m"], dtype=np.float64)
    pb_w = np.array(tc["point_b_world_m"], dtype=np.float64)
    tape_m = float(tc["tape_m"])
    measured_m = float(np.linalg.norm(pb_w - pa_w))
    residual_m = abs(measured_m - tape_m)
    phone_measured = float(tc.get("measured_m", measured_m))
    phone_residual = float(tc.get("residual_m", residual_m))
    phone_mismatch = (abs(phone_measured - measured_m) > 1e-3) or (abs(phone_residual - residual_m) > 1e-3)
    out = dict(tc)
    out["tape_m"] = tape_m
    out["measured_m"] = measured_m
    out["residual_m"] = residual_m
    out["phone_measured_m"] = phone_measured
    out["phone_residual_m"] = phone_residual
    out["phone_mismatch"] = phone_mismatch
    # YS-86: the tape has its own error (grass ~5 cm, along a wall ~2 cm). Only the part of the residual the tape
    # cannot explain counts against the phone; independent errors combine in quadrature.
    u_m = max(0.0, float(tc.get("tape_uncertainty_m", 0.0)))
    out["tape_uncertainty_m"] = u_m
    out["residual_beyond_tape_m"] = residual_beyond_tape(residual_m, u_m)
    return out


def residual_beyond_tape(residual_m: float, tape_uncertainty_m: float) -> float:
    """Part of a tape residual not explained by the tape's own error: sqrt(max(0, r^2 - u^2)) (YS-86)."""
    return math.sqrt(max(0.0, residual_m * residual_m - tape_uncertainty_m * tape_uncertainty_m))


def compute_tolerance_at_range(
    r_m: float,
    primary_tape_checks: list[dict[str, Any]],
    scale_error_eff: float,
    loop_gap_per_m: float,
    sigma_tracking_m: float,
    sigma_lidar_m: float = SIGMA_LIDAR_M,
) -> float:
    """Evaluate closed-form 95% distance tolerance tol(r) at range r_m (YS-41, Q029)."""
    primary_floor = 0.0
    for tc in primary_tape_checks:
        rtc = recompute_tape_check(tc) if "point_a_world_m" in tc else tc
        res_m = residual_beyond_tape(float(rtc["residual_m"]), float(rtc.get("tape_uncertainty_m", 0.0)))
        tape_m = max(float(rtc["tape_m"]), 2.0)
        scaled = res_m * max(1.0, float(r_m) / tape_m)
        primary_floor = max(primary_floor, res_m, scaled)

    rss = math.sqrt(
        sigma_lidar_m * sigma_lidar_m
        + (scale_error_eff * r_m) ** 2
        + (loop_gap_per_m * r_m) ** 2
        + sigma_tracking_m * sigma_tracking_m
    )
    return max(primary_floor, 1.96 * rss)


def compute_grade_and_tolerances(
    integrity_status: str,
    frame_info: dict[str, Any],
    marker_scale_info: dict[str, Any],
    tape_checks_world: list[dict[str, Any]],
    T_yard_from_world: np.ndarray,
    walk_length_m: float,
    limited_tracking_share: float,
) -> dict[str, Any]:
    """Compute grade.json (tolerances_95pct, tape_checks, marker_scale, caps_applied, reasons) (YS-41..43)."""
    R_yw = T_yard_from_world[:3, :3]
    t_yw = T_yard_from_world[:3, 3]

    recomputed_tapes = [recompute_tape_check(tc) for tc in tape_checks_world]
    primary_tapes = [tc for tc in recomputed_tapes if tc.get("role", "primary") == "primary"]
    max_primary_residual_m = max((float(tc["residual_m"]) for tc in primary_tapes), default=0.0)

    scale_error_ratio = float(marker_scale_info.get("scale_error_ratio", 0.0))
    scale_error_se = float(marker_scale_info.get("scale_error_se", 0.0))
    scale_error_eff = float(marker_scale_info.get("scale_error_eff", max(0.0, abs(scale_error_ratio) - 2.0 * scale_error_se)))

    # Marker 0 loop-closure gap (track whether Marker 0 actually closed the loop before applying jump_sum)
    raw_m0_gap: float | None = None
    for gap_entry in frame_info.get("loop_closure_gaps", []):
        if int(gap_entry["marker_id"]) == 0:
            raw_m0_gap = float(gap_entry["gap_m"])
            break
    has_m0_loop_closure = raw_m0_gap is not None

    pose_jumps = frame_info.get("pose_jumps", [])
    from yardstick.geometry import counted_pose_jump_sum

    jump_sum = counted_pose_jump_sum(pose_jumps, frame_info.get("pose_jump_window_t_s"))
    m0_gap: float | None = raw_m0_gap
    if m0_gap is not None:
        m0_gap = max(m0_gap, jump_sum)
    elif jump_sum > 0.0:
        m0_gap = jump_sum

    eff_walk_len = max(float(walk_length_m), 10.0)
    # If Marker 0 loop closure is missing, assume a conservative 0.8% open-loop drift rate for tol(r)
    loop_gap_per_m = (m0_gap / eff_walk_len) if m0_gap is not None else 0.008

    sigma_tracking_m = SIGMA_TRACKING_BASE_M + SIGMA_TRACKING_SLOPE_M * float(limited_tracking_share)

    primary_floor_10m = 0.0
    for tc in primary_tapes:
        res_m = float(tc["residual_beyond_tape_m"])
        tape_m = max(float(tc["tape_m"]), 2.0)
        primary_floor_10m = max(primary_floor_10m, res_m * max(1.0, REFERENCE_RANGE_M / tape_m))

    dist_tol_10m = compute_tolerance_at_range(
        r_m=REFERENCE_RANGE_M,
        primary_tape_checks=primary_tapes,
        scale_error_eff=scale_error_eff,
        loop_gap_per_m=loop_gap_per_m,
        sigma_tracking_m=sigma_tracking_m,
    )
    pos_tol_10m = dist_tol_10m / math.sqrt(2.0)
    ground_res_m = float(frame_info.get("ground_plane_residual_m", 0.0))
    ground_res_p95_m = float(
        frame_info.get("ground_plane_residual_p95_m", ground_res_m * 1.96)
    )
    height_tol_m = max(
        ground_res_p95_m,
        ground_res_m * 1.96,
        1.96 * math.sqrt(SIGMA_LIDAR_M * SIGMA_LIDAR_M + (scale_error_eff * 3.0) ** 2),
    )

    # Transform tape check points to yard frame and evaluate each against tol(tape_m)
    tape_checks_yard: list[dict[str, Any]] = []
    phone_mismatches: list[str] = []
    for tc in recomputed_tapes:
        pa_w = np.array(tc["point_a_world_m"], dtype=np.float64)
        pb_w = np.array(tc["point_b_world_m"], dtype=np.float64)
        pa_y = R_yw @ pa_w + t_yw
        pb_y = R_yw @ pb_w + t_yw
        tape_m = float(tc["tape_m"])
        measured_m = float(tc["measured_m"])
        residual_m = float(tc["residual_m"])
        if tc.get("phone_mismatch", False):
            phone_mismatches.append(
                f"tape_check_id={tc['id']}: phone wrote measured={tc['phone_measured_m']:.4f}m/res={tc['phone_residual_m']:.4f}m "
                f"vs recomputed measured={measured_m:.4f}m/res={residual_m:.4f}m"
            )
        tol_at_tape = compute_tolerance_at_range(
            r_m=tape_m,
            primary_tape_checks=primary_tapes,
            scale_error_eff=scale_error_eff,
            loop_gap_per_m=loop_gap_per_m,
            sigma_tracking_m=sigma_tracking_m,
        )
        tape_checks_yard.append(
            {
                "id": int(tc["id"]),
                "role": str(tc.get("role", "primary")),
                "point_a_yard_m": [round(float(x), 6) for x in pa_y],
                "point_b_yard_m": [round(float(x), 6) for x in pb_y],
                "tape_m": round(tape_m, 6),
                "measured_m": round(measured_m, 6),
                "residual_m": round(residual_m, 6),
                "tape_uncertainty_m": round(float(tc["tape_uncertainty_m"]), 6),
                "residual_beyond_tape_m": round(float(tc["residual_beyond_tape_m"]), 6),
                "tolerance_at_tape_m": round(tol_at_tape, 6),
                "within_tolerance": bool(float(tc["residual_beyond_tape_m"]) <= tol_at_tape + 1e-9),
            }
        )

    # Determine raw band from distance_m and m0_gap (YS-42)
    # Band order: A=3, B=2, C=1, fail=0
    band_names = {3: "A", 2: "B", 1: "C", 0: "fail"}
    reasons: list[str] = []
    caps_applied: list[str] = []

    reasons.append(f"distance_tolerance_10m_m={dist_tol_10m:.4f}m")
    if has_m0_loop_closure and m0_gap is not None:
        reasons.append(f"loop_closure_gap_m={m0_gap:.4f}m")
    else:
        reasons.append("loop_closure_gap_m=null (Marker 0 not revisited at end of walk)")

    if primary_tapes:
        reasons.append(f"max_primary_tape_residual_m={max_primary_residual_m:.4f}m")
        for tc in primary_tapes:
            if float(tc["tape_uncertainty_m"]) > 0.0:
                reasons.append(
                    f"tape_uncertainty_m={float(tc['tape_uncertainty_m']):.3f}m -> residual_beyond_tape_m="
                    f"{float(tc['residual_beyond_tape_m']):.4f}m (id={tc['id']})"
                )
    else:
        reasons.append("primary_tape_checks=0")

    for mm_msg in phone_mismatches:
        reasons.append(f"phone_tape_residual_mismatch: {mm_msg}")

    reasons.append(
        f"marker_scale_error_ratio={scale_error_ratio:.4f} (se={scale_error_se:.4f}, eff={scale_error_eff:.4f})"
    )

    if pose_jumps:
        for pj in pose_jumps:
            reasons.append(
                f"pose_jump_detected: {pj['jump_m']:.4f}m at t={pj['t_s']:.3f}s "
                f"(frames {pj['saved_index_from']}->{pj['saved_index_to']})"
            )

    gap_for_band = m0_gap if m0_gap is not None else 0.0
    if dist_tol_10m <= 0.050 and gap_for_band <= 0.050:
        band = 3  # A
    elif dist_tol_10m <= 0.100 and gap_for_band <= 0.120:
        band = 2  # B
    elif dist_tol_10m <= 0.200 and gap_for_band <= 0.250:
        band = 1  # C
    else:
        band = 0  # fail
        if dist_tol_10m > 0.200:
            reasons.append(f"fail_threshold: distance_tolerance_10m_m ({dist_tol_10m:.4f}m) > 0.20m")
        if gap_for_band > 0.250:
            reasons.append(f"fail_threshold: loop_closure_gap_m ({gap_for_band:.4f}m) > 0.25m")

    # Apply caps (YS-31, YS-32, YS-42, YS-11)
    if integrity_status in ("partial", "invalid"):
        band = 0
        caps_applied.append(f"integrity_{integrity_status}")
        reasons.append(f"cap_applied: integrity={integrity_status} -> fail")

    if frame_info.get("method") != "marker":
        if band > 1:
            band = 1
        caps_applied.append("no_marker_0")
        reasons.append("cap_applied: no_marker_0 -> max grade C")
    elif not frame_info.get("marker_0_normal_ok", True):
        if band > 1:
            band = 1
        caps_applied.append("marker_0_tilted")
        reasons.append("cap_applied: marker_0_tilted (>30 deg from ground up) -> max grade C")

    if not has_m0_loop_closure and frame_info.get("method") == "marker":
        if band > 1:
            band = 1
        caps_applied.append("no_loop_closure")
        reasons.append("cap_applied: no_loop_closure (Marker 0 not revisited at end) -> max grade C")

    if len(primary_tapes) == 0:
        if band > 1:
            band = 1
        caps_applied.append("no_tape_check")
        reasons.append("cap_applied: no_tape_check -> max grade C")

    # Held-out tape checks (YS-42): 1 failing held-out pair caps at C; >= 2 failing held-out pairs grade fail.
    failed_held_out = [
        tc for tc in tape_checks_yard if tc.get("role") == "held_out" and not tc["within_tolerance"]
    ]
    if len(failed_held_out) >= 2:
        band = 0
        caps_applied.append("held_out_tape_failed_2plus")
        reasons.append(
            f"cap_applied: held_out_tape_failed_2plus ({len(failed_held_out)} held-out tape checks outside tolerance) -> fail"
        )
    elif len(failed_held_out) == 1:
        if band > 1:
            band = 1
        caps_applied.append("held_out_tape_failed_1")
        f_tc = failed_held_out[0]
        reasons.append(
            f"cap_applied: held_out_tape_failed_1 (id={f_tc['id']} residual={f_tc['residual_m']:.4f}m > tol={f_tc['tolerance_at_tape_m']:.4f}m) -> max grade C"
        )

    if limited_tracking_share > 0.10:
        if band > 2:
            band = 2
        caps_applied.append("limited_tracking_over_10pct")
        reasons.append(
            f"cap_applied: limited_tracking_over_10pct ({limited_tracking_share * 100.0:.1f}%) -> max grade B"
        )

    if integrity_status == "warnings" and band > 2:
        band = 2
        caps_applied.append("integrity_warnings")
        reasons.append("cap_applied: integrity=warnings -> max grade B")

    evidence_basis = ["lidar_noise_floor", "tracking_quality"]
    if primary_tapes:
        evidence_basis.insert(0, "tape_checks")
    if m0_gap is not None:
        evidence_basis.append("loop_closure_gaps")
    if int(marker_scale_info.get("sightings_used", 0)) > 0:
        evidence_basis.append("marker_scale_error")

    return {
        "contract": CONTRACT_VERSION,
        "grading_version": GRADING_VERSION,
        "grade": band_names[band],
        "tolerances_95pct": {
            "distance_m": round(dist_tol_10m, 6),
            "height_m": round(height_tol_m, 6),
            "position_m": round(pos_tol_10m, 6),
            "reference_range_m": REFERENCE_RANGE_M,
            "formula": FORMULA_STRING,
            "coefficients": {
                "max_tape_residual_m": round(max_primary_residual_m, 6),
                "primary_tape_floor_10m_m": round(primary_floor_10m, 6),
                "sigma_lidar_m": SIGMA_LIDAR_M,
                "scale_error_ratio": round(scale_error_ratio, 6),
                "scale_error_se": round(scale_error_se, 6),
                "scale_error_eff": round(scale_error_eff, 6),
                "loop_closure_gap_m": round(m0_gap, 6) if m0_gap is not None else None,
                "walk_length_m": round(eff_walk_len, 4),
                "loop_gap_per_m": round(loop_gap_per_m, 8),
                "sigma_tracking_m": round(sigma_tracking_m, 6),
                "limited_tracking_share": round(limited_tracking_share, 6),
            },
            "evidence_basis": evidence_basis,
        },
        "tape_checks": tape_checks_yard,
        "marker_scale": marker_scale_info,
        "caps_applied": caps_applied,
        "reasons": reasons,
    }

