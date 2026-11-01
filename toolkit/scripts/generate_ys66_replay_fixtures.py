#!/usr/bin/env python3
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

"""Generate YS-66 golden toolkit verdicts and the trimmed Walk-5 sample fixture.

Command:
    cd toolkit && YARDSTICK_REPLAY_ROOT=/path/to/2026-10-31 PYTHONPATH=src .venv/bin/python scripts/generate_ys66_replay_fixtures.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from yardstick import CONTRACT_VERSION, GRADING_VERSION
from yardstick.converter import convert_recording_to_standard

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "ios" / "Yardstick" / "Tests" / "Fixtures"
GOLDEN_PATH = FIXTURES_DIR / "toolkit_golden_walks_1_5.json"
TRIMMED_WALK5_DIR = FIXTURES_DIR / "walk5_trimmed_sample"

WALK_SPECS: list[tuple[str, str, str]] = [
    ("walk-1", "walks-1-4/20261101T003322Z_yardstick", "20261101T003322Z_yardstick"),
    ("walk-2", "walks-1-4/20261101T003455Z_yardstick", "20261101T003455Z_yardstick"),
    ("walk-3", "walks-1-4/20261101T003609Z_yardstick", "20261101T003609Z_yardstick"),
    ("walk-4", "walks-1-4/20261101T003631Z_yardstick", "20261101T003631Z_yardstick"),
    ("walk-5", "walk-5/20261101T012134Z_yardstick", "20261101T012134Z_yardstick"),
]

# Representative saved_index frames from Walk 5:
# - 0 (t=712.073208): opening Marker 0 sighting
# - 381 (t=724.773532): Marker 0 median size sighting (172.474 mm)
# - 438 (t=726.673596, dt=14.60 s): final Marker 0 sighting that triggers the phone's false 0.075326 m loop close
# - 1020 (t=746.074101): Marker 1 median size sighting (177.788 mm)
# - 1929 (t=776.374903): Marker 2 median size sighting (176.080 mm)
# - 2652 (t=800.475516): Marker 3 median size sighting (175.891 mm)
TRIMMED_SAVED_INDICES: list[int] = [0, 381, 438, 1020, 1929, 2652]


def _extract_walk_summary(walk_key: str, rel_path: str, capture_dir: Path) -> dict[str, Any]:
    std_dir = capture_dir / "standard"
    if not (std_dir / "frame.json").exists() or not (std_dir / "grade.json").exists():
        convert_recording_to_standard(capture_dir)

    session_data = json.loads((capture_dir / "session.json").read_text(encoding="utf-8"))
    frame_data = json.loads((std_dir / "frame.json").read_text(encoding="utf-8"))
    grade_data = json.loads((std_dir / "grade.json").read_text(encoding="utf-8"))
    coeff = grade_data["tolerances_95pct"]["coefficients"]
    coeff_gap_m = coeff["loop_closure_gap_m"]
    frame_gaps = frame_data.get("loop_closure_gaps", [])
    m0_frame_gap_m = next(
        (float(g["gap_m"]) for g in frame_gaps if int(g["marker_id"]) == 0),
        None,
    )
    caps_applied = grade_data.get("caps_applied", [])
    has_no_loop_cap = "no_loop_closure" in caps_applied or "no_marker_0" in caps_applied

    per_marker = {}
    for entry in grade_data.get("marker_scale", {}).get("per_marker", []):
        per_marker[str(entry["marker_id"])] = {
            "sightings_count": int(entry["sightings_count"]),
            "size_mm_printed": float(entry["size_mm_printed"]),
            "size_mm_measured": float(entry["size_mm_measured"]),
            "scale_error_ratio": float(entry["scale_error_ratio"]),
            "scale_error_se": float(entry["scale_error_se"]),
        }

    return {
        "walk_key": walk_key,
        "relative_path": rel_path,
        "capture_id": str(session_data["capture_id"]),
        "grade": str(grade_data["grade"]),
        "caps_applied": list(caps_applied),
        "grade_coeff_loop_closure_gap_m": float(coeff_gap_m) if coeff_gap_m is not None else None,
        "grade_coeff_reports_closed": coeff_gap_m is not None,
        "frame_marker0_loop_closure_gap_m": m0_frame_gap_m,
        "frame_marker0_loop_closed": (m0_frame_gap_m is not None) and (not has_no_loop_cap),
        "pose_jumps": frame_data.get("pose_jumps", []),
        "reasons": grade_data.get("reasons", []),
        "per_marker": per_marker,
    }


def _build_trimmed_walk5_sample(walk5_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "stills").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)

    keep_set = {str(idx) for idx in TRIMMED_SAVED_INDICES}

    shutil.copy2(walk5_dir / "session.json", out_dir / "session.json")
    shutil.copy2(walk5_dir / "events.csv", out_dir / "events.csv")

    # frames.csv: keep the 9 startup dropped rows + the 6 selected saved_index rows
    with (walk5_dir / "frames.csv").open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        kept_frames = [
            row
            for row in reader
            if row.get("dropped", "").strip().lower() == "true" or row.get("saved_index", "").strip() in keep_set
        ]
    with (out_dir / "frames.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept_frames)

    for csv_name in ("poses.csv", "stills.csv", "markers.csv"):
        with (walk5_dir / csv_name).open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            c_fields = reader.fieldnames or []
            kept_rows = [row for row in reader if row.get("saved_index", "").strip() in keep_set]
        with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=c_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept_rows)

    for idx in TRIMMED_SAVED_INDICES:
        shutil.copy2(walk5_dir / "stills" / f"{idx:06d}.jpg", out_dir / "stills" / f"{idx:06d}.jpg")
        shutil.copy2(walk5_dir / "depth" / f"{idx:06d}.png", out_dir / "depth" / f"{idx:06d}.png")
        shutil.copy2(walk5_dir / "confidence" / f"{idx:06d}.png", out_dir / "confidence" / f"{idx:06d}.png")


def main() -> int:
    replay_root_env = os.environ.get("YARDSTICK_REPLAY_ROOT", "").strip()
    if not replay_root_env:
        print("ERROR: Set YARDSTICK_REPLAY_ROOT to the directory containing walks-1-4/ and walk-5/.", file=sys.stderr)
        return 1

    replay_root = Path(replay_root_env).resolve()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    walks_out: dict[str, Any] = {}
    for walk_key, rel_path, _ in WALK_SPECS:
        cap_dir = replay_root / rel_path
        if not cap_dir.exists():
            print(f"ERROR: Missing capture directory {cap_dir}", file=sys.stderr)
            return 1
        walks_out[walk_key] = _extract_walk_summary(walk_key, rel_path, cap_dir)

    walk5_dir = replay_root / "walk-5/20261101T012134Z_yardstick"
    _build_trimmed_walk5_sample(walk5_dir, TRIMMED_WALK5_DIR)

    golden_doc = {
        "schema": "yardstick.ys66.toolkit_golden.v1",
        "contract_version": CONTRACT_VERSION,
        "grading_version": GRADING_VERSION,
        "generated_by_command": (
            "cd toolkit && YARDSTICK_REPLAY_ROOT=$YARDSTICK_REPLAY_ROOT "
            "PYTHONPATH=src .venv/bin/python scripts/generate_ys66_replay_fixtures.py"
        ),
        "trimmed_walk5_saved_indices": TRIMMED_SAVED_INDICES,
        "walks": walks_out,
    }
    GOLDEN_PATH.write_text(json.dumps(golden_doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}")
    print(f"Wrote trimmed Walk 5 sample to {TRIMMED_WALK5_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
