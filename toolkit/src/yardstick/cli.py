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

"""Command-line interface for the Yardstick laptop toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yardstick.converter import convert_recording_to_standard
from yardstick.synth import (
    SynthOptions,
    generate_synthetic_capture,
    pack_capture_zip,
    unpack_capture_zip,
)
from yardstick.validator import validate_recording, validate_standard_capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yardstick",
        description="Yardstick LiDAR capture validator, converter, grader, and synthetic generator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a Level 1 recording or Level 2 standard capture.")
    p_val.add_argument("path", type=Path, help="Path to capture folder or standard/ folder.")
    p_val.add_argument("--standard", action="store_true", help="Validate standard/ capture as well.")

    p_conv = sub.add_parser("convert", help="Convert a Level 1 recording into standard/ capture.")
    p_conv.add_argument("path", type=Path, help="Path to Level 1 capture folder.")
    p_conv.add_argument("--voxel-size", type=float, default=0.025, help="Voxel size in meters (default 0.025).")

    p_grade = sub.add_parser("grade", help="Print the grade and tolerance report for a capture.")
    p_grade.add_argument("path", type=Path, help="Path to capture folder.")

    p_synth = sub.add_parser("synth", help="Generate a deterministic synthetic Level 1 capture.")
    p_synth.add_argument("--output", "-o", type=Path, required=True, help="Output capture directory.")
    p_synth.add_argument("--convert", action="store_true", help="Also convert to standard/ after generating.")

    p_unpack = sub.add_parser("unpack", help="Unpack a capture .zip archive and verify hashes.")
    p_unpack.add_argument("zip_path", type=Path, help="Path to capture .zip file.")
    p_unpack.add_argument("--output", "-o", type=Path, required=True, help="Directory to extract into.")

    p_pack = sub.add_parser("pack", help="Pack a capture directory into a store-mode .zip archive.")
    p_pack.add_argument("path", type=Path, help="Path to capture directory.")
    p_pack.add_argument("--output", "-o", type=Path, required=True, help="Output .zip file path.")

    args = parser.parse_args(argv)

    if args.command == "validate":
        target: Path = args.path.resolve()
        if (target / "capture.json").exists() and not (target / "session.json").exists():
            rep = validate_standard_capture(target)
            print(json.dumps(rep, indent=2))
            return 0 if rep["status"] == "clean" else 1

        rep1 = validate_recording(target)
        out = {"recording": rep1}
        code = 0 if rep1["status"] == "clean" else 1
        if args.standard or (target / "standard" / "capture.json").exists():
            rep2 = validate_standard_capture(target / "standard")
            out["standard"] = rep2
            if rep2["status"] != "clean":
                code = 1
        print(json.dumps(out, indent=2))
        return code

    if args.command == "convert":
        target = args.path.resolve()
        result = convert_recording_to_standard(target, voxel_size_m=args.voxel_size)
        rep2 = validate_standard_capture(target / "standard")
        print(
            json.dumps(
                {
                    "capture_id": result["capture"]["capture_id"],
                    "grade": result["grade"]["grade"],
                    "reasons": result["grade"]["reasons"],
                    "standard_validation": rep2,
                },
                indent=2,
            )
        )
        return 0 if rep2["status"] == "clean" else 1

    if args.command == "grade":
        target = args.path.resolve()
        grade_path = (
            target / "standard" / "grade.json"
            if (target / "standard" / "grade.json").exists()
            else target / "grade.json"
        )
        if not grade_path.exists():
            convert_recording_to_standard(target)
            grade_path = target / "standard" / "grade.json"
        grade_data = json.loads(grade_path.read_text(encoding="utf-8"))
        print(json.dumps(grade_data, indent=2))
        return 0

    if args.command == "synth":
        out_dir = generate_synthetic_capture(args.output, SynthOptions())
        if args.convert:
            convert_recording_to_standard(out_dir)
        print(json.dumps({"status": "ok", "output": str(out_dir)}, indent=2))
        return 0

    if args.command == "unpack":
        extracted = unpack_capture_zip(args.zip_path, args.output)
        rep = validate_recording(extracted)
        print(json.dumps({"extracted_to": str(extracted), "validation": rep}, indent=2))
        return 0 if rep["status"] == "clean" else 1

    if args.command == "pack":
        zpath = pack_capture_zip(args.path, args.output)
        print(json.dumps({"zip_path": str(zpath)}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
