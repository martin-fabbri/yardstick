<!--
Copyright 2026 Martin Fabbri

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Yardstick — Build Plan (`v1`)

Yardstick is an open-source iPhone LiDAR capture app and laptop processing toolkit built in public as a standalone BuildNexus project under the Apache License 2.0. It records an outdoor yard walk tied to printed ArUco markers, checks accuracy against a physical measuring tape, and produces a **standard capture** (`yardstick-capture/1`) graded for measurement trust.

LotFit, the BuildNexus hackathon entry built with Muse Code, is one consumer of Yardstick's published `yardstick-capture/1` contract.

---

## 1. Finish Line & Schedule

**Target (`YS-62`):** Monday, 2026-11-02 at 12:00 PT.

A real yard walk recorded on an iPhone 17 Pro with Yardstick is:
1. Shared from the phone as `<capture-id>.zip` and unpacked on the laptop within the `YS-22` floor (`<= 3 min` for a 3-minute walk);
2. Validated `clean` by `yardstick validate` against the published JSON Schemas in `contract/` (`YS-20`, `YS-21`, `YS-23`);
3. Calibrated and graded `B` or better with explicit evidence and reasons by `yardstick grade` (`YS-41`, `YS-42`, `YS-43`);
4. Converted into `<capture-id>/standard/` in the Marker-0 yard frame with `z = 0` ground plane and scanned-ground coverage polygon (`YS-50`–`YS-53`);
5. Loaded cleanly by LotFit's contract loader.

---

## 2. Priority Order & Scope

The 43 requirements in Factory (`/req/yardstick`, change sets `CS001` and `CS002`) are built strictly in priority rank order:

### Stage `v1` — P0 (31 items, built first)
1. **`YS-60`, `YS-61`, `YS-63` (Area 6 — Open, Clean-Room & Source Distribution):**
   - Public Apache-2.0 repository (`github.com/martin-fabbri/yardstick`), Apache-2.0 header on every source file, full dependency licence table, clean-room declaration, and step-by-step README instructions for building from source and installing onto an iPhone via USB cable with a free personal Apple ID (including the 7-day signing expiry).
2. **`YS-20`, `YS-21`, `YS-23` (Area 2 — Contract Schemas, Manifest Sealing & Validator):**
   - Authoritative JSON Schema draft-2020-12 files in `contract/` (`session.schema.json`, `manifest.schema.json`, `capture.schema.json`, `frame.schema.json`, `keyframes.schema.json`, `trajectory.schema.json`, `coverage.schema.json`, `events.schema.json`, `grade.schema.json`).
   - Deterministic validator (`yardstick validate`) checking schema compliance, CSV column headers and monotonic timestamps, HEVC frame count and variable-framerate duration, PNG bit-depths, `manifest.json` SHA-256 hashes and bidirectional file presence, and `standard/` non-interpretation rules (`YS-53`).
3. **`YS-31`, `YS-50`, `YS-51`, `YS-52`, `YS-53` (Areas 3 & 5 — Yard Frame & Standard Capture Converter):**
   - RANSAC ground-plane fitting (`z = 0`, up normal `+Z`), Marker 0 origin projection (`origin_height_m` and `origin_residual_m`), `+X` axis projection, point cloud filtering (`confidence >= 1`, `0.2 m <= range_m <= 4.5 m`) and `2.5 cm` voxel downsampling (`cloud.ply`), keyframes (`keyframes.json` + JPEGs), `trajectory.json`, and `20 cm` grid / `10 cm` Douglas-Peucker CCW simple polygon (`coverage.json`).
4. **`YS-32`, `YS-40`, `YS-41`, `YS-42`, `YS-43` (Areas 3 & 4 — Drift, Tape Check, Tolerance & Grading):**
   - Loop-closure gap per marker seen twice (`>= 2` distinct sighting clusters), closed-form range-dependent 95% tolerance `tol(r)` evaluated at `r = 10 m`, outdoor `grading-v1` thresholds (`A <= 5/5 cm`, `B <= 10/12 cm`, `C <= 20/25 cm`, caps for missing Marker 0 or missing tape check), and byte-identical geometry invariance with/without tape checks (`YS-43`).
5. **`YS-00`–`YS-11`, `YS-22`, `YS-30` (Areas 1, 2 & 3 — iPhone Capture App Core):**
   - iPhone 17 Pro / LiDAR hardware gate (`YS-00`);
   - Monotonic camera clock (`t = 0` at first frame) paired with start/stop UTC samples (`YS-01`);
   - Complete frame accounting (`frames.csv`, `saved_index 0..n`, drop reasons, tracking state/reason, `<= 5%` drop floor) (`YS-02`, `YS-11`);
   - Fixed highest-res 4:3 `@ 30 fps` video format selection and requested-vs-actual check (`YS-03`);
   - VFR HEVC `rgb.mp4`, background fast-compressed 16-bit mm `depth/NNNNNN.png` and 8-bit `confidence/NNNNNN.png`, native ARKit `poses.csv` (`YS-04`, `YS-05`);
   - 16 kHz mono PCM `audio.wav` on the shared clock (`YS-06`);
   - Tap event buttons (`marker`, `tape_a`, `tape_b`, `done`, `note`) (`YS-07`);
   - Lifecycle state machine (`starting` -> `recording` -> `stopped`), 5 s heartbeat, interruption/background safe flush, idle timer disable, and disk-write failure handling (`YS-08`, `YS-09`, `YS-10`);
   - Live 5–10 Hz ArUco `DICT_5X5_250` (`IDs 0–5`) detector in Swift using bit patterns generated from OpenCV's published dictionary (with attribution), pre-walk marker size config (`250 mm` default, `180 mm` option), and LiDAR-refined marker poses (`YS-30`);
   - Store-mode `.zip` archive creation and AirDrop share sheet (`YS-22`).

### Stage `v1` — P1 (6 items, built immediately after P0)
- **`YS-13`:** Sharp stills (`stills/NNNNNN.jpg` + `stills.csv`) extracted from uncompressed `ARFrame.capturedImage` (lowest angular speed per 2 s window + every tap event + sharpest sighting per marker pass) without adding frame drops.
- **`YS-14`:** Single location fix (`latitude`, `longitude`, `horizontal_accuracy_m`) and north calibration at Marker 0 (`>= 10` pairs of `(camera_yaw_rad, true_heading_deg, heading_accuracy_deg)` yielding the yard frame's `north_angle_deg` and `north_uncertainty_deg` in `frame.json`, or explicit `"denied"` / `"accuracy_exceeded"` status).
- **`YS-33`:** Independent marker scale check comparing printed marker size (`size_mm_printed`) against the mean of the 4 outer edges projected onto the LiDAR depth plane (`size_mm_measured`, 10% inward border, fused across every frame of a close pass at `0.6–2.0 m`), reporting `scale_error_ratio` and its standard error `scale_error_se` across frames, with `scale_error_eff = max(0, |scale_error| - 2 * SE)` feeding `YS-41`.
- **`YS-44`:** On-phone provisional grade shown at the end of the walk from loop-closure gap, primary tape check residual, and tracking quality.
- **`YS-12` (if time permits after `YS-44`, currently `next`):** Live pitch/roll and tracking guide.
- **`YS-45` (currently `next`):** Published cross-device grading specification.

### Stage `next` & `later` (Deferred past Monday 12:00 PT)
- `YS-12` (P1, `next`), `YS-45` (P1, `next`), `YS-15` (P2, `next` — 1 Hz `health.csv`), `YS-19` (P2, `next` — 12MP+ tap photos), `YS-16` (P2, `later` — live voice stamps), `YS-17` (P2, `later` — multi-pass `ARWorldMap`), `YS-18` (P2, `later` — `ARMeshAnchor` export), `YS-54` (P2, `later` — multi-device adapters).

---

## 3. Stack & Open-Source Dependencies

Every source file in `yardstick` carries the standard Apache-2.0 SPDX/copyright header. All third-party dependencies and their licences are documented in `README.md`:

### iOS App (`ios/Yardstick/`)
- **Language & SDK:** Swift 6, SwiftUI, iOS 18+ / latest iOS SDK, Xcode 16+.
- **Apple Frameworks (Platform SDK):** `ARKit`, `AVFoundation`, `CoreLocation`, `Vision`, `Accelerate` (`simd`), `CryptoKit`, `Compression` / `Foundation`.
- **Generated Data:** ArUco `DICT_5X5_250` bit patterns (`IDs 0–5`) generated via `toolkit/scripts/generate_aruco_dict.py` from OpenCV's `cv2.aruco.DICT_5X5_250` (Apache License 2.0, attributed in source and `README.md`).
- **Third-Party iOS Libraries:** None (zero external Swift package dependencies).

### Laptop Toolkit (`toolkit/`)
- **Runtime:** Python 3.12+ managed via `uv`.
- **Dependencies:**
  - `numpy` — BSD-3-Clause
  - `scipy` — BSD-3-Clause
  - `opencv-contrib-python-headless` — Apache-2.0
  - `Pillow` — MIT-CMU
  - `jsonschema` — MIT
  - `pydantic` — MIT
  - `pytest` — MIT (development/test)

---

## 4. Checkpoints & Workflow

- **Git Workflow:** Branch per change (`feat/...`, `docs/...`), commits authored as `martin-fabbri@users.noreply.github.com`, squash-merged to `main` on Martin's approval.
- **Checkpoint `C001a` (Schemas + Laptop Toolkit + Synthetic Yard Suite):**
  - `contract/*.schema.json` frozen;
  - `yardstick synth`, `yardstick validate`, `yardstick convert`, `yardstick grade` implemented and passing 100% of unit and planted-fault tests (`YS-20`, `YS-21`, `YS-23`, `YS-31`–`YS-33`, `YS-41`–`YS-43`, `YS-50`–`YS-53`).
- **Checkpoint `C001b` (iOS App + Cable Install Guide):**
  - Xcode project in `ios/` building cleanly with a free personal Apple team (`YS-00`–`YS-11`, `YS-13`, `YS-14`, `YS-22`, `YS-30`, `YS-32`, `YS-40`, `YS-44`, `YS-63`).
- **Checkpoint `C001c` (Calibration & Demo Walk — `YS-62`):**
  - Outdoor walk with 4 tape pairs (1 primary + 3 held-out) and Markers `0–3` on tabloid foam board; validated `clean`, graded `B` or better, converted to `standard/`, and read by LotFit's loader.

