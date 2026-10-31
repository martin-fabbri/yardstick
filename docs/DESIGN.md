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

# Yardstick — Architecture & Contract Design (`yardstick-capture/1`)

This document specifies the clean-room architecture, coordinate systems, algorithms, and exact data schemas (`yardstick-capture/1`) for the Yardstick iPhone app and laptop toolkit.

---

## 1. System Overview

Yardstick is divided into two decoupled levels connected by an immutable, hash-sealed archive:

1. **Level 1 — The Recording (`<capture-id>/`, written on the iPhone):**
   - Records raw sensor measurements on a single monotonic clock (`t = 0.0` at first delivered camera frame) without interpretation or silent coordinate flips.
   - Detects printed ArUco markers (`DICT_5X5_250`, IDs `0–5`) live at 5–10 Hz, logs interactive raycast tape checks, computes a live loop-closure gap and provisional grade before leaving the yard, seals all files in `manifest.json` with SHA-256 hashes, and shares `<capture-id>.zip` (ZIP store mode, compression level 0) via AirDrop.
2. **Level 2 — The Standard Capture (`<capture-id>/standard/`, built on the laptop):**
   - Verifies Level 1 integrity (`yardstick validate`), fits the yard ground plane (`z = 0`) and Marker 0 origin (`yardstick convert`), evaluates independent marker scale error and the closed-form 95% measurement tolerance (`yardstick grade`), and writes the standard geometry and metadata files consumed by LotFit.

---

## 2. Clocks, Coordinate Systems & Conventions

### 2.1 Shared Monotonic Clock & UTC Anchor (`YS-01`, `YS-14`)
- **Primary clock (`t`):** Every timed row in Level 1 and Level 2 uses `t` in seconds (float64) relative to the first delivered `ARFrame.timestamp` (`t0_monotonic_s`):
  $$t = \text{frame.timestamp} - t_{0,\text{monotonic}}$$
  Frame 0 is never dropped; if the ISP or `CVPixelBufferPool` drops frames upstream between consecutive `session(_:didUpdate:)` callbacks ($\Delta t > 1.5 / \text{fps}$), the recorder synthesizes `round(delta_t * fps) - 1` rows in `frames.csv` with `dropped = true` and `drop_reason = "upstream_isp_or_pool_drop"`.
- **Zero `ARFrame` Retention Rule:** `session(_:didUpdate:)` never passes an `ARFrame` or `CVPixelBuffer` reference across asynchronous queue boundaries (which would starve ARKit's ~10-buffer hardware pool within $0.35\text{ s}$). Depth (`Float32` m $\rightarrow$ `UInt16` mm) and confidence (`UInt8`) buffers are copied synchronously (`< 0.2 ms`) into a bounded ring buffer, and `YS-13` keeps only a single running-best raw frame copy per 2-second window whenever a new minimum angular speed is set.
- **Wall-clock mapping:** At recording start and stop, the app samples `(monotonic_s, utc_iso8601)` pairs (`start_clock_pair` and `stop_clock_pair` in `session.json`, plus `started_at_utc`). Any monotonic timestamp `t` maps to UTC via:
  $$\text{UTC}(t) = \text{start\_clock\_pair.utc} + (t - \text{start\_clock\_pair.monotonic\_s})$$
- **Audio sync (`YS-01`, `YS-06`):** `ARWorldTrackingConfiguration.providesAudioData = true` is enabled so ARKit delivers microphone buffers via `session(_:didOutputAudioSampleBuffer:)` whose `CMSampleBufferGetPresentationTimeStamp` is natively on the exact same `CMClockGetHostTimeClock()` domain as `ARFrame.timestamp`. Incoming audio is converted to 16,000 Hz, 1-channel, 16-bit signed PCM (`s16le`) via `AVAudioConverter` and streamed to `audio.wav` (updating the 44-byte WAV header on every 5 s heartbeat and on close). `session.json` records `audio_first_sample_t_s` and `audio_sample_rate_hz`.
- **Crash-Safe Recording (`YS-08`, `YS-21`, `Q022`):** `AVAssetWriter` uses `movieFragmentInterval = CMTime(seconds: 5.0, preferredTimescale: 600)` (fragmented MP4), CSV file handles flush every 1 s, and `session.json` is atomically rewritten on start and every 5 s heartbeat. Each PNG/JPEG image file's SHA-256 is computed in memory (`CryptoKit.SHA256`) as it is written, while `rgb.mp4` (along with `audio.wav`, CSVs, and `session.json`) is hashed at `Stop` from the finished file on disk after `AVAssetWriter.finishWriting()` completes.

### 2.2 Level 1 Native Pose & Unprojection Convention (`YS-04`, `YS-05`)
- `session.json` declares `"pose_convention": "arkit_right_handed_y_up"`.
- In ARKit's camera-to-world transform (`ARCamera.transform`, $4 \times 4$ column-major `simd_float4x4` in memory, serialized row-major or as translation + Hamilton quaternion):
  - **World frame:** Right-handed, $+Y$ up (aligned with gravity; true/magnetic heading is recorded via `CLLocationManager` in `session.json#location`).
  - **Camera local frame:** Camera $+X$ is right along the sensor width, $+Y$ is up along the sensor height, and $-Z$ is the forward optical viewing ray (camera looks down $-Z_{\text{cam}}$).
  - **Grip-Invariant Pitch Angle (`YS-05`, `YS-12`):** Phone tilt/pitch relative to the horizon is defined grip-invariantly from the optical viewing ray $\vec{v}_{\text{look}} = R_{\text{world} \leftarrow \text{cam}} [0, 0, -1]^T$ and the world up vector $\hat{y}_{\text{world}} = [0, 1, 0]^T$:
    $$\text{pitch\_deg} = \arcsin(\vec{v}_{\text{look}} \cdot \hat{y}_{\text{world}}) \times \frac{180^\circ}{\pi}$$
    so holding the phone $30^\circ$ down toward the ground yields $\text{pitch\_deg} = -30.0^\circ$ regardless of portrait vs. landscape grip.
  - **Intrinsics & Planar Depth Unprojection (`poses.csv`, `stills.csv`, `keyframes.json`, `YS-04`, `YS-50`):** `fx, fy, cx, cy, width, height` correspond to the saved color image dimensions $(W, H) = (1920, 1440)$. `depth/NNNNNN.png` stores planar optical-axis depth $z_d = -z_{\text{cam}}$ in uint16 millimetres ($z_{d,\text{m}} = \text{pixel\_uint16} / 1000.0$). When unprojecting a depth pixel $(u_d, v_d) \in [0, W_d - 1] \times [0, H_d - 1]$ (where $+u_d$ is right and $+v_d$ is down in image coordinates) with scaled intrinsics $f_{x,d} = f_x \frac{W_d}{W}$, $f_{y,d} = f_y \frac{H_d}{H}$, $c_{x,d} = c_x \frac{W_d}{W}$, $c_{y,d} = c_y \frac{H_d}{H}$:
    $$x_{\text{cam}} = \frac{(u_d - c_{x,d}) \cdot z_{d,\text{m}}}{f_{x,d}}, \quad y_{\text{cam}} = -\frac{(v_d - c_{y,d}) \cdot z_{d,\text{m}}}{f_{y,d}}, \quad z_{\text{cam}} = -z_{d,\text{m}}$$
    $$\text{range\_m} = \sqrt{x_{\text{cam}}^2 + y_{\text{cam}}^2 + z_{\text{cam}}^2}$$
- **Quaternions:** All quaternions are unit Hamilton quaternions representing the rotation from camera local axes (or marker local axes) to the target world frame, serialized in order `qx, qy, qz, qw` where $q = q_w + q_x i + q_y j + q_z k$.

### 2.3 Printed Marker Coordinate Frame (`YS-30`, `YS-31`, `MARKERS.md`)
- Dictionary: `DICT_5X5_250`. Single markers: IDs `0–5` (`250.0 mm` on tabloid or `180.0 mm` on letter). Optional ChArUco board: $6 \times 9$ squares, $40\text{ mm}$ squares, $30\text{ mm}$ markers, IDs `100–126` (detected on the laptop).
- **Marker local axes (`MARKERS.md` & `CAPTURE-CONTRACT.md`):**
  - Corners are ordered `(c0: top-left, c1: top-right, c2: bottom-right, c3: bottom-left)` looking at the printed face, where **top** is the edge away from the printed text label.
  - Origin $(0, 0, 0)$ is the center of the black outer square.
  - $+X_{\text{marker}}$ runs from top-left ($c_0$) to top-right ($c_1$).
  - $+Y_{\text{marker}}$ runs from bottom-left ($c_3$) to top-left ($c_0$).
  - $+Z_{\text{marker}}$ points **out of the printed face** (upward into the air when the marker lies flat on the ground), completing a right-handed frame ($+X \times +Y = +Z$).
  - For printed side length $s$ (in metres, $0.250$ or $0.180$), the 3D corners in the marker's local frame are:
    - $c_0 = (-s/2, +s/2, 0)$
    - $c_1 = (+s/2, +s/2, 0)$
    - $c_2 = (+s/2, -s/2, 0)$
    - $c_3 = (-s/2, -s/2, 0)$

### 2.4 Level 2 Yard Frame Construction (`YS-31`, `YS-51`)
All geometry in `<capture-id>/standard/` (`cloud.ply`, `keyframes.json`, `trajectory.json`, `coverage.json`, `grade.json`) is expressed in the **Yard Frame** (metres, right-handed, $+Z$ up, ground at $z = 0$).

The $4 \times 4$ rigid transform $T_{\text{yard} \leftarrow \text{world}}$ (stored row-major in `frame.json` as `world_to_yard_transform`) is computed deterministically as follows:

1. **Ground Plane Fit (`YS-51`):**
   - Collect candidate ground points in the recording's world frame from high/medium-confidence LiDAR depth samples whose height in gravity-aligned ARKit world coordinates ($y_{\text{world}}$) lies within $\pm 0.50\text{ m}$ of the lowest 15th percentile of camera heights minus $1.2\text{ m}$ (or within $\pm 0.40\text{ m}$ of Marker 0's height if Marker 0 was sighted).
   - Run deterministic RANSAC plane fitting (distance threshold $0.03\text{ m}$, fixed seed `42`, refit via SVD on inliers) subject to a **gravity-normal tilt gate** $|\hat{n}_{\text{ground}} \cdot \hat{y}_{\text{world}}| \ge \cos 15^\circ \approx 0.966$ (rejecting walls, fences, and vertical steps) to obtain unit normal $\hat{n}_{\text{ground}}$ (oriented so $\hat{n}_{\text{ground}} \cdot \hat{y}_{\text{world}} > 0$, i.e. pointing skyward) and plane offset $d$ such that $\hat{n}_{\text{ground}} \cdot p + d = 0$ on the ground plane.
   - The yard $+Z$ axis in world coordinates is $\hat{z}_{\text{yard}} = \hat{n}_{\text{ground}}$.
   - Report `ground_plane_residual_m` as the median $|\hat{n}_{\text{ground}} \cdot p + d|$ over all ground inliers (`YS-51` success criterion: $\le 0.02\text{ m}$ on the test yard).
2. **Origin & $+X / +Y$ Axes from Marker 0 (`YS-31`):**
   - **If Marker 0 was sighted:**
     - Group Marker 0 sightings into the **initial pass** (sightings within the first $30\text{ s}$ or within $10\text{ s}$ of the first Marker 0 sighting) and **return pass** (sightings separated by $\ge 3.0\text{ s}$ and $\ge 1.0\text{ m}$ of camera trajectory excursion from the initial pass).
     - Require Marker 0's normal $\hat{z}_{m0}$ to be within $30^\circ$ of $\hat{z}_{\text{yard}}$ ($\hat{z}_{m0} \cdot \hat{z}_{\text{yard}} \ge \cos 30^\circ$) so $+X_{\text{marker}}$ projection onto the ground plane is never degenerate.
     - Compute the initial Marker 0 center $p_{m0}$ (median of initial-pass positions) and initial Marker 0 $+X$ axis vector $v_{x,m0}$ (mean of initial-pass $+X_{\text{marker}}$ axes).
     - Signed height of Marker 0 above the fitted ground plane:
       $$\text{origin\_height\_m} = \hat{n}_{\text{ground}} \cdot p_{m0} + d$$
     - Project $p_{m0}$ orthogonally onto the ground plane to obtain the yard origin in world coordinates:
       $$o_{\text{yard}} = p_{m0} - \text{origin\_height\_m} \cdot \hat{n}_{\text{ground}}$$
     - Project $v_{x,m0}$ onto the ground plane and normalize to obtain $\hat{x}_{\text{yard}}$:
       $$\hat{x}_{\text{yard}} = \frac{v_{x,m0} - (v_{x,m0} \cdot \hat{z}_{\text{yard}})\hat{z}_{\text{yard}}}{\|v_{x,m0} - (v_{x,m0} \cdot \hat{z}_{\text{yard}})\hat{z}_{\text{yard}}\|}$$
     - Set $\hat{y}_{\text{yard}} = \hat{z}_{\text{yard}} \times \hat{x}_{\text{yard}}$.
     - Report `origin_residual_m` as the 3D Euclidean distance between the initial Marker 0 position and the return-pass Marker 0 position in the yard frame (`YS-31` success criterion: $\le 0.05\text{ m}$ on test walk).
   - **Fallback (`method: "first_pose"` when Marker 0 is absent):**
     - Project the first saved camera position $p_{\text{cam},0}$ onto the ground plane as $o_{\text{yard}}$, project the camera's horizontal forward ray onto the ground plane as $\hat{x}_{\text{yard}}$, set $\hat{y}_{\text{yard}} = \hat{z}_{\text{yard}} \times \hat{x}_{\text{yard}}$, and cap the capture grade at `C` (`YS-31`, `YS-42`).
3. **North Calibration at Marker 0 (`YS-14`, `Q020`):**
   - Because `ARWorldTrackingConfiguration` runs in **gravity-only** alignment (`YS-03`), true north is calibrated at the start of the walk while viewing Marker 0: the app records at least 10 samples of `(t_s, camera_yaw_world_deg, true_heading_deg, heading_accuracy_deg)` in `session.json#location.north_samples`.
   - From these samples and $T_{\text{yard} \leftarrow \text{world}}$, the converter computes the circular mean and circular standard error / uncertainty of True North's azimuth in the Yard Frame (`north_angle_deg`: counter-clockwise angle in degrees from $+X_{\text{yard}}$ to True North in the XY ground plane, and `north_uncertainty_deg`: $\sqrt{\text{SE}_{\text{samples}}^2 + \text{median}(\text{heading\_accuracy\_deg})^2}$), writing both into `frame.json`. If Location permission is `"denied"` or `heading_accuracy_deg > 25.0`, `north_status` is recorded explicitly as `"denied"` or `"accuracy_exceeded"` (never left blank).
4. **Camera Pose in Yard Frame (`keyframes.json`, `trajectory.json`):**
   - Each camera pose in `standard/` is a $4 \times 4$ row-major SE(3) matrix $T_{\text{yard} \leftarrow \text{cam}} = T_{\text{yard} \leftarrow \text{world}} \cdot T_{\text{world} \leftarrow \text{cam}}$ ($\|R^T R - I\|_{\max} \le 10^{-4}$, $|\det(R) - 1| \le 10^{-4}$ per `YS-23`), where the camera local frame matches ARKit's camera frame ($+X$ right, $+Y$ up, $-Z$ look direction) and `pose_convention` is recorded in `frame.json`.

---

## 3. Marker Detection, Scale Check & Loop Closure (`YS-30`, `YS-32`, `YS-33`)

### 3.1 Clean-Room Live Detector on iOS (`YS-30`)
- A build script (`toolkit/scripts/generate_aruco_dict.py`) queries OpenCV (`cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)`) and exports the 25-bit patterns (and their 4 $90^\circ$ rotations) for IDs `0–5` into a generated Swift constant file with explicit Apache-2.0 attribution to OpenCV.
- During recording, a background vision worker runs at 5–10 Hz on a copied luma/depth snapshot from the latest `ARFrame`:
  1. Detects candidate dark-on-light square quadrilaterals (`VNDetectContoursRequest` with `detectsDarkOnLight = true` or multi-quad contour approximation so the outer white foam board never masks the inner black marker);
  2. Applies a perspective homography warp to a $7 \times 7$ cell image ($56 \times 56$ px), samples the inner $4 \times 4$ pixels of each cell, verifies all 24 outer border cells are dark relative to the local inner-vs-border threshold, reads the inner $5 \times 5$ bits, and matches against IDs `0–5`;
  3. Refines the 4 corner locations to sub-pixel accuracy on the luma buffer, logs the 2D corner coordinates, and triggers a still frame (`YS-13`) on the sharpest sighting of each marker pass;
  4. Solves initial orientation $R_{\text{world} \leftarrow \text{marker}}$ and ray direction from the 4 corners and `size_mm_printed`, then refines 3D translation and plane normal using the LiDAR depth samples inside the marker quadrilateral (`depth/NNNNNN.png`, eroded inward by 10% to avoid foam-board edge bleeding).

### 3.2 Independent Scale Error from LiDAR (`YS-33`, `Q027`)
To measure marker size without circular dependence on the PnP size input and with honest statistical uncertainty across frames:
1. For each close, high-confidence sighting frame $f$ in a pass ($0.6\text{ m} \le \text{range\_m} \le 2.0\text{ m}$, viewing angle $\le 45^\circ$ from normal, $\ge 16$ interior depth pixels with `confidence >= 1` after 10% inward border erosion), unproject the interior LiDAR depth pixels into 3D camera coordinates and fit a 3D plane $\pi_{\text{marker}, f}$;
2. Intersect the 4 RGB corner rays $\vec{r}_{f,i} = K^{-1}[u_{f,i}, v_{f,i}, 1]^T$ with $\pi_{\text{marker}, f}$ to obtain 3D corner positions $P_{f,0}, P_{f,1}, P_{f,2}, P_{f,3}$ in metres, transform into world coordinates, and compute the frame's measured size $s_f = \frac{1000}{4} \sum_{i=0}^{3} \|P_{f,(i+1)\bmod 4} - P_{f,i}\|$ as well as the multi-frame fused 3D corners $\bar{P}_i$ across the pass:
   $$\text{size\_mm\_measured} = \frac{1000}{4} \sum_{i=0}^{3} \|\bar{P}_{(i+1)\bmod 4} - \bar{P}_i\|$$
3. Compute the relative scale error $e_{\text{scale}, f} = (s_f - \text{size\_mm\_printed}) / \text{size\_mm\_printed}$ for each sighting frame $f$, the fused scale error $\epsilon_{\text{scale}} = (\text{size\_mm\_measured} - \text{size\_mm\_printed}) / \text{size\_mm\_printed}$, and its **standard error** $\text{SE}_{\text{scale}} = \text{std}(e_{\text{scale}, f}) / \sqrt{N_{\text{frames}}}$ from the empirical spread across frames (`YS-33`).
4. In `YS-41`, the effective scale error entering `tol(r)` is:
   $$\epsilon_{\text{scale,eff}} = \max\left(0.0,\; |\epsilon_{\text{scale}}| - 2 \cdot \text{SE}_{\text{scale}}\right)$$
   so a real scale error (such as declaring 180 mm on a 250 mm marker) counts in full while zero-mean frame-to-frame depth noise within $2 \cdot \text{SE}$ does not inflate the 10 m tolerance.

### 3.3 Loop-Closure Drift Gap & Relocalization Snap Guard (`YS-32`)
- Sightings of the same `marker_id` are clustered in time: two consecutive sightings belong to the same pass if separated by $< 3.0\text{ s}$, and start a new pass if separated by $\ge 3.0\text{ s}$ and at least $1.0\text{ m}$ of camera trajectory excursion from the previous pass.
- To prevent ARKit visual relocalization from snapping `ARCamera.transform` right before the return sighting of Marker 0 and hiding accumulated mid-walk drift, the converter also scans consecutive saved poses in `poses.csv` for non-physical single-frame pose jumps $\Delta p_{\text{snap}, i}$ where $\|p_i - p_{i-1}\|_2 > 0.08\text{ m}$ within $\Delta t < 0.10\text{ s}$. Every detected jump is recorded in `frame.json#pose_jumps` and named in `grade.json#reasons`.
- For any marker seen in $\ge 2$ passes, its loop-closure gap `gap_m` is:
  $$\text{gap\_m}(id) = \max\left(\|\bar{p}_{\text{last}}(id) - \bar{p}_{\text{first}}(id)\|_2,\; \sum_{\text{snaps}} \|\Delta p_{\text{snap}}\|_2\right)$$
- If Marker 0 is seen only in a single pass (walker never looped back to Marker 0 at the end of the walk), `loop_closure_gap_m` for Marker 0 is `null` (unverified loop closure), which caps the grade at `C` (`YS-32`, `YS-42`).

---

## 4. Tape Check, Closed-Form Tolerance & Grading (`YS-40`–`YS-45`)

### 4.1 In-App Tape Check (`YS-40`)
- Using a **Center Crosshair** aim-and-tap workflow (walking up close $\sim 1.2\text{ m}$ to the `0.00 m` end of the tape to tap `Tape A`, then walking to the far end of the tape to tap `Tape B` from $\sim 1.2\text{ m}$ away), the app unprojects the median depth of the center $5 \times 5$ LiDAR depth patch (`confidence >= 1`), records the 3D world coordinates $p_A, p_B$, prompts for the physical tape reading `tape_m` on a lightweight inline keypad, and immediately displays:
  - `measured_m` $= \|p_B - p_A\|_2$
  - `residual_m` $= |\text{measured\_m} - \text{tape\_m}|$
- The walker records 1 primary tape pair on a normal walk (`"role": "primary"`, recommended length $\ge 2.0\text{ m}$), plus 3 held-out tape pairs (`"role": "held_out"`) on a calibration walk (`YS-41`).
- **Geometry Invariance (`YS-43`):** Neither `yardstick convert` nor the iPhone app ever uses `tape_m` to scale, rotate, or translate any point, pose, or frame.

### 4.2 Closed-Form 95% Tolerance Formula (`YS-41`)
Grading `v1` (`"grading_version": "yardstick-grade/1"`) computes the 95% distance tolerance $\text{tol}_{\text{dist}}(r)$ at range $r$ (in metres) using ONLY `"role": "primary"` tape checks for the floor (so `"role": "held_out"` checks are strictly independent and non-circular):

$$\text{tol}_{\text{dist}}(r) = \max\left(\text{primary\_tape\_floor}(r),\; 1.96 \sqrt{\sigma_{\text{lidar}}^2 + (\epsilon_{\text{scale,eff}} \cdot r)^2 + \left(\frac{g_{\text{loop}}}{L_{\text{walk}}} \cdot r\right)^2 + \sigma_{\text{tracking}}^2}\right)$$

where:
- $\text{primary\_tape\_floor}(r) = \max_{k \in \text{primary}} \left(\text{residual\_m}_k \cdot \max\left(1.0,\; \frac{r}{\max(\text{tape\_m}_k, 2.0)}\right)\right)$, guaranteeing both the `YS-41` floor ($\text{tol}_{\text{dist}}(r) \ge \text{max\_tape\_residual\_m}$) and proportional scaling when a short primary tape check reveals a scale error;
- $\sigma_{\text{lidar}} = 0.015\text{ m}$ ($1.5\text{ cm}$ 1-$\sigma$ LiDAR range + raycast endpoint noise floor);
- $\epsilon_{\text{scale,eff}} = \max(0.0,\; |\epsilon_{\text{scale}}| - 2 \cdot \text{SE}_{\text{scale}})$ (`YS-33`, `Q027`);
- $g_{\text{loop}}$ is Marker 0's loop-closure gap in metres (`loop_closure_gap_m`), and $L_{\text{walk}} = \max(\text{total\_trajectory\_length\_m},\; 10.0\text{ m})$;
- $\sigma_{\text{tracking}} = 0.005\text{ m} + 0.05\text{ m} \cdot f_{\text{limited}}$, where $f_{\text{limited}} \in [0, 1]$ is the fraction of saved frames with `tracking_state != "normal"`.
- **Held-Out Calibration Verification (`YS-41`):** Each held-out tape check $j$ (`"role": "held_out"`) with length $r_j = \text{tape\_m}_j$ is tested against $\text{tol}_{\text{dist}}(r_j)$ (`residual_m` $\le \text{tol}_{\text{dist}}(r_j)$).
- **Summary Scalars in `grade.json`:**
  - `distance_m`: $\text{tol}_{\text{dist}}(10.0)$ evaluated at reference range $r = 10.0\text{ m}$.
  - `position_m`: $\text{tol}_{\text{dist}}(10.0) / \sqrt{2}$ (horizontal position uncertainty in the yard frame at $10\text{ m}$).
  - `height_m`: $\max\left(\text{ground\_plane\_residual\_m} \cdot 1.96,\; 1.96 \sqrt{\sigma_{\text{lidar}}^2 + (\epsilon_{\text{scale,eff}} \cdot 3.0)^2}\right)$ (vertical height uncertainty at a $3.0\text{ m}$ structure height above the $z=0$ ground plane).

### 4.3 Grading Thresholds & Caps (`YS-42`, `YS-44`)

| Grade | `distance_m` (95% tol at 10 m) | Marker 0 `loop_closure_gap_m` | Integrity (`YS-23`) | Required Evidence |
|---|---|---|---|---|
| **A** | $\le 0.05\text{ m}$ ($5\text{ cm}$) | $\le 0.05\text{ m}$ ($5\text{ cm}$) | `clean` | Marker 0 start & end + $\ge 1$ primary tape check |
| **B** | $\le 0.10\text{ m}$ ($10\text{ cm}$) | $\le 0.12\text{ m}$ ($12\text{ cm}$) | `clean` or `warnings` | Marker 0 start & end + $\ge 1$ primary tape check |
| **C** | $\le 0.20\text{ m}$ ($20\text{ cm}$) | $\le 0.25\text{ m}$ ($25\text{ cm}$) (or `null` if capped) | `clean` or `warnings` | Any valid capture |
| **fail** | $> 0.20\text{ m}$ | $> 0.25\text{ m}$ | `partial` or `invalid` | — |

**Mandatory Caps (`YS-31`, `YS-32`, `YS-42`):**
- `no_marker_0`: Marker 0 never sighted $\rightarrow$ grade capped at `C`.
- `no_loop_closure`: Marker 0 sighted at start only (no return sighting to measure drift) $\rightarrow$ grade capped at `C`.
- `no_tape_check`: Zero primary tape checks recorded $\rightarrow$ grade capped at `C`.
- `limited_tracking_over_10pct`: $> 10\%$ of saved frames had limited tracking (`YS-11`) $\rightarrow$ recorded in `reasons` and caps grade at `B` (or lower if $\text{tol}_{\text{dist}}$ exceeds `B`).
- `integrity_not_usable`: Integrity is `partial` or `invalid` $\rightarrow$ grade is `fail`.

---

## 5. Point Cloud & Scanned Ground Polygon (`YS-50`, `YS-52`)

### 5.1 Point Cloud Generation (`cloud.ply`, `YS-50`)
- To avoid redundant accumulation across 5,400 frames while preserving complete coverage, `yardstick convert` unprojects depth frames (strided every $k = 5$ saved frames, i.e., $6\text{ Hz}$, plus every keyframe/still frame) using pixels where:
  - `confidence >= 1` (medium or high);
  - $0.20\text{ m} \le \text{range\_m} \le 4.50\text{ m}$ (Euclidean distance from camera center at capture time).
- Each 3D point is transformed into the Yard Frame and quantized into a $2.5\text{ cm}$ ($0.025\text{ m}$) 3D voxel grid:
  - Position $(x, y, z)$: centroid (mean) of points falling into the voxel (float32);
  - Color `(red, green, blue)`: mean RGB color sampled from the corresponding uncompressed still or decoded video frame (uint8);
  - `confidence`: maximum confidence value in the voxel (uint8, `1` or `2`);
  - `range_m`: minimum measurement range from the phone in the voxel (float32).
- Written in canonical **`format binary_little_endian 1.0`** PLY format with exact vertex properties `x, y, z, confidence, red, green, blue, range_m`.

### 5.2 Scanned Ground Polygon (`coverage.json`, `YS-52`)
1. Filter unprojected points in the Yard Frame to ground points (`|z| <= 0.25 m`, `range_m <= 4.0 m`, `confidence >= 1`).
2. Project onto a 2D XY grid anchored at the yard origin with cell size $0.20\text{ m} \times 0.20\text{ m}$, where cell $(i, j) = (\lfloor x / 0.20 \rfloor, \lfloor y / 0.20 \rfloor)$ has center $((i + 0.5) \cdot 0.20, (j + 0.5) \cdot 0.20)$. Mark a cell occupied if it contains $\ge 1$ point; mark it high-confidence if it contains $\ge 1$ point with `confidence == 2`.
3. Apply a 1-cell ($3 \times 3$ structuring element) binary morphological closing to bridge small stride gaps, and fill only small interior occlusion holes up to area $\le 1.0\text{ m}^2$ ($\le 25$ grid cells, such as behind a tree trunk or patio table); any unscanned interior void $> 1.0\text{ m}^2$ is preserved in `holes_xy_m` (or excluded via a cut) so LotFit never assumes an unscanned yard center was scanned.
4. Select the largest 4-connected component.
5. Trace its outer boundary contour in counter-clockwise (CCW) order and simplify with Douglas-Peucker ($\epsilon = 0.10\text{ m}$). If simplification introduces a self-intersection, fall back iteratively to smaller $\epsilon$ down to $0.02\text{ m}$ so the resulting polygon is strictly simple, CCW, and has `area_m2 > 0`.
6. Compute `high_confidence_share` as the ratio of occupied grid cells before closing that had `confidence == 2` to total occupied cells inside the polygon.

---

## 6. File-by-File Contract Specification (`yardstick-capture/1`)

Every JSON file is UTF-8 with sorted keys, `"contract": "yardstick-capture/1"`, and `"additionalProperties": false` in its JSON Schema. The authoritative JSON Schemas live in `contract/*.schema.json`.

### 6.1 Level 1 CSV Headers
- **`frames.csv`:**
  ```csv
  t,saved_index,dropped,drop_reason,tracking_state,tracking_reason
  0.000000,0,false,,normal,
  0.033333,,true,writer_backpressure,limited,excessive_motion
  ```
- **`poses.csv`:**
  ```csv
  t,saved_index,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height
  ```
- **`events.csv`:**
  ```csv
  t,saved_index,kind,detail
  ```
  where `kind` $\in \{\text{"marker"}, \text{"tape\_a"}, \text{"tape\_b"}, \text{"done"}, \text{"note"}\}$.
- **`stills.csv`:**
  ```csv
  t,saved_index,file,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height
  ```
- **`markers.csv`** (`x,y,z,qx,qy,qz,qw` is marker-to-world $T_{\text{world} \leftarrow \text{marker}}$ in ARKit world coordinates, plus 2D pixel corners `c0_u..c3_v` in RGB image coordinates):
  ```csv
  t,saved_index,marker_id,size_mm_printed,size_mm_measured,x,y,z,qx,qy,qz,qw,c0_u,c0_v,c1_u,c1_v,c2_u,c2_v,c3_u,c3_v
  ```

### 6.2 Level 1 JSON Files (`session.json`, `manifest.json`)
- **`session.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `capture_id`: string
  - `device`: `{ "model": "iPhone18,1", "name": "iPhone 17 Pro", "ios_version": "19.0", "lidar_supported": true }`
  - `app_version`: string (`"0.1.0"`)
  - `clock`: `{ "started_at_utc": "...", "t0_monotonic_s": 12345.678, "start_clock_pair": { "t_s": 0.0, "utc": "..." }, "stop_clock_pair": { "t_s": 180.0, "utc": "..." }, "audio_first_sample_t_s": 0.002, "audio_sample_rate_hz": 16000 }`
  - `camera_format`: `{ "requested": { "width": 1920, "height": 1440, "fps": 30 }, "actual": { "width": 1920, "height": 1440, "fps": 30 } }` (where `actual.fps` is the active nominal format rate and empirical delivery rate is checked by `YS-02`)
  - `pose_convention`: `"arkit_right_handed_y_up"`
  - `state`: `{ "status": "stopped", "last_heartbeat_t_s": 180.0, "stop_reason": "user_done", "failed_stream": null }`
  - `location`: `{ "status": "ok" | "denied" | "accuracy_exceeded" | "unavailable", "latitude_deg": 44.0521, "longitude_deg": -123.0868, "horizontal_accuracy_m": 4.5, "true_heading_deg": 182.4, "magnetic_heading_deg": 167.1, "heading_accuracy_deg": 8.2, "north_samples": [{ "t_s": 0.2, "camera_yaw_world_deg": -12.4, "true_heading_deg": 182.4, "heading_accuracy_deg": 8.2 }] }`
  - `marker_config`: `{ "dictionary": "DICT_5X5_250", "printed_sizes_mm": { "0": 250.0, "1": 250.0, "2": 250.0, "3": 250.0 } }`
  - `guide_thresholds`: `{ "pitch_min_deg": -65.0, "pitch_max_deg": -10.0, "roll_max_abs_deg": 20.0 }`
  - `loop_closure_gaps`: `[{ "marker_id": 0, "first_t_s": 1.2, "last_t_s": 176.4, "gap_m": 0.038 }]`
  - `tape_checks`: `[{ "id": 0, "role": "primary", "t_a_s": 40.1, "t_b_s": 45.2, "saved_index_a": 1203, "saved_index_b": 1356, "point_a_world_m": [x, y, z], "point_b_world_m": [x, y, z], "tape_m": 5.000, "measured_m": 5.021, "residual_m": 0.021 }]`
  - `provisional_grade`: `{ "grade": "A" | "B" | "C" | "fail", "distance_tolerance_10m_m": 0.048, "loop_closure_gap_m": 0.038, "reasons": [...] }`
- **`manifest.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `capture_id`: string
  - `sealed_at_utc`: ISO-8601 string
  - `streams_declared`: array of stream identifiers (`["session", "frames", "rgb_video", "depth", "confidence", "poses", "audio", "events", "stills", "markers"]`)
  - `files`: array of `{ "path": "relative/path", "size_bytes": 1234, "sha256": "64-hex-chars" }` for every file in `<capture-id>/` except `manifest.json` itself and the generated `standard/` directory.

### 6.3 Level 2 JSON Files (`<capture-id>/standard/`) & Integrity Mapping (`YS-23`)
- **Validator Integrity Mapping (`YS-23`):**
  - `invalid`: hash mismatch, undeclared or missing file, JSON Schema failure, `saved_index` gap, dropped frame without `drop_reason`, `requested != actual` camera format (`YS-03`), non-orthonormal SE(3) rotation ($|\det(R) - 1| > 10^{-4}$ or $\|R^T R - I\|_{\max} > 10^{-4}$), or forbidden interpretation field (`YS-53`).
  - `partial`: `session.json#state.status != "stopped"` or `stop_reason != "user_done"` (`YS-08`, `YS-10`).
  - `warnings`: recording completed (`user_done`) with valid hashes/schemas, but unplanned drops $> 5\%$ (`YS-02`), limited tracking $> 10\%$ (`YS-11`), thermal state serious/critical (`YS-15`), or location/north `denied`/`accuracy_exceeded`/`unavailable` (`YS-14`).
  - `clean`: all checks pass with zero errors or warnings.
- **`capture.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `capture_id`: string
  - `recording_manifest_sha256`: 64-char hex SHA-256 of `<capture-id>/manifest.json`
  - `captured_at_utc`: ISO-8601 string (`started_at_utc`)
  - `source`: `{ "app": "Yardstick", "app_version": "0.1.0", "device_model": "iPhone18,1", "device_name": "iPhone 17 Pro", "ios_version": "19.0", "sensor": "lidar" }`
  - `location`: same structure as `session.json#location`
  - `voxel_size_m`: `0.025`
  - `integrity`: `{ "status": "clean" | "warnings" | "partial" | "invalid", "findings": [{ "code": "...", "severity": "info" | "warning" | "error", "message": "..." }] }`
  - `files`: array of `{ "path": "relative/path", "size_bytes": 1234, "sha256": "64-hex-chars" }` sealing every sibling file in `standard/` (`frame.json`, `cloud.ply`, `keyframes.json`, `keyframes/*.jpg`, `trajectory.json`, `coverage.json`, `audio.wav`, `events.json`, `grade.json`).
- **`frame.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `method`: `"marker"` | `"first_pose"`
  - `marker_id`: `0` (or `null` when `method == "first_pose"`)
  - `marker_0_normal_ok`: boolean (true when Marker 0's normal is within $30^\circ$ of ground up; if false, grade is capped at `C` per `YS-31`)
  - `pose_convention`: `"yard_right_handed_z_up"`
  - `world_to_yard_transform`: 16-element array of numbers ($4 \times 4$ row-major homogeneous matrix)
  - `origin_height_m`: number (signed height of Marker 0 above fitted ground plane $z = 0$)
  - `origin_residual_m`: number or `null` (3D distance between Marker 0 start and end sightings in yard frame)
  - `ground_plane_residual_m`: number (median $|z|$ of ground plane inliers)
  - `north_calibration`: `{ "status": "ok" | "denied" | "accuracy_exceeded" | "insufficient_samples", "north_angle_deg": 92.4, "north_uncertainty_deg": 8.5, "sample_count": 12 }`
  - `loop_closure_gaps`: `[{ "marker_id": 0, "first_t_s": 1.2, "last_t_s": 176.4, "gap_m": 0.038 }]`
  - `pose_jumps`: `[{ "saved_index_from": 2400, "saved_index_to": 2401, "t_s": 80.033, "jump_m": 0.112 }]`
- **`keyframes.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `keyframes`: `[{ "id": 0, "t": 0.0, "saved_index": 0, "file": "keyframes/000000.jpg", "pose": [16 numbers, 4x4 camera->yard row-major], "intrinsics": { "fx": ..., "fy": ..., "cx": ..., "cy": ..., "width": 1920, "height": 1440 } }]`
- **`trajectory.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `poses`: `[{ "t": 0.0, "saved_index": 0, "pose": [16 numbers, 4x4 camera->yard row-major] }]`
- **`coverage.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `polygon_xy_m`: array of `[x, y]` vertices in counter-clockwise order (no duplicate closing vertex)
  - `holes_xy_m`: array of interior hole polygons ($\ge 1.0\text{ m}^2$), each in clockwise order (empty array `[]` when fully scanned)
  - `area_m2`: number ($> 0$, net scanned area)
  - `high_confidence_share`: number in $[0, 1]$
  - `parameters`: `{ "z_max_abs_m": 0.25, "range_max_m": 4.0, "grid_cell_m": 0.20, "closing_cells": 1, "max_filled_hole_m2": 1.0, "simplify_epsilon_m": 0.10 }`
- **`events.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `audio_file`: `"audio.wav"`
  - `audio_first_sample_t_s`: number
  - `audio_sample_rate_hz`: `16000`
  - `events`: `[{ "t": 1.5, "saved_index": 45, "kind": "marker" | "tape_a" | "tape_b" | "done" | "note", "detail": "..." }]`
- **`grade.json`:**
  - `contract`: `"yardstick-capture/1"`
  - `grading_version`: `"yardstick-grade/1"`
  - `grade`: `"A"` | `"B"` | `"C"` | `"fail"`
  - `tolerances_95pct`: `{ "distance_m": 0.048, "height_m": 0.039, "position_m": 0.034, "reference_range_m": 10.0, "formula": "max(primary_tape_floor(r), 1.96 * sqrt(sigma_lidar_m^2 + (scale_error_eff * r)^2 + (loop_gap_per_m * r)^2 + sigma_tracking_m^2))", "coefficients": { "max_tape_residual_m": 0.021, "primary_tape_floor_10m_m": 0.042, "sigma_lidar_m": 0.015, "scale_error_ratio": 0.004, "scale_error_se": 0.0025, "scale_error_eff": 0.000, "loop_closure_gap_m": 0.038, "walk_length_m": 42.5, "loop_gap_per_m": 0.000894, "sigma_tracking_m": 0.006, "limited_tracking_share": 0.02 }, "evidence_basis": ["tape_checks", "loop_closure_gaps", "marker_scale_error", "tracking_quality", "lidar_noise_floor"] }`
  - `tape_checks`: `[{ "id": 0, "role": "primary" | "held_out", "point_a_yard_m": [x, y, z], "point_b_yard_m": [x, y, z], "tape_m": 5.000, "measured_m": 5.021, "residual_m": 0.021, "tolerance_at_tape_m": 0.035, "within_tolerance": true }]`
  - `marker_scale`: `{ "scale_error_ratio": 0.004, "scale_error_se": 0.0025, "scale_error_eff": 0.000, "sightings_used": 12, "per_marker": [{ "marker_id": 0, "size_mm_printed": 250.0, "size_mm_measured": 251.0, "scale_error_ratio": 0.004, "scale_error_se": 0.0025, "sightings_count": 12 }] }`
  - `caps_applied`: array of cap codes (`[]` or `["no_tape_check"]`, etc.)
  - `reasons`: array of human-readable and machine-parseable evidence strings explaining the grade and every cap applied.

