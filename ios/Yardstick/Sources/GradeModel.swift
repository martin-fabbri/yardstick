// Copyright 2026 Martin Fabbri
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Swift port of the toolkit's `compute_grade_and_tolerances` and `evaluate_marker_scale_errors` aggregation
/// (`toolkit/src/yardstick/grading.py`, `markers.py`; YS-41, YS-42, YS-44), so the phone shows the grade the
/// toolkit will give. Keep the two in step: shared vectors live in `GradeModelTests` and
/// `toolkit/tests/test_yardstick.py::test_grade_parity_vectors`.
public enum GradeModel {
    public static let sigmaLidarM = 0.015
    public static let sigmaTrackingBaseM = 0.005
    public static let sigmaTrackingSlopeM = 0.050
    public static let referenceRangeM = 10.0

    public struct TapeCheck: Sendable, Equatable {
        public var tapeM: Double
        public var residualM: Double
        public var isHeldOut: Bool
        public init(tapeM: Double, residualM: Double, isHeldOut: Bool = false) {
            self.tapeM = tapeM
            self.residualM = residualM
            self.isHeldOut = isHeldOut
        }
    }

    public struct Inputs: Sendable, Equatable {
        public var integrityStatus: String = "ok"
        public var hasMarker0: Bool = true
        public var marker0NormalOk: Bool = true
        public var tapeChecks: [TapeCheck] = []
        public var scaleErrorEff: Double = 0.0
        /// Marker 0 opening-to-closing gap; nil when Marker 0 was not revisited.
        public var rawMarker0GapM: Double?
        public var poseJumpSumM: Double = 0.0
        public var walkLengthM: Double = 0.0
        public var limitedTrackingShare: Double = 0.0
        public init() {}
    }

    public struct Result: Sendable, Equatable {
        public var grade: String
        public var distanceTolerance10mM: Double
        public var primaryTapeFloor10mM: Double
        public var loopClosureGapM: Double?
        public var capsApplied: [String]
        public var reasons: [String]
    }

    /// Closed-form 95 % distance tolerance at range `r` (`compute_tolerance_at_range`).
    public static func tolerance(
        atRangeM r: Double,
        primaryTapes: [TapeCheck],
        scaleErrorEff: Double,
        loopGapPerM: Double,
        sigmaTrackingM: Double
    ) -> Double {
        var floor = 0.0
        for tc in primaryTapes {
            let tapeM = max(tc.tapeM, 2.0)
            floor = max(floor, tc.residualM, tc.residualM * max(1.0, r / tapeM))
        }
        let rss = sqrt(
            sigmaLidarM * sigmaLidarM
                + (scaleErrorEff * r) * (scaleErrorEff * r)
                + (loopGapPerM * r) * (loopGapPerM * r)
                + sigmaTrackingM * sigmaTrackingM
        )
        return max(floor, 1.96 * rss)
    }

    public static func grade(_ inp: Inputs) -> Result {
        let primary = inp.tapeChecks.filter { !$0.isHeldOut }
        let heldOut = inp.tapeChecks.filter { $0.isHeldOut }
        let maxPrimaryRes = primary.map(\.residualM).max() ?? 0.0

        let hasLoop = inp.rawMarker0GapM != nil
        var m0Gap: Double? = inp.rawMarker0GapM
        if let g = m0Gap {
            m0Gap = max(g, inp.poseJumpSumM)
        } else if inp.poseJumpSumM > 0.0 {
            m0Gap = inp.poseJumpSumM
        }
        let effWalkLen = max(inp.walkLengthM, 10.0)
        let loopGapPerM = m0Gap.map { $0 / effWalkLen } ?? 0.008
        let sigmaTracking = sigmaTrackingBaseM + sigmaTrackingSlopeM * inp.limitedTrackingShare

        var floor10 = 0.0
        for tc in primary {
            floor10 = max(floor10, tc.residualM * max(1.0, referenceRangeM / max(tc.tapeM, 2.0)))
        }
        let distTol = tolerance(
            atRangeM: referenceRangeM, primaryTapes: primary, scaleErrorEff: inp.scaleErrorEff,
            loopGapPerM: loopGapPerM, sigmaTrackingM: sigmaTracking)

        var reasons = [String(format: "distance_tolerance_10m_m=%.4fm", distTol)]
        var caps: [String] = []
        if hasLoop, let g = m0Gap {
            reasons.append(String(format: "loop_closure_gap_m=%.4fm", g))
        } else {
            reasons.append("loop_closure_gap_m=null (Marker 0 not revisited at end of walk)")
        }
        if !primary.isEmpty {
            reasons.append(String(format: "max_primary_tape_residual_m=%.4fm", maxPrimaryRes))
        }

        let gapForBand = m0Gap ?? 0.0
        var band: Int
        if distTol <= 0.050 && gapForBand <= 0.050 {
            band = 3
        } else if distTol <= 0.100 && gapForBand <= 0.120 {
            band = 2
        } else if distTol <= 0.200 && gapForBand <= 0.250 {
            band = 1
        } else {
            band = 0
        }

        if inp.integrityStatus == "partial" || inp.integrityStatus == "invalid" {
            band = 0
            caps.append("integrity_\(inp.integrityStatus)")
        }
        if !inp.hasMarker0 {
            band = min(band, 1)
            caps.append("no_marker_0")
        } else if !inp.marker0NormalOk {
            band = min(band, 1)
            caps.append("marker_0_tilted")
        }
        if !hasLoop && inp.hasMarker0 {
            band = min(band, 1)
            caps.append("no_loop_closure")
        }
        if primary.isEmpty {
            band = min(band, 1)
            caps.append("no_tape_check")
        }
        let failedHeldOut = heldOut.filter { tc in
            tc.residualM > tolerance(
                atRangeM: tc.tapeM, primaryTapes: primary, scaleErrorEff: inp.scaleErrorEff,
                loopGapPerM: loopGapPerM, sigmaTrackingM: sigmaTracking) + 1e-9
        }
        if failedHeldOut.count >= 2 {
            band = 0
            caps.append("held_out_tape_failed_2plus")
        } else if failedHeldOut.count == 1 {
            band = min(band, 1)
            caps.append("held_out_tape_failed_1")
        }
        if inp.limitedTrackingShare > 0.10 {
            band = min(band, 2)
            caps.append("limited_tracking_over_10pct")
        }
        if inp.integrityStatus == "warnings" {
            band = min(band, 2)
            caps.append("integrity_warnings")
        }
        reasons.append(contentsOf: caps.map { "cap_applied: \($0)" })

        let labels = [3: "A", 2: "B", 1: "C", 0: "fail"]
        return Result(
            grade: labels[band] ?? "fail",
            distanceTolerance10mM: distTol,
            primaryTapeFloor10mM: floor10,
            loopClosureGapM: m0Gap,
            capsApplied: caps,
            reasons: reasons
        )
    }

    /// `scale_error_eff` from per-sighting marker sizes, as the toolkit aggregates them when it falls back to logged
    /// sizes: per marker, mean error ratio and its standard error (0.5 % for a single sighting); the marker with the
    /// largest |error| wins; eff = max(0, |error| − 2·SE).
    public static func scaleErrorEff(sizesByMarker: [Int: (printedMM: Double, measuredMM: [Double])]) -> Double {
        var worst: (err: Double, se: Double)?
        for mid in sizesByMarker.keys.sorted() {
            guard let entry = sizesByMarker[mid], entry.printedMM > 0, !entry.measuredMM.isEmpty else { continue }
            let errs = entry.measuredMM.map { ($0 - entry.printedMM) / entry.printedMM }
            let mean = errs.reduce(0, +) / Double(errs.count)
            let se: Double
            if errs.count >= 2 {
                let v = errs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(errs.count - 1)
                se = sqrt(v) / sqrt(Double(errs.count))
            } else {
                se = 0.005
            }
            if worst == nil || abs(mean) > abs(worst!.err) {
                worst = (mean, se)
            }
        }
        guard let w = worst else { return 0.0 }
        return max(0.0, abs(w.err) - 2.0 * w.se)
    }
}
