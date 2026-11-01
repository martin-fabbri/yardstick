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
import XCTest
import simd

final class CaptureReplayTests: XCTestCase {
    private static let walk5URL = URL(
        fileURLWithPath: "/Users/martinfabbri/Workspace/scrap/yardstick-captures/2026-10-31/walk-5/20261101T012134Z_yardstick"
    )

    private static let walks1To4BaseURL = URL(
        fileURLWithPath: "/Users/martinfabbri/Workspace/scrap/yardstick-captures/2026-10-31/walks-1-4"
    )

    /// `YS-66`: Replay Walk 5 (`2,865` frames, `455` stills) through the phone's own 2D detector and 3D LiDAR
    /// plane-fit rules, verify it runs in `< 15 s` in `xcodebuild test`, reproduces the phone's original
    /// marker sizes within `±0.1 mm`, and reproduces the phone's false `7.5 cm` loop close while comparing
    /// against the toolkit's `"not closed"` verdict.
    func testYS66_ReplayWalk5_LiveDetectorOnStills_ReproducesMarkerSizesAndFalseCloseUnder15s() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.walk5URL.path),
            "Walk 5 fixture not present at \(Self.walk5URL.path)"
        )

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: Self.walk5URL,
            mode: .liveDetectorOnStills,
            defaultPrintedSizeMM: 180.0
        )

        XCTAssertEqual(result.totalFrameRows, 2865, "YS-66: Walk 5 must have 2,865 total frame rows")
        XCTAssertEqual(result.savedFrameCount, 2856, "YS-66: Walk 5 must have 2,856 saved frames")
        XCTAssertEqual(result.droppedFrameCount, 9, "YS-66: Walk 5 must have 9 startup dropped frames")
        XCTAssertEqual(result.stillCount, 455, "YS-66: Walk 5 must have 455 stills")
        XCTAssertLessThan(
            result.elapsedSeconds,
            15.0,
            String(format: "YS-66: Walk 5 replay took %.2f s (must be < 15.0 s)", result.elapsedSeconds)
        )

        for markerID in 0...3 {
            guard let origMedian = CaptureReplayResult.medianSizeMM(for: markerID, in: result.originalSightings),
                  let replayMedian = CaptureReplayResult.medianSizeMM(for: markerID, in: result.replayedSightings)
            else {
                XCTFail("YS-66: Missing sightings for marker \(markerID)")
                continue
            }
            let diffMM = abs(replayMedian - origMedian)
            XCTAssertLessThanOrEqual(
                diffMM,
                0.10,
                String(
                    format: "YS-66: Marker %d median size %.3f mm differs from phone original %.3f mm by %.3f mm (> 0.1 mm)",
                    markerID, replayMedian, origMedian, diffMM
                )
            )
        }

        // Verify the phone's original loop-closure rule reproduces the false 7.5 cm (0.0753 m) close during the opening hold
        guard let gapM = result.phoneLoopState.loopClosureGapMeters else {
            XCTFail("YS-66: Expected phone's original rule to report false loop close on Walk 5")
            return
        }
        XCTAssertEqual(gapM, 0.0753, accuracy: 0.002, "YS-66: Expected false 7.5 cm (0.0753 m) loop close on Walk 5")
        XCTAssertEqual(
            result.phoneLoopState.marker0LastPassTimeS ?? 0.0,
            726.672,
            accuracy: 0.05,
            "YS-66: False loop close occurred at t = 726.672 s (t_rel = 14.60 s)"
        )

        // Compare with the toolkit's reference loop-closure rule on the same sightings
        XCTAssertFalse(result.toolkitLoopVerdict.closed, "YS-66: Toolkit rule must report Walk 5 as not closed")
        XCTAssertEqual(result.toolkitLoopVerdict.reason, "single_pass_only", "YS-66: Walk 5 had only 1 pass on Marker 0")
    }

    /// `YS-66`: Replay Walk 5's `432` recorded 2D marker quads against the raw 16-bit depth and 8-bit confidence
    /// PNGs and verify every single 3D size measurement matches `markers.csv` within `±0.01 mm` (well inside `±0.1 mm`).
    func testYS66_ReplayWalk5_RecordedCornersFromCSV_MatchesEverySightingWithin0_01mm() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.walk5URL.path),
            "Walk 5 fixture not present at \(Self.walk5URL.path)"
        )

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: Self.walk5URL,
            mode: .recordedCornersFromCSV,
            defaultPrintedSizeMM: 180.0
        )

        XCTAssertEqual(result.originalSightings.count, 432, "YS-66: Walk 5 has 432 original marker rows")
        XCTAssertEqual(result.replayedSightings.count, 432, "YS-66: All 432 sightings must reconstruct on LiDAR plane")

        var maxAbsDiffMM = 0.0
        for (orig, rep) in zip(result.originalSightings, result.replayedSightings) {
            XCTAssertEqual(orig.savedIndex, rep.savedIndex)
            XCTAssertEqual(orig.sighting.markerID, rep.sighting.markerID)
            let diff = abs(orig.sighting.sizeMMMeasured - rep.sighting.sizeMMMeasured)
            if diff > maxAbsDiffMM { maxAbsDiffMM = diff }
            XCTAssertLessThanOrEqual(
                diff,
                0.01,
                "YS-66: Sighting at saved_index=\(orig.savedIndex) differed by \(diff) mm"
            )
        }

        XCTAssertEqual(
            result.phoneLoopState.loopClosureGapMeters ?? -1.0,
            0.075327,
            accuracy: 0.0001,
            "YS-66: Exact 7.53 cm false loop close reproduced"
        )
    }

    /// `YS-66`: Replay Walks 1–5 and compare the phone's loop verdict and marker counts against the toolkit's.
    func testYS66_ReplayWalks1To5_ComparesPhoneAndToolkitVerdicts() throws {
        let expectations: [(name: String, url: URL, expectedPhoneGapM: Double?, expectedToolkitClosed: Bool)] = [
            ("walk-1", Self.walks1To4BaseURL.appendingPathComponent("20261101T003322Z_yardstick"), 0.2542, true),
            ("walk-2", Self.walks1To4BaseURL.appendingPathComponent("20261101T003455Z_yardstick"), 1.0545, false),
            ("walk-3", Self.walks1To4BaseURL.appendingPathComponent("20261101T003609Z_yardstick"), nil, false),
            ("walk-4", Self.walks1To4BaseURL.appendingPathComponent("20261101T003631Z_yardstick"), nil, false),
            ("walk-5", Self.walk5URL, 0.0753, false),
        ]

        for item in expectations {
            try XCTSkipUnless(
                FileManager.default.fileExists(atPath: item.url.path),
                "YS-66: Fixture \(item.name) not present at \(item.url.path)"
            )
            let res = try CaptureReplayHarness.replayCaptureDirectory(
                at: item.url,
                mode: .recordedCornersFromCSV
            )
            XCTAssertGreaterThan(res.totalFrameRows, 0, "YS-66: \(item.name) must have frames")
            if let expGap = item.expectedPhoneGapM {
                guard let actualGap = res.phoneLoopState.loopClosureGapMeters else {
                    XCTFail("YS-66: Expected phone loop gap on \(item.name), got nil")
                    continue
                }
                XCTAssertEqual(actualGap, expGap, accuracy: 0.002, "YS-66: \(item.name) phone loop gap mismatch")
            } else {
                XCTAssertNil(res.phoneLoopState.loopClosureGapMeters, "YS-66: Expected nil phone loop gap on \(item.name)")
            }
            XCTAssertEqual(
                res.toolkitLoopVerdict.closed,
                item.expectedToolkitClosed,
                "YS-66: \(item.name) toolkit loop verdict mismatch (reason: \(res.toolkitLoopVerdict.reason))"
            )
        }
    }
}

