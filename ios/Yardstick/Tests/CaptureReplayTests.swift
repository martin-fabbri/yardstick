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
import os
import XCTest
import simd

/// Observes `YS-66` test execution so any run where a `YS-66` test was skipped explicitly prints
/// `YS-66 VERDICT: NOT RUN` instead of claiming a full pass.
final class YS66SkipObserver: NSObject, XCTestObservation, @unchecked Sendable {
    static let shared = YS66SkipObserver()

    private struct State: Sendable {
        var registered = false
        var executedTests: [String] = []
        var skippedTests: [String] = []
        var failedTests: [String] = []
    }

    private let stateLock = OSAllocatedUnfairLock(initialState: State())

    static func ensureRegistered() {
        let shouldRegister = shared.stateLock.withLock { state -> Bool in
            if state.registered { return false }
            state.registered = true
            return true
        }
        if shouldRegister {
            XCTestObservationCenter.shared.addTestObserver(shared)
        }
    }

    func testCaseDidFinish(_ testCase: XCTestCase) {
        guard let run = testCase.testRun else { return }
        let name = testCase.name
        guard name.contains("YS66") else { return }
        let skipped = run.hasBeenSkipped
        let failed = run.failureCount > 0
        stateLock.withLock { state in
            if skipped {
                state.skippedTests.append(name)
            } else if failed {
                state.failedTests.append(name)
            } else {
                state.executedTests.append(name)
            }
        }
    }

    func testBundleDidFinish(_ testBundle: Bundle) {
        let snap = stateLock.withLock { $0 }
        let total = snap.executedTests.count + snap.skippedTests.count + snap.failedTests.count
        guard total > 0 else { return }
        if !snap.failedTests.isEmpty {
            print("YS-66 VERDICT: FAILED (\(snap.failedTests.count) failed, \(snap.skippedTests.count) skipped, \(snap.executedTests.count) run)")
        } else if !snap.skippedTests.isEmpty {
            print("YS-66 VERDICT: NOT RUN (\(snap.skippedTests.count) test(s) skipped: \(snap.skippedTests.joined(separator: ", ")))")
        } else {
            print("YS-66 VERDICT: PASSED (\(snap.executedTests.count) of \(total) test(s) run, 0 skipped)")
        }
    }
}

final class CaptureReplayTests: XCTestCase {
    override class func setUp() {
        super.setUp()
        YS66SkipObserver.ensureRegistered()
    }

    private static var fixturesDirectoryURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures", isDirectory: true)
    }

    private static var toolkitGoldenURL: URL {
        fixturesDirectoryURL.appendingPathComponent("toolkit_golden_walks_1_5.json")
    }

    private static var trimmedWalk5SampleURL: URL {
        fixturesDirectoryURL.appendingPathComponent("walk5_trimmed_sample", isDirectory: true)
    }

    /// Reads `YARDSTICK_REPLAY_ROOT` from the environment (no personal paths in source).
    private static var replayRootURL: URL? {
        let env = ProcessInfo.processInfo.environment
        for key in ["YARDSTICK_REPLAY_ROOT", "TEST_RUNNER_YARDSTICK_REPLAY_ROOT"] {
            if let val = env[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
               !val.isEmpty,
               !val.hasPrefix("$(") {
                return URL(fileURLWithPath: val, isDirectory: true)
            }
        }
        return nil
    }

    // MARK: - Always-On Checked-In Fixture Suite (0 Skips in CI and Local Runs)

    /// `YS-66`: Always-run replay proof on the checked-in trimmed Walk-5 sample (`walk5_trimmed_sample`,
    /// 6 representative stills across Markers 0–3 + 9 startup dropped rows) using `.liveDetectorOnStills`.
    /// Never skips; verifies 2D+3D marker sizes within `±0.1 mm`, the phone's false `7.5 cm` (`0.0753 m`)
    /// opening loop close, and comparison against the Python toolkit's checked-in golden verdict.
    func testYS66_TrimmedWalk5Sample_LiveDetectorOnStills_ReproducesMarkerSizesAndFalseClose() throws {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.trimmedWalk5SampleURL.path),
            "YS-66: Checked-in trimmed Walk-5 fixture missing at \(Self.trimmedWalk5SampleURL.path)"
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.toolkitGoldenURL.path),
            "YS-66: Checked-in Python toolkit golden file missing at \(Self.toolkitGoldenURL.path)"
        )

        let golden = try ToolkitGoldenFile.load(from: Self.toolkitGoldenURL)
        guard let walk5Golden = golden.walks["walk-5"] else {
            XCTFail("YS-66: Missing walk-5 entry in toolkit_golden_walks_1_5.json")
            return
        }

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: Self.trimmedWalk5SampleURL,
            mode: .liveDetectorOnStills,
            defaultPrintedSizeMM: 180.0
        )

        XCTAssertEqual(result.totalFrameRows, 15, "YS-66: Trimmed Walk 5 has 9 dropped + 6 saved frame rows")
        XCTAssertEqual(result.savedFrameCount, 6, "YS-66: Trimmed Walk 5 has 6 saved frames")
        XCTAssertEqual(result.droppedFrameCount, 9, "YS-66: Trimmed Walk 5 has 9 startup dropped frames")
        XCTAssertEqual(result.stillCount, 6, "YS-66: Trimmed Walk 5 has 6 stills")
        XCTAssertLessThan(result.elapsedSeconds, 5.0, "YS-66: Trimmed Walk 5 replay must complete in < 5 s")

        for markerID in 0...3 {
            guard let origMedian = CaptureReplayResult.medianSizeMM(for: markerID, in: result.originalSightings),
                  let replayMedian = CaptureReplayResult.medianSizeMM(for: markerID, in: result.replayedSightings)
            else {
                XCTFail("YS-66: Missing sightings for marker \(markerID) in trimmed Walk 5")
                continue
            }
            let diffMM = abs(replayMedian - origMedian)
            XCTAssertLessThanOrEqual(
                diffMM,
                0.10,
                String(
                    format: "YS-66: Trimmed Walk 5 marker %d size %.3f mm differs from phone original %.3f mm by %.3f mm (> 0.1 mm)",
                    markerID, replayMedian, origMedian, diffMM
                )
            )
        }

        // Phone's original rule produces the false 7.5 cm (0.0753 m) loop close on Marker 0 at saved_index=438
        guard let phoneGapM = result.phoneLoopState.loopClosureGapMeters else {
            XCTFail("YS-66: Expected phone's original rule to report false loop close on trimmed Walk 5")
            return
        }
        XCTAssertEqual(phoneGapM, 0.0753, accuracy: 0.002, "YS-66: Expected false 7.5 cm (0.0753 m) loop close")

        // Real Python toolkit golden verdict reports Walk 5 as not closed
        XCTAssertFalse(walk5Golden.grade_coeff_reports_closed, "YS-66: Python toolkit grade.json reports Walk 5 not closed")
        XCTAssertFalse(walk5Golden.frame_marker0_loop_closed, "YS-66: Python toolkit frame.json reports Walk 5 not closed")
        XCTAssertTrue(walk5Golden.caps_applied.contains("no_loop_closure"), "YS-66: Python toolkit caps Walk 5 with no_loop_closure")
    }

    /// `YS-66`: Always-run replay proof on the checked-in trimmed Walk-5 sample using `.recordedCornersFromCSV`.
    func testYS66_TrimmedWalk5Sample_RecordedCornersFromCSV_MatchesEverySightingWithin0_01mm() throws {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.trimmedWalk5SampleURL.path),
            "YS-66: Checked-in trimmed Walk-5 fixture missing at \(Self.trimmedWalk5SampleURL.path)"
        )

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: Self.trimmedWalk5SampleURL,
            mode: .recordedCornersFromCSV,
            defaultPrintedSizeMM: 180.0
        )

        XCTAssertEqual(result.originalSightings.count, 6)
        XCTAssertEqual(result.replayedSightings.count, 6)

        for (orig, rep) in zip(result.originalSightings, result.replayedSightings) {
            XCTAssertEqual(orig.savedIndex, rep.savedIndex)
            XCTAssertEqual(orig.sighting.markerID, rep.sighting.markerID)
            let diff = abs(orig.sighting.sizeMMMeasured - rep.sighting.sizeMMMeasured)
            XCTAssertLessThanOrEqual(diff, 0.01, "YS-66: Sighting at saved_index=\(orig.savedIndex) differed by \(diff) mm")
        }

        XCTAssertEqual(result.phoneLoopState.loopClosureGapMeters ?? -1.0, 0.075326, accuracy: 0.0001)
    }

    // MARK: - Optional Full Walk 1–5 Local Suite (`YARDSTICK_REPLAY_ROOT`)

    /// `YS-66`: Full Walk 5 (`2,865` frames, `455` stills) replay in `.liveDetectorOnStills` mode under `< 15 s`.
    func testYS66_FullWalk5_LiveDetectorOnStills_ReproducesMarkerSizesAndFalseCloseUnder15s() throws {
        guard let rootURL = Self.replayRootURL else {
            throw XCTSkip("NOT RUN: Set YARDSTICK_REPLAY_ROOT to run the full Walk 5 replay suite")
        }
        let walk5URL = rootURL.appendingPathComponent("walk-5/20261101T012134Z_yardstick")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: walk5URL.path),
            "YS-66: Walk 5 directory missing under YARDSTICK_REPLAY_ROOT at \(walk5URL.path)"
        )

        let golden = try ToolkitGoldenFile.load(from: Self.toolkitGoldenURL)
        guard let walk5Golden = golden.walks["walk-5"] else {
            XCTFail("YS-66: Missing walk-5 entry in toolkit_golden_walks_1_5.json")
            return
        }

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: walk5URL,
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
        XCTAssertFalse(walk5Golden.grade_coeff_reports_closed, "YS-66: Toolkit golden reports Walk 5 not closed")
    }

    /// `YS-66`: Full Walk 5 (`432` sightings) replay in `.recordedCornersFromCSV` mode.
    func testYS66_FullWalk5_RecordedCornersFromCSV_MatchesEverySightingWithin0_01mm() throws {
        guard let rootURL = Self.replayRootURL else {
            throw XCTSkip("NOT RUN: Set YARDSTICK_REPLAY_ROOT to run the full Walk 5 recorded-corners suite")
        }
        let walk5URL = rootURL.appendingPathComponent("walk-5/20261101T012134Z_yardstick")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: walk5URL.path),
            "YS-66: Walk 5 directory missing under YARDSTICK_REPLAY_ROOT at \(walk5URL.path)"
        )

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: walk5URL,
            mode: .recordedCornersFromCSV,
            defaultPrintedSizeMM: 180.0
        )

        XCTAssertEqual(result.originalSightings.count, 432, "YS-66: Walk 5 has 432 original marker rows")
        XCTAssertEqual(result.replayedSightings.count, 432, "YS-66: All 432 sightings must reconstruct on LiDAR plane")

        for (orig, rep) in zip(result.originalSightings, result.replayedSightings) {
            XCTAssertEqual(orig.savedIndex, rep.savedIndex)
            XCTAssertEqual(orig.sighting.markerID, rep.sighting.markerID)
            let diff = abs(orig.sighting.sizeMMMeasured - rep.sighting.sizeMMMeasured)
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

    /// `YS-66`: Replay Walks 1–5 and compare the phone's loop verdict and marker counts against the
    /// checked-in Python toolkit golden file (`toolkit_golden_walks_1_5.json`).
    func testYS66_FullWalks1To5_ComparesPhoneVerdictsAgainstToolkitGoldenFile() throws {
        guard let rootURL = Self.replayRootURL else {
            throw XCTSkip("NOT RUN: Set YARDSTICK_REPLAY_ROOT to run the full Walks 1–5 comparison suite")
        }
        let golden = try ToolkitGoldenFile.load(from: Self.toolkitGoldenURL)

        let expectations: [(
            walkKey: String,
            expectedPhoneGapM: Double?,
            expectedToolkitCoeffClosed: Bool,
            expectedToolkitCoeffGapM: Double?,
            expectedToolkitFrameClosed: Bool
        )] = [
            ("walk-1", 0.2542, true, 0.246154, true),
            ("walk-2", 1.0545, false, nil, false),
            ("walk-3", nil, false, nil, false),
            ("walk-4", nil, true, 0.208437, false),
            ("walk-5", 0.0753, false, nil, false),
        ]

        for exp in expectations {
            guard let gWalk = golden.walks[exp.walkKey] else {
                XCTFail("YS-66: Missing \(exp.walkKey) in toolkit_golden_walks_1_5.json")
                continue
            }
            let walkURL = rootURL.appendingPathComponent(gWalk.relative_path)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: walkURL.path),
                "YS-66: Missing walk directory at \(walkURL.path)"
            )

            let res = try CaptureReplayHarness.replayCaptureDirectory(
                at: walkURL,
                mode: .recordedCornersFromCSV
            )
            XCTAssertGreaterThan(res.totalFrameRows, 0, "YS-66: \(exp.walkKey) must have frames")

            if let expPhoneGap = exp.expectedPhoneGapM {
                guard let actualGap = res.phoneLoopState.loopClosureGapMeters else {
                    XCTFail("YS-66: Expected phone loop gap on \(exp.walkKey), got nil")
                    continue
                }
                XCTAssertEqual(actualGap, expPhoneGap, accuracy: 0.002, "YS-66: \(exp.walkKey) phone loop gap mismatch")
            } else {
                XCTAssertNil(res.phoneLoopState.loopClosureGapMeters, "YS-66: Expected nil phone loop gap on \(exp.walkKey)")
            }

            XCTAssertEqual(gWalk.grade_coeff_reports_closed, exp.expectedToolkitCoeffClosed, "YS-66: \(exp.walkKey) toolkit coeff closed mismatch")
            XCTAssertEqual(gWalk.frame_marker0_loop_closed, exp.expectedToolkitFrameClosed, "YS-66: \(exp.walkKey) toolkit frame closed mismatch")
            if let expToolkitGap = exp.expectedToolkitCoeffGapM {
                XCTAssertEqual(gWalk.grade_coeff_loop_closure_gap_m ?? -1.0, expToolkitGap, accuracy: 0.0001)
            } else {
                XCTAssertNil(gWalk.grade_coeff_loop_closure_gap_m)
            }

            // Verify toolkit golden sightings_count (after YS-69 OpenCV sub-pixel + 3%/4% gates) is non-empty and <= raw phone sightings
            for (midStr, markerEntry) in gWalk.per_marker {
                guard let mid = Int(midStr) else { continue }
                let count = res.replayedSightings.filter { $0.sighting.markerID == mid }.count
                XCTAssertGreaterThan(markerEntry.sightings_count, 0, "YS-66: \(exp.walkKey) marker \(mid) toolkit sightings must be > 0")
                XCTAssertGreaterThanOrEqual(count, markerEntry.sightings_count, "YS-66: \(exp.walkKey) marker \(mid) raw sightings must be >= toolkit filtered sightings")
            }
        }
    }
}
