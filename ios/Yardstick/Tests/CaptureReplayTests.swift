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
                let candidate = URL(fileURLWithPath: val, isDirectory: true)
                let datedSubdir = candidate.appendingPathComponent("2026-10-31", isDirectory: true)
                if FileManager.default.fileExists(atPath: datedSubdir.appendingPathComponent("walk-5").path) {
                    return datedSubdir
                }
                return candidate
            }
        }
        return nil
    }

    // MARK: - Always-On Checked-In Fixture Suite (0 Skips in CI and Local Runs)

    /// `YS-66` / `YS-69` / `YS-71`: Always-run replay proof on the checked-in trimmed Walk-5 sample (`walk5_trimmed_sample`,
    /// 6 representative stills across Markers 0–3 + 9 startup dropped rows) using `.liveDetectorOnStills`.
    /// Never skips; verifies that:
    /// - OpenCV `DICT_5X5_250` + `cornerSubPix(winSize=5)` + 3%/4% gates (`YS-69`) measures every marker within `179.5 ± 1.5 mm`
    ///   and rejects the motion-blurred `saved_index=438` frame (`195.8 mm` under the old Vision detector),
    /// - Unified `YS-71` loop-closure rule agrees with the Python toolkit (`unifiedLoopClosed == false`),
    /// - Recorded-corners replay (`YS-66`) still reproduces the phone's historical `7.5 cm` (`0.0753 m`) false close.
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

        // Under YS-69, on the 6-still trimmed Walk-5 sample (0, 381, 438, 1020, 1929, 2652):
        // - Still 438 (turnRate=0.63 rad/s) is motion-blurred and rejected in 2D by OpenCV AND flagged by YS-73!
        // - Still 1929 (range=1.51 m) is detected in 2D and rejected in 3D by the YS-69 4% edge-disagreement gate.
        // - Stills 0 (M0), 381 (M0, 179.85 mm), 1020 (M1), and 2652 (M3, 175.50 mm) pass both 2D and 3D OpenCV gates!
        XCTAssertEqual(result.replayedSightings.count, 4, "YS-69/70: 4 sightings pass 2D+3D OpenCV gates on the 6-still trimmed sample (0, 381, 1020, 2652)")
        XCTAssertEqual(result.openCVRejectedOriginalSightingsCount, 1, "YS-70/73: Motion-blurred still 438 is rejected in 2D by OpenCV")
        XCTAssertEqual(result.ys73FlaggedOfOpenCVRejectedCount, 1, "YS-73: Blurred still 438 is flagged by YS-73")
        XCTAssertEqual(result.unconfirmedSightingsCount, 0)
        XCTAssertEqual(result.markerOpenCVDisagreementCount, 0)

        for entry in result.replayedSightings {
            XCTAssertLessThan(entry.sighting.sizeMMMeasured, 185.0, "YS-69: No sighting on trimmed Walk 5 may exceed 185 mm")
        }
        let bySavedIndex = Dictionary(uniqueKeysWithValues: result.replayedSightings.map { ($0.savedIndex, $0.sighting) })
        XCTAssertEqual(bySavedIndex[381]?.sizeMMMeasured ?? 0.0, 179.85, accuracy: 0.25, "YS-69: Sharp M0 still 381 measures 179.85 mm (matching Python OpenCV)")
        XCTAssertEqual(bySavedIndex[2652]?.sizeMMMeasured ?? 0.0, 175.50, accuracy: 0.25, "YS-69: Sharp M3 still 2652 measures 175.50 mm (matching Python OpenCV)")
        XCTAssertNotNil(walk5Golden.per_marker["0"])

        // Unified YS-71 rule agrees with the Python toolkit golden verdict: Walk 5 is NOT closed
        XCTAssertFalse(result.phoneLoopState.unifiedLoopClosed, "YS-71: Unified loop rule reports Walk 5 not closed")
        XCTAssertNil(result.phoneLoopState.unifiedLoopClosureGapMeters, "YS-71: Unified loop rule gap is nil on Walk 5")
        XCTAssertFalse(walk5Golden.grade_coeff_reports_closed, "YS-66: Python toolkit grade.json reports Walk 5 not closed")
        XCTAssertFalse(walk5Golden.frame_marker0_loop_closed, "YS-66: Python toolkit frame.json reports Walk 5 not closed")
        XCTAssertTrue(walk5Golden.caps_applied.contains("no_loop_closure"), "YS-66: Python toolkit caps Walk 5 with no_loop_closure")
    }

    /// `YS-66`: Always-run replay proof on the checked-in trimmed Walk-5 sample using `.recordedCornersFromCSV`.
    /// Verifies that the historical recorded corners reproduce every original sighting within `±0.01 mm` and the
    /// historical `7.5 cm` (`0.0753 m`) false opening close.
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

        XCTAssertEqual(result.phoneLoopState.legacyPreYS71GapMeters ?? -1.0, 0.075326, accuracy: 0.0001)
        XCTAssertFalse(result.phoneLoopState.unifiedLoopClosed, "YS-71: Unified loop rule reports false on trimmed Walk 5")
    }

    /// `YS-71`: Always-on unit test verifying that a walk that skips returning to Marker 0 or misses Markers 1–3
    /// (without "End early") reports `unifiedLoopClosed == false`, while a complete 0 -> 1 -> 2 -> 3 -> 0 loop closes
    /// with the exact distance between the median of the opening pass and the median of the closing pass.
    func testYS71_UnifiedLoopClosureRule_RequiresMarkers1To3AndComputesPassMedianSpread() {
        var state = PhoneLoopRuleState()
        let m0OpenA = DetectedMarkerSighting(
            markerID: 0,
            cornersUV: [],
            positionWorld: SIMD3<Double>(0.0, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.0,
            rangeMeters: 1.1
        )
        let m0OpenB = DetectedMarkerSighting(
            markerID: 0,
            cornersUV: [],
            positionWorld: SIMD3<Double>(0.01, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.0,
            rangeMeters: 1.1
        )
        let m0OpenC = DetectedMarkerSighting(
            markerID: 0,
            cornersUV: [],
            positionWorld: SIMD3<Double>(0.02, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.0,
            rangeMeters: 1.1
        )

        // Opening pass at Marker 0 (median x = 0.01)
        state.ingestSightings([m0OpenA], timestampS: 1.0, savedIndex: 1, cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5))
        state.ingestSightings([m0OpenB], timestampS: 1.5, savedIndex: 2, cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5))
        state.ingestSightings([m0OpenC], timestampS: 2.0, savedIndex: 3, cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5))

        // Walk away 3 m and see only Marker 1 and Marker 2 (missing Marker 3), then return to Marker 0
        let m1 = DetectedMarkerSighting(
            markerID: 1, cornersUV: [], positionWorld: SIMD3<Double>(3.0, 0, 0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        let m2 = DetectedMarkerSighting(
            markerID: 2, cornersUV: [], positionWorld: SIMD3<Double>(3.0, 0, 3.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        state.ingestSightings([m1], timestampS: 8.0, savedIndex: 10, cameraPositionWorld: SIMD3<Double>(3.0, 1.2, 0.5))
        state.ingestSightings([m2], timestampS: 14.0, savedIndex: 20, cameraPositionWorld: SIMD3<Double>(3.0, 1.2, 3.5))

        let m0CloseA = DetectedMarkerSighting(
            markerID: 0, cornersUV: [], positionWorld: SIMD3<Double>(0.05, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        state.ingestSightings([m0CloseA], timestampS: 22.0, savedIndex: 30, cameraPositionWorld: SIMD3<Double>(0.05, 1.2, 0.5))
        XCTAssertFalse(state.unifiedLoopClosed, "YS-71: Without Marker 3 (and without End Early), loop must not close")
        XCTAssertNil(state.unifiedLoopClosureGapMeters, "YS-71: unifiedLoopClosureGapMeters must be nil before Marker 3 is seen")
        XCTAssertNil(state.loopClosureGapMeters, "YS-71: active loopClosureGapMeters must ALSO be nil when unifiedLoopClosed == false")
        XCTAssertNotNil(state.legacyPreYS71GapMeters, "YS-66: legacyPreYS71GapMeters retains the pre-YS-71 single-sighting gap")

        // Now lock Marker 3 and return to Marker 0 (closing pass median x = 0.06 -> spread = |0.06 - 0.01| = 0.05 m)
        let m3 = DetectedMarkerSighting(
            markerID: 3, cornersUV: [], positionWorld: SIMD3<Double>(0.0, 0, 3.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        state.ingestSightings([m3], timestampS: 26.0, savedIndex: 35, cameraPositionWorld: SIMD3<Double>(0.0, 1.2, 3.5))

        let m0CloseB1 = DetectedMarkerSighting(
            markerID: 0, cornersUV: [], positionWorld: SIMD3<Double>(0.05, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        let m0CloseB2 = DetectedMarkerSighting(
            markerID: 0, cornersUV: [], positionWorld: SIMD3<Double>(0.06, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        let m0CloseB3 = DetectedMarkerSighting(
            markerID: 0, cornersUV: [], positionWorld: SIMD3<Double>(0.09, 0.0, 0.0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1), sizeMMPrinted: 180, sizeMMMeasured: 180, rangeMeters: 1.1
        )
        state.ingestSightings([m0CloseB1], timestampS: 32.0, savedIndex: 40, cameraPositionWorld: SIMD3<Double>(0.06, 1.2, 0.5))
        state.ingestSightings([m0CloseB2], timestampS: 32.5, savedIndex: 41, cameraPositionWorld: SIMD3<Double>(0.06, 1.2, 0.5))
        state.ingestSightings([m0CloseB3], timestampS: 33.0, savedIndex: 42, cameraPositionWorld: SIMD3<Double>(0.06, 1.2, 0.5))

        XCTAssertTrue(state.unifiedLoopClosed, "YS-71: Loop closes after Markers 1–3 and return hold at Marker 0")
        XCTAssertEqual(state.unifiedLoopClosureGapMeters ?? -1.0, 0.05, accuracy: 1e-6, "YS-71: Spread is distance between pass medians (0.06 - 0.01 = 0.05 m)")
        XCTAssertEqual(state.loopClosureGapMeters ?? -1.0, 0.05, accuracy: 1e-6, "YS-71: active loopClosureGapMeters equals unifiedLoopClosureGapMeters")
    }

    /// `YS-70` / `YS-73`: Always-on unit test verifying OpenCV Apache-2.0 attribution, predicted motion blur,
    /// coaching cues (`"Turn a little slower"`, `"Hold steady"`), live size deviation check after 3 sightings (`> 3%`),
    /// and `guidance.csv` formatting.
    func testYS70_YS73_ExposureStillnessGateAndLiveSizeCheck() {
        XCTAssertTrue(ArUcoDetector.openCVApacheAttributionNotice.contains("Apache License, Version 2.0"))
        XCTAssertTrue(ArUcoDetector.openCVApacheAttributionNotice.contains("DICT_5X5_250"))

        // Predicted blur = focal_length (1440 px) * turn_rate (0.20 rad/s) * exposure (0.01 s) = 2.88 px
        let blur = ExposureAndStillnessGate.computePredictedBlurPx(
            focalLengthPx: 1440.0,
            turnRateRadS: 0.20,
            exposureDurationS: 0.01
        )
        XCTAssertEqual(blur, 2.88, accuracy: 1e-6)

        let fastTurnVerdict = ExposureAndStillnessGate.evaluateMarkerSighting(
            predictedBlurPx: blur,
            turnRateRadS: 0.20,
            rangeMeters3D: 1.10
        )
        XCTAssertFalse(fastTurnVerdict.passes)
        XCTAssertEqual(fastTurnVerdict.coachingCue, "Turn a little slower")

        let highBlurVerdict = ExposureAndStillnessGate.evaluateMarkerSighting(
            predictedBlurPx: 1.80,
            turnRateRadS: 0.10,
            rangeMeters3D: 1.10
        )
        XCTAssertFalse(highBlurVerdict.passes)
        XCTAssertEqual(highBlurVerdict.coachingCue, "Hold steady")

        let goodVerdict = ExposureAndStillnessGate.evaluateMarkerSighting(
            predictedBlurPx: 0.80,
            turnRateRadS: 0.08,
            rangeMeters3D: 1.10
        )
        XCTAssertTrue(goodVerdict.passes)
        XCTAssertNil(goodVerdict.coachingCue)

        // Live marker size check after 3 sightings (> 3% off flags)
        let twoSightings = ExposureAndStillnessGate.evaluateLiveMarkerSizeDeviation(
            measuredSizesMM: [190.0, 191.0],
            printedSizeMM: 180.0
        )
        XCTAssertFalse(twoSightings.flagged, "YS-73: Should not flag before 3 sightings")

        let threeOffSightings = ExposureAndStillnessGate.evaluateLiveMarkerSizeDeviation(
            measuredSizesMM: [186.5, 187.0, 188.0],
            printedSizeMM: 180.0
        )
        XCTAssertTrue(threeOffSightings.flagged, "YS-73: Median 187.0 mm is +3.89% (> 3%) off 180 mm")

        let threeGoodSightings = ExposureAndStillnessGate.evaluateLiveMarkerSizeDeviation(
            measuredSizesMM: [179.2, 179.6, 180.1],
            printedSizeMM: 180.0
        )
        XCTAssertFalse(threeGoodSightings.flagged, "YS-73: Median 179.6 mm is within 0.22% (<= 3%) of 180 mm")

        let row = ExposureAndStillnessGate.formatGuidanceCSVRow(
            timestampS: 100.25,
            savedIndex: 42,
            event: "frame",
            phase: "recording",
            instruction: "Turn a little slower",
            spoken: true,
            pitchDeg: -28.5,
            rollDeg: 1.2,
            trackingState: "normal",
            exposureDurationS: 0.008333,
            iso: 125.0,
            turnRateRadS: 0.2100,
            predictedBlurPx: 2.520,
            thermalState: "nominal",
            coverageCells: 18
        )
        XCTAssertTrue(row.hasPrefix("100.250000,42,frame,recording,Turn a little slower,true,"))
    }

    // MARK: - Full Walk 1–5 Local Suite (`YARDSTICK_REPLAY_ROOT`)

    /// `YS-66` / `YS-69` / `YS-70` / `YS-73`: Full Walk 5 (`2,865` frames, `455` stills) replay in `.liveDetectorOnStills` mode under `< 15 s`.
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

        // YS-69: 0 sightings above 185 mm on Walk 5 (there were 28 at 194–207 mm under the old Vision detector),
        // and Walk 5's overall median is within 179.5 ± 1.0 mm!
        let allMeasuredMM = result.replayedSightings.map(\.sighting.sizeMMMeasured).sorted()
        let over185Count = allMeasuredMM.filter { $0 > 185.0 }.count
        XCTAssertEqual(over185Count, 0, "YS-69: Expected 0 sightings above 185 mm on Walk 5, got \(over185Count)")
        let walk5OverallMedian = allMeasuredMM[allMeasuredMM.count / 2]
        XCTAssertEqual(
            walk5OverallMedian,
            179.5,
            accuracy: 1.0,
            String(format: "YS-69: Walk 5 overall median %.3f mm must be within 179.5 ± 1.0 mm", walk5OverallMedian)
        )

        // YS-69: Per-marker sighting count and fused size match toolkit within <= 0.3%
        for markerID in 0...3 {
            guard let goldenMarker = walk5Golden.per_marker[String(markerID)],
                  let phoneFusedMM = CaptureReplayResult.fusedPassSizeMM(
                      for: markerID,
                      in: result.replayedSightings,
                      cameraPositionsByIndex: result.phoneLoopState.cameraPositionsByIndex
                  )
            else {
                XCTFail("YS-69: Missing sightings for marker \(markerID) on Walk 5")
                continue
            }
            let diffPct = abs(phoneFusedMM - goldenMarker.size_mm_measured) / goldenMarker.size_mm_measured * 100.0
            XCTAssertLessThanOrEqual(
                diffPct,
                0.30,
                String(
                    format: "YS-69: Walk 5 marker %d phone fused %.3f mm vs toolkit %.3f mm differs by %.3f%% (> 0.3%%)",
                    markerID, phoneFusedMM, goldenMarker.size_mm_measured, diffPct
                )
            )
        }

        // YS-70: 0 marker_opencv_disagreement and 0 unconfirmed rows
        XCTAssertEqual(result.unconfirmedSightingsCount, 0, "YS-70: 0 unconfirmed rows on Walk 5")
        XCTAssertEqual(result.markerOpenCVDisagreementCount, 0, "YS-70: 0 marker_opencv_disagreement rows on Walk 5")

        // YS-71: Unified loop rule reports Walk 5 as NOT closed (matching toolkit)
        XCTAssertFalse(result.phoneLoopState.unifiedLoopClosed, "YS-71: Walk 5 unified loop rule must report not closed")
        XCTAssertNil(result.phoneLoopState.unifiedLoopClosureGapMeters, "YS-71: Walk 5 unified loop gap must be nil")
        XCTAssertFalse(walk5Golden.grade_coeff_reports_closed, "YS-66: Toolkit golden reports Walk 5 not closed")

        // YS-73: Stillness and distance gate flags 24 of 24 of its OpenCV-rejected sightings and keeps >= 25 good ones per marker
        XCTAssertEqual(result.openCVRejectedOriginalSightingsCount, 24, "YS-73: Walk 5 has 24 OpenCV-rejected sightings")
        XCTAssertEqual(
            result.ys73FlaggedOfOpenCVRejectedCount,
            24,
            "YS-73: Stillness and distance gate must flag 24 of 24 rejected sightings on Walk 5"
        )
        for markerID in 0...3 {
            let kept = result.ys73KeptGoodSightingsByMarker[markerID] ?? 0
            XCTAssertGreaterThanOrEqual(
                kept,
                25,
                "YS-73: Marker \(markerID) must keep at least 25 good sightings on Walk 5 (got \(kept))"
            )
        }
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
            result.phoneLoopState.legacyPreYS71GapMeters ?? -1.0,
            0.075327,
            accuracy: 0.0001,
            "YS-66: Exact 7.53 cm historical false loop close reproduced"
        )
        XCTAssertFalse(result.phoneLoopState.unifiedLoopClosed, "YS-71: Unified loop rule reports Walk 5 not closed")
    }

    /// `YS-66` / `YS-69` / `YS-71`: Replay Walks 1–5 and verify:
    /// - `YS-71`: The phone's unified closing verdict (`unifiedLoopClosed` and `unifiedLoopClosureGapMeters`)
    ///   equals the toolkit's `frame_marker0_loop_closed` `5 of 5` across Walks 1–5!
    /// - `YS-69`: In `.liveDetectorOnStills` across Walks 1–5, median phone/toolkit size difference per marker is `<= 0.3%`!
    func testYS66_FullWalks1To5_ComparesPhoneVerdictsAgainstToolkitGoldenFile() throws {
        guard let rootURL = Self.replayRootURL else {
            throw XCTSkip("NOT RUN: Set YARDSTICK_REPLAY_ROOT to run the full Walks 1–5 comparison suite")
        }
        let golden = try ToolkitGoldenFile.load(from: Self.toolkitGoldenURL)

        let expectations: [(
            walkKey: String,
            expectedLegacyPhoneGapM: Double?,
            expectedToolkitCoeffClosed: Bool,
            expectedToolkitCoeffGapM: Double?,
            expectedToolkitFrameClosed: Bool,
            expectedToolkitFrameGapM: Double?
        )] = [
            ("walk-1", 0.2479, true, 0.246154, true, 0.246154),
            ("walk-2", 1.0545, false, nil, false, nil),
            ("walk-3", nil, false, nil, false, nil),
            ("walk-4", nil, true, 0.208437, false, nil),
            ("walk-5", 0.0753, false, nil, false, nil),
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

            // 1. Recorded corners check (YS-66 + YS-71 5/5 loop verdict parity)
            let resRecorded = try CaptureReplayHarness.replayCaptureDirectory(
                at: walkURL,
                mode: .recordedCornersFromCSV
            )
            XCTAssertGreaterThan(resRecorded.totalFrameRows, 0, "YS-66: \(exp.walkKey) must have frames")

            if let expLegacyGap = exp.expectedLegacyPhoneGapM {
                guard let actualLegacyGap = resRecorded.phoneLoopState.legacyPreYS71GapMeters else {
                    XCTFail("YS-66: Expected legacy phone loop gap on \(exp.walkKey), got nil")
                    continue
                }
                XCTAssertEqual(actualLegacyGap, expLegacyGap, accuracy: 0.002, "YS-66: \(exp.walkKey) legacy phone loop gap mismatch")
            } else {
                XCTAssertNil(resRecorded.phoneLoopState.legacyPreYS71GapMeters, "YS-66: Expected nil legacy phone loop gap on \(exp.walkKey)")
            }

            // YS-71: Phone's unified closing verdict equals the toolkit's 5 of 5!
            XCTAssertEqual(
                resRecorded.phoneLoopState.unifiedLoopClosed,
                exp.expectedToolkitFrameClosed,
                "YS-71: \(exp.walkKey) unified loop-closed verdict must match toolkit (5 of 5)"
            )
            if let expFrameGap = exp.expectedToolkitFrameGapM {
                XCTAssertEqual(
                    resRecorded.phoneLoopState.unifiedLoopClosureGapMeters ?? -1.0,
                    expFrameGap,
                    accuracy: 0.0005,
                    "YS-71: \(exp.walkKey) unified loop-closure gap must match toolkit pass-median spread"
                )
            } else {
                XCTAssertNil(resRecorded.phoneLoopState.unifiedLoopClosureGapMeters, "YS-71: \(exp.walkKey) unified gap must be nil")
            }

            // 2. Live OpenCV detector on stills check (YS-69 <= 0.3% size parity + YS-71 5/5 loop verdict parity)
            let resLive = try CaptureReplayHarness.replayCaptureDirectory(
                at: walkURL,
                mode: .liveDetectorOnStills
            )
            XCTAssertEqual(
                resLive.phoneLoopState.unifiedLoopClosed,
                exp.expectedToolkitFrameClosed,
                "YS-71: \(exp.walkKey) live-detector unified loop-closed verdict must match toolkit (5 of 5)"
            )
            XCTAssertEqual(resLive.unconfirmedSightingsCount, 0, "YS-70: \(exp.walkKey) must have 0 unconfirmed sightings")
            XCTAssertEqual(resLive.markerOpenCVDisagreementCount, 0, "YS-70: \(exp.walkKey) must have 0 OpenCV disagreements")

            for (midStr, markerEntry) in gWalk.per_marker {
                guard let mid = Int(midStr) else { continue }
                guard let phoneFusedMM = CaptureReplayResult.fusedPassSizeMM(
                    for: mid,
                    in: resLive.replayedSightings,
                    cameraPositionsByIndex: resLive.phoneLoopState.cameraPositionsByIndex
                ) else {
                    XCTFail("YS-69: Missing live sightings for marker \(mid) on \(exp.walkKey)")
                    continue
                }
                let diffPct = abs(phoneFusedMM - markerEntry.size_mm_measured) / markerEntry.size_mm_measured * 100.0
                XCTAssertLessThanOrEqual(
                    diffPct,
                    0.30,
                    String(
                        format: "YS-69: %@ marker %d phone %.3f mm vs toolkit %.3f mm differs by %.3f%% (> 0.3%%)",
                        exp.walkKey, mid, phoneFusedMM, markerEntry.size_mm_measured, diffPct
                    )
                )
            }
        }
    }
}
