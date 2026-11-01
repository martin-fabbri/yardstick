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

    /// `YS-74`: Always-on unit test verifying the deterministic `GuidedCaptureEngine` state machine:
    /// - `0.7 s` debounce and `1.5 s` minimum dwell,
    /// - 3-still lock rule (`>= 1.0 s` span, `trackingState == "normal"`, `YS-73` gates met, viewing angle `<= 40°`,
    ///   3D position spread `<= 2.5 cm`, size within `±3%` of printed),
    /// - pauses rather than resets when a condition breaks,
    /// - rising haptic tick (`0.5 -> 0.7 -> 0.9`), `nil` haptic on corrections, `nil` haptic during voice note recording,
    /// - Markers 1–3 can be locked in any order (`2 -> 1 -> 3`),
    /// - `"Fix now"` (`requestFixNow`) requires Marker 0 to be locked again at the end before `Finish` becomes available.
    func testYS74_GuidedCaptureEngine_3StillLockPauseHapticsAndFixNowReturnToMarker0() {
        var engine = GuidedCaptureEngine(requireLookUpStep: true)
        XCTAssertEqual(engine.phase, .openingMarker0)
        XCTAssertEqual(engine.activeInstruction, "Stand over marker 0 and point at it")
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: false))

        func makeSighting(
            id: Int,
            pos: SIMD3<Double> = SIMD3<Double>(0, 0, 0),
            printedMM: Double = 180.0,
            measuredMM: Double = 179.8,
            angleDeg: Double = 22.0,
            rangeM: Double = 1.05
        ) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(900, 660),
                    SIMD2<Double>(1020, 660),
                    SIMD2<Double>(1020, 780),
                    SIMD2<Double>(900, 780),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: printedMM,
                sizeMMMeasured: measuredMM,
                rangeMeters: rangeM,
                viewingAngleDeg: angleDeg
            )
        }

        let camAtM0 = SIMD3<Double>(0, 1.2, 0.5)

        // 1. First confirmed still on Marker 0 at t = 1.0 s -> segment 1 confirmed, haptic intensity 0.5
        let s1 = engine.step(
            timestampS: 1.0,
            savedIndex: 10,
            dt: 0.033,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.0, 0.0, 0.0))],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.0, 0.0, 0.0))],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.activeRingState?.confirmedSegments, 1)
        XCTAssertEqual(engine.activeRingState?.isPaused, false)
        XCTAssertEqual(s1.haptic, .holdSegmentConfirmed(markerID: 0, segment: 1, intensity: 0.5))

        // 1b. Non-detection 30 Hz frame (`didRunMarkerDetection: false`) 33 ms later preserves `activeRingState`
        // and keeps `thermalSchedule.detectionHz == 10.0` even under `.serious` thermal state (`YS-74`, `YS-85`)
        let sNonDet = engine.step(
            timestampS: 1.033,
            savedIndex: 11,
            dt: 0.033,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [],
            rawSightings: [],
            unifiedLoopClosed: false,
            thermalLevel: .serious,
            didRunMarkerDetection: false
        )
        XCTAssertEqual(engine.activeRingState?.confirmedSegments, 1, "YS-74: 30 Hz non-detection frame must not clear activeRingState")
        XCTAssertEqual(sNonDet.thermalSchedule.detectionHz, 10.0, "YS-85: Serious thermal detection stays at 10 Hz during an active hold across 30 Hz frames")
        XCTAssertEqual(sNonDet.thermalSchedule.detectionFrameStrideAt30FPS, 3)

        // 2. Condition breaks at t = 1.2 s (fast turn / blur) -> ring PAUSES at 1 segment (does NOT reset to 0!)
        // and correction does NOT emit a haptic
        let sPause = engine.step(
            timestampS: 1.2,
            savedIndex: 16,
            dt: 0.20,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.35,
            predictedBlurPx: 3.2,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.0, 0.0, 0.0))],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.activeRingState?.confirmedSegments, 1, "YS-74: Hold must pause rather than reset when a condition breaks")
        XCTAssertEqual(engine.activeRingState?.isPaused, true)
        XCTAssertNil(sPause.haptic, "YS-74: Corrections must never trigger a haptic")

        // 3. Second confirmed still at t = 1.6 s -> segment 2 confirmed, haptic intensity 0.7
        let s2 = engine.step(
            timestampS: 1.6,
            savedIndex: 28,
            dt: 0.40,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.01, 0.0, 0.0))],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.01, 0.0, 0.0))],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.activeRingState?.confirmedSegments, 2)
        XCTAssertEqual(engine.activeRingState?.isPaused, false)
        XCTAssertEqual(s2.haptic, .holdSegmentConfirmed(markerID: 0, segment: 2, intensity: 0.7))

        // 4. Reject a sample whose viewing angle > 40° or size > ±3% off -> stays at 2 segments
        _ = engine.step(
            timestampS: 2.0,
            savedIndex: 40,
            dt: 0.40,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.01, 0.0, 0.0), measuredMM: 190.0, angleDeg: 45.0)],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.01, 0.0, 0.0), measuredMM: 190.0, angleDeg: 45.0)],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.activeRingState?.confirmedSegments, 2, "YS-74: Sample failing 40° viewing angle or ±3% size must not advance hold")
        XCTAssertFalse(engine.lockedMarkerIDs.contains(0))

        // 5. Third confirmed still at t = 2.2 s (span = 1.2 s >= 1.0 s, 3D spread <= 2.5 cm) -> locks Marker 0!
        let s3 = engine.step(
            timestampS: 2.2,
            savedIndex: 46,
            dt: 0.20,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: camAtM0,
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.015, 0.0, 0.0))],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.015, 0.0, 0.0))],
            unifiedLoopClosed: false
        )
        XCTAssertTrue(engine.lockedMarkerIDs.contains(0))
        XCTAssertEqual(engine.phase, .markers1To3)
        XCTAssertEqual(s3.haptic, .markerLocked(markerID: 0))
        XCTAssertEqual(engine.activeInstruction, "Got marker 0. Every marker works like that.")

        // 6. Verify voice-note recording suppresses all haptics (YS-74)
        engine.setRecordingVoiceNote(true, at: 3.9)
        let sMuted = engine.step(
            timestampS: 4.0,
            savedIndex: 100,
            dt: 0.033,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.04,
            predictedBlurPx: 0.5,
            cameraPositionWorld: SIMD3<Double>(4.0, 1.2, 0.5),
            confirmedSightings: [makeSighting(id: 2, pos: SIMD3<Double>(4.0, 0, 0))],
            rawSightings: [makeSighting(id: 2, pos: SIMD3<Double>(4.0, 0, 0))],
            unifiedLoopClosed: false
        )
        XCTAssertNil(sMuted.haptic, "YS-74: No haptics may fire while a voice note is recording")
        engine.setRecordingVoiceNote(false, at: 4.1)

        // 7. Lock Markers 2, 1, 3 in out-of-sequence order (2 -> 1 -> 3)
        var t = 4.5
        var sIdx = 115
        for mid in [2, 1, 3] {
            let mPos = SIMD3<Double>(Double(mid) * 2.0, 0, 0)
            let cPos = SIMD3<Double>(Double(mid) * 2.0, 1.2, 0.5)
            for k in 0..<3 {
                t += 0.55
                sIdx += 16
                _ = engine.step(
                    timestampS: t,
                    savedIndex: sIdx,
                    dt: 0.55,
                    trackingState: "normal",
                    pitchDeg: -30.0,
                    turnRateRadS: 0.03,
                    predictedBlurPx: 0.4,
                    cameraPositionWorld: cPos,
                    confirmedSightings: [makeSighting(id: mid, pos: mPos + SIMD3<Double>(Double(k) * 0.005, 0, 0))],
                    rawSightings: [makeSighting(id: mid, pos: mPos)],
                    unifiedLoopClosed: false
                )
            }
            XCTAssertTrue(engine.lockedMarkerIDs.contains(mid), "YS-74: Marker \(mid) must lock after 3 confirmed stills spanning >= 1.0 s")
        }

        // After Markers 1–3 are locked, phase transitions to `.houseWallLookUp` (YS-79)
        XCTAssertEqual(engine.phase, .houseWallLookUp)
        engine.skipLookUpStep(at: t + 0.5)
        XCTAssertEqual(engine.phase, .closingMarker0)
        XCTAssertEqual(engine.activeInstruction, "Head back to marker 0")
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: false))

        // 8. Lock Marker 0 on return -> transitions to `.readyToFinish` ("All done. Tap Finish.")
        var lastHaptic: GuidedHapticEvent? = nil
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            let res = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.03,
                predictedBlurPx: 0.4,
                cameraPositionWorld: camAtM0,
                confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(Double(k) * 0.004, 0, 0))],
                rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0, 0, 0))],
                unifiedLoopClosed: k == 2
            )
            lastHaptic = res.haptic
        }
        XCTAssertTrue(engine.isClosingMarker0Locked)
        XCTAssertTrue(engine.isFinishAvailable(unifiedLoopClosed: true))
        XCTAssertEqual(engine.phase, .readyToFinish)
        XCTAssertEqual(engine.activeInstruction, "All done. Tap Finish.")
        XCTAssertEqual(lastHaptic, .loopClosedPathComplete)

        // 9. Trigger "Fix now" on Marker 2 -> unlocks Marker 2 AND requires closing Marker 0 to be locked again!
        engine.beginFixNowRequiringReturnToMarker0(targetMarkerID: 2, at: t + 1.0)
        XCTAssertFalse(engine.lockedMarkerIDs.contains(2))
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: true), "YS-74: Fix now must require re-locking Marker 2 and Marker 0 at the end")
        XCTAssertEqual(engine.phase, .markers1To3)
        XCTAssertEqual(engine.activeInstruction, "Walk to marker 2")

        // 10. Adversarial regression checks:
        // (a) Pairwise 3D spread > 2.5 cm (e.g., [-0.02, 0.0, +0.02] m -> 4.0 cm pairwise, 2.0 cm from median) must exceed 0.025 m!
        let spreadSamples: [ConfirmedMarkerStillSample] = [
            ConfirmedMarkerStillSample(timestampS: 10.0, savedIndex: 1, sighting: makeSighting(id: 2, pos: SIMD3<Double>(-0.020, 0, 0))),
            ConfirmedMarkerStillSample(timestampS: 10.5, savedIndex: 2, sighting: makeSighting(id: 2, pos: SIMD3<Double>(0.000, 0, 0))),
            ConfirmedMarkerStillSample(timestampS: 11.1, savedIndex: 3, sighting: makeSighting(id: 2, pos: SIMD3<Double>(0.020, 0, 0))),
        ]
        XCTAssertGreaterThan(MarkerHoldProgress.maxSpreadFromMedianM(spreadSamples), 0.025, "YS-74: Pairwise 3D spread of 4.0 cm must exceed 2.5 cm gate")

        // (b) Portrait coordinate rotation for landscape-right 1920x1440 sensor buffers (`YS-74` HUD ring):
        let sensorRing = MarkerRingVisualState(
            markerID: 0,
            isClosingHold: false,
            confirmedSegments: 1,
            isPaused: false,
            pauseReason: nil,
            showStepCloserChevron: false,
            isLocked: false,
            positionWorld: .zero,
            cornersUV: [SIMD2<Double>(480.0, 360.0)],
            rangeMeters: 1.0
        )
        let normPt = sensorRing.normalizedPortraitCenter(rgbWidth: 1920.0, rgbHeight: 1440.0)
        XCTAssertEqual(normPt.x, 0.75, accuracy: 1e-4, "YS-74: Portrait X must be 1.0 - (v / 1440)")
        XCTAssertEqual(normPt.y, 0.25, accuracy: 1e-4, "YS-74: Portrait Y must be u / 1920")
    }

    /// `YS-78`: Always-on unit test verifying the `0.25 m` `GroundCoverageGrid` filters depth to
    /// `range 0.5–4.0 m`, within `±0.25 m` of the marker plane, surface normal within `35°` of vertical,
    /// and matches toolkit ground cells `>= 90%`.
    func testYS78_GroundCoverageGrid_FiltersRangeHeightNormalAndMatchesToolkitCellsAtLeast90Percent() throws {
        var grid = GroundCoverageGrid(cellSizeM: 0.25)
        let marker0Origin = SIMD3<Double>(0.10, 0.0, -0.15)
        let marker0XAxis = SIMD3<Double>(cos(0.25), 0.0, sin(0.25))
        grid.anchorGroundPlaneY(0.0)
        grid.anchorMarker0Frame(originWorld: marker0Origin, xAxisWorld: marker0XAxis)

        // Create a synthetic 64x48 depth map where the camera is at (0, 1.30, 0) pitched down -30°
        let dW = 64
        let dH = 48
        let rgbW = 1920
        let rgbH = 1440
        let fx = 1440.0
        let fy = 1440.0
        let cx = 960.0
        let cy = 720.0

        // Camera pitched -30° around X axis at height y = 1.30 m
        let pitchRad = -30.0 * Double.pi / 180.0
        let c = cos(pitchRad)
        let s = sin(pitchRad)
        let camToWorld = simd_double4x4(
            SIMD4<Double>(1, 0, 0, 0),
            SIMD4<Double>(0, c, s, 0),
            SIMD4<Double>(0, -s, c, 0),
            SIMD4<Double>(0, 1.30, 0, 1)
        )

        let scaleX = Double(dW) / Double(rgbW)
        let scaleY = Double(dH) / Double(rgbH)
        let fxD = fx * scaleX
        let fyD = fy * scaleY
        let cxD = cx * scaleX
        let cyD = cy * scaleY

        var depthMM = [UInt16](repeating: 0, count: dW * dH)
        let confMap = [UInt8](repeating: 2, count: dW * dH)
        var toolkitGroundSamples: [(positionWorld: SIMD3<Double>, confidence: UInt8, rangeM: Double)] = []

        for v in 0..<dH {
            for u in 0..<dW {
                // Ray in camera space through (u, v): dirCam = ((u+0.5-cxD)/fxD, -(v+0.5-cyD)/fyD, -1)
                let rx = (Double(u) + 0.5 - cxD) / fxD
                let ry = -(Double(v) + 0.5 - cyD) / fyD
                let rz = -1.0
                let dirW4 = camToWorld * SIMD4<Double>(rx, ry, rz, 0.0)
                let dirW = SIMD3<Double>(dirW4.x, dirW4.y, dirW4.z)
                guard dirW.y < -0.05 else { continue }
                // Intersect with flat ground plane y = 0: 1.30 + tRay * dirW.y = 0 -> tRay = -1.30 / dirW.y
                let zCamDepth = -1.30 / dirW.y
                guard zCamDepth > 0.3 && zCamDepth < 5.0 else { continue }
                depthMM[v * dW + u] = UInt16(clamping: Int((zCamDepth * 1000.0).rounded()))

                let ptW = SIMD3<Double>(0, 1.30, 0) + zCamDepth * dirW
                let rng = simd_length(SIMD3<Double>(rx * zCamDepth, ry * zCamDepth, -zCamDepth))
                if rng >= 0.50 && rng <= 4.00 && abs(ptW.y) <= 0.25 && u >= 4 && u < dW - 4 && v >= 4 && v < dH - 4 && (u % 4 == 0) && (v % 4 == 0) {
                    toolkitGroundSamples.append((positionWorld: ptW, confidence: 2, rangeM: rng))
                }
            }
        }

        let stats = grid.ingestDepthFrame(
            timestampS: 1.0,
            depthMapMM: depthMM,
            confidenceMap: confMap,
            depthWidth: dW,
            depthHeight: dH,
            rgbWidth: rgbW,
            rgbHeight: rgbH,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            cameraToWorld: camToWorld,
            sampleStep: 4
        )
        XCTAssertGreaterThan(stats.addedCells, 10, "YS-78: Flat ground plane must populate 0.25 m coverage cells")
        XCTAssertGreaterThan(grid.coveredAreaM2, 0.5)

        let agreement = GroundCoverageGrid.computeCellAgreementWithToolkitGroundPoints(
            worldGroundPoints: toolkitGroundSamples,
            groundYWorld: 0.0,
            originWorld: marker0Origin,
            xAxisWorld: marker0XAxis,
            ingestedGrid: grid
        )
        XCTAssertGreaterThanOrEqual(
            agreement.agreementRatio,
            0.90,
            String(format: "YS-78: Phone ground grid must match toolkit cells >= 90%% (got %.1f%%)", agreement.agreementRatio * 100.0)
        )
        let gridVsToolkitUnion = max(1, grid.coveredCells.union(agreement.toolkitCells).count)
        let gridVsToolkitRatio = Double(grid.coveredCells.intersection(agreement.toolkitCells).count) / Double(gridVsToolkitUnion)
        XCTAssertGreaterThanOrEqual(
            gridVsToolkitRatio,
            0.90,
            String(format: "YS-78: Ingested depth grid cells must match toolkit cells >= 90%% (got %.1f%%)", gridVsToolkitRatio * 100.0)
        )

        // Verify out-of-height plane (y = +0.80 m, > 0.25 m above marker plane) adds 0 ground cells
        var highGrid = GroundCoverageGrid(cellSizeM: 0.25)
        highGrid.anchorGroundPlaneY(0.80)
        let highStats = highGrid.ingestDepthFrame(
            timestampS: 2.0,
            depthMapMM: depthMM,
            confidenceMap: confMap,
            depthWidth: dW,
            depthHeight: dH,
            rgbWidth: rgbW,
            rgbHeight: rgbH,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            cameraToWorld: camToWorld,
            sampleStep: 4
        )
        XCTAssertEqual(highStats.addedCells, 0, "YS-78: Surface > 0.25 m from marker ground plane must be excluded")
    }

    /// `YS-79`: Always-on unit test verifying the House-Wall Look-Up step:
    /// - tilting `+15°` to `+35°` with valid depth for `>= 3.0 s` and points `> 2 m` above ground completes the step,
    /// - tilting above `+45°` with no depth warns `"Point at the roof edge, not the sky"`,
    /// - one-tap `Skip` transitions to `"Head back to marker 0"`.
    func testYS79_HouseWallLookUpStep_Accumulates3SecondsWarnsOnSkyAndLogsSkipToGuidanceCSV() {
        var tracker = HouseWallLookUpTracker()
        XCTAssertFalse(tracker.isDone)

        // 1. Tilt above +45° (+52°) with no valid depth -> warns "Point at the roof edge, not the sky"
        tracker.update(
            dt: 0.5,
            pitchDeg: 52.0,
            hasValidDepth: false,
            elevatedPointsAbove2m: 0,
            trackingState: "normal",
            trackingReason: ""
        )
        XCTAssertTrue(tracker.skyWarningActive)
        XCTAssertEqual(tracker.accumulatedValidDurationS, 0.0, accuracy: 1e-6)

        // 2. Tilt within +15°..+35° (+24°) with valid depth and elevated points > 2 m for 20 x 0.20 s = 4.0 s -> completes!
        for _ in 0..<20 {
            tracker.update(
                dt: 0.20,
                pitchDeg: 24.0,
                hasValidDepth: true,
                elevatedPointsAbove2m: 14,
                trackingState: "normal",
                trackingReason: ""
            )
        }
        XCTAssertFalse(tracker.skyWarningActive)
        XCTAssertTrue(tracker.isCompleted, "YS-79: >= 3.0 s of +24° tilt with valid depth must complete Look-Up step")
        XCTAssertTrue(tracker.isDone)
        XCTAssertGreaterThan(tracker.elevatedPointsAbove2mCount, 0, "YS-79: Points > 2 m above ground must be recorded")
        XCTAssertEqual(tracker.insufficientFeaturesLossCount, 0, "YS-79: 0 tracking losses for insufficient features")

        // 3. Verify GuidedCaptureEngine one-tap Skip transitions to .closingMarker0 and formats guidance.csv row
        var engine = GuidedCaptureEngine(requireLookUpStep: true)
        engine.skipLookUpStep(at: 50.0)
        XCTAssertTrue(engine.lookUpTracker.isSkipped)
        let skipRow = ExposureAndStillnessGate.formatGuidanceCSVRow(
            timestampS: 50.0,
            savedIndex: 500,
            event: "lookup_skipped",
            phase: "house_wall_lookup",
            instruction: "Head back to marker 0",
            spoken: false,
            pitchDeg: 10.0,
            rollDeg: 0.0,
            trackingState: "normal",
            exposureDurationS: 0.008,
            iso: 100.0,
            turnRateRadS: 0.0,
            predictedBlurPx: 0.0,
            thermalState: "nominal",
            coverageCells: 20
        )
        XCTAssertTrue(skipRow.contains(",lookup_skipped,house_wall_lookup,Head back to marker 0,"), "YS-79: One-tap Skip must format into guidance.csv")
    }

    /// `YS-85`: Always-on unit test verifying thermal adaptation at `.serious` thermal state:
    /// - detection drops from `10 Hz` to `5 Hz` when no marker is being held (`10 Hz` during a hold),
    /// - ground coverage drops from `2.0 Hz` to `0.5 Hz`,
    /// - animations become static (`animationsStatic == true`),
    /// - RGB and depth recording at `30 Hz` is never reduced (`recordingFPS == 30`).
    func testYS85_ThermalAdaptation_DropsDetectionTo5HzCoverageTo0_5HzAndStaticAnimationsWhileKeeping30HzRecording() {
        let nominal = ThermalAdaptationSchedule.evaluate(thermalLevel: .nominal, isMarkerHoldActive: false)
        XCTAssertEqual(nominal.detectionHz, 10.0)
        XCTAssertEqual(nominal.detectionFrameStrideAt30FPS, 3)
        XCTAssertEqual(nominal.coverageHz, 2.0)
        XCTAssertEqual(nominal.coverageMinIntervalS, 0.5)
        XCTAssertFalse(nominal.animationsStatic)
        XCTAssertEqual(nominal.recordingFPS, 30)

        let seriousIdle = ThermalAdaptationSchedule.evaluate(thermalLevel: .serious, isMarkerHoldActive: false)
        XCTAssertEqual(seriousIdle.detectionHz, 5.0, "YS-85: Detection drops from 10 Hz to 5 Hz outside holds at serious thermal")
        XCTAssertEqual(seriousIdle.detectionFrameStrideAt30FPS, 6)
        XCTAssertEqual(seriousIdle.coverageHz, 0.5, "YS-85: Coverage drops from 2 Hz to 0.5 Hz at serious thermal")
        XCTAssertEqual(seriousIdle.coverageMinIntervalS, 2.0)
        XCTAssertTrue(seriousIdle.animationsStatic, "YS-85: Animations become static at serious thermal")
        XCTAssertEqual(seriousIdle.recordingFPS, 30, "YS-85: 30 Hz recording is never reduced")

        let seriousHolding = ThermalAdaptationSchedule.evaluate(thermalLevel: .serious, isMarkerHoldActive: true)
        XCTAssertEqual(seriousHolding.detectionHz, 10.0, "YS-85: Detection stays at 10 Hz during an active marker hold")
        XCTAssertEqual(seriousHolding.detectionFrameStrideAt30FPS, 3)
        XCTAssertEqual(seriousHolding.coverageHz, 0.5)
        XCTAssertTrue(seriousHolding.animationsStatic)
        XCTAssertEqual(seriousHolding.recordingFPS, 30)
    }

    /// `YS-75`: Always-on unit test verifying Voice Coach rules:
    /// - speaks fixed phrases only for `"Locked"`, phase changes, and non-counting corrections,
    /// - never speaks during hold counting (`hold_steady`, segments `1 -> 2 -> 3`),
    /// - speaks `"Hold steady."` on `hold_paused_moving` only after `>= 2.0 s` of pause,
    /// - speaks marker digits as words (`"zero"`, `"one"`, `"two"`, `"three"`),
    /// - suppresses all speech while a voice note is recording.
    func testYS75_VoiceCoach_SpeaksFixedPhrasesOnlyForLocksAndPhaseChangesNeverCountsAndSuppressesDuringVoiceNotes() {
        var engine = GuidedCaptureEngine(requireLookUpStep: true)

        func makeSighting(id: Int, pos: SIMD3<Double>, range: Double = 1.0) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(400, 300),
                    SIMD2<Double>(600, 300),
                    SIMD2<Double>(600, 500),
                    SIMD2<Double>(400, 500),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: 180.0,
                sizeMMMeasured: 180.0,
                rangeMeters: range,
                viewingAngleDeg: 15.0
            )
        }

        // 1. Initial frame speaks "Stand over marker zero and point at it."
        _ = engine.step(
            timestampS: 0.0,
            savedIndex: 0,
            dt: 0.033,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
            confirmedSightings: [],
            rawSightings: [],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.lastSpokenPhraseThisStep, "Stand over marker zero and point at it.")

        // 2. First confirmed still on Marker 0 -> glass shows "Hold steady", voice MUST BE SILENT (never speak counts!)
        _ = engine.step(
            timestampS: 1.0,
            savedIndex: 30,
            dt: 0.033,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
            confirmedSightings: [makeSighting(id: 0, pos: .zero)],
            rawSightings: [makeSighting(id: 0, pos: .zero)],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.activeInstruction, "Hold steady")
        XCTAssertNil(engine.lastSpokenPhraseThisStep, "YS-75: Hold counting must never speak")

        // 3. Second confirmed still -> still silent!
        _ = engine.step(
            timestampS: 1.5,
            savedIndex: 45,
            dt: 0.50,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.005, 0, 0))],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.005, 0, 0))],
            unifiedLoopClosed: false
        )
        XCTAssertNil(engine.lastSpokenPhraseThisStep, "YS-75: Segment 2 count must never speak")

        // 4. Third confirmed still locks Marker 0 -> speaks "Got it. Every marker works like that."
        _ = engine.step(
            timestampS: 2.1,
            savedIndex: 63,
            dt: 0.60,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
            confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.010, 0, 0))],
            rawSightings: [makeSighting(id: 0, pos: SIMD3<Double>(0.010, 0, 0))],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.lastSpokenPhraseThisStep, "Got it. Every marker works like that.")

        // 5. While recording a voice note, NO voice coach lines may be spoken
        engine.setRecordingVoiceNote(true, at: 3.0)
        XCTAssertNil(engine.lastSpokenPhraseThisStep, "YS-75: Starting a voice note must be silent")
        _ = engine.step(
            timestampS: 4.0,
            savedIndex: 120,
            dt: 1.0,
            trackingState: "normal",
            pitchDeg: -30.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(2, 1.2, 0.5),
            confirmedSightings: [],
            rawSightings: [],
            unifiedLoopClosed: false,
            thermalLevel: .serious
        )
        XCTAssertNil(engine.lastSpokenPhraseThisStep, "YS-75: Voice coach must remain completely silent while a note records")
        engine.setRecordingVoiceNote(false, at: 4.1)
        XCTAssertEqual(engine.activeInstruction, "Note added", "YS-75: Stopping a voice note must show 'Note added' on glass")
        XCTAssertNil(engine.lastSpokenPhraseThisStep, "YS-75: 'Note added' must remain silent on the voice coach")

        // 6. Verify fixed phrase catalog for marker words and once-per-walk warnings
        XCTAssertEqual(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "locked_marker_2"), "Got marker two.")
        XCTAssertEqual(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "walk_to_marker_3"), "Now walk to marker three.")
        XCTAssertEqual(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "return_marker_0"), "That's all of them. Head back to marker zero.")
        XCTAssertEqual(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "all_done_finish"), "You're back at the start. All done.")
        XCTAssertNil(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "hold_paused_moving", holdPausedDurationS: 1.2))
        XCTAssertEqual(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "hold_paused_moving", holdPausedDurationS: 2.1), "Hold steady.")
        XCTAssertNil(GuidedVoiceCatalog.spokenPhrase(forInstructionKey: "storage_low", alreadySpokenOnceKeys: ["storage_low"]))

        // 7. Verify VoiceOver announcement vs AVSpeechSynthesizer routing (YS-75)
        XCTAssertEqual(
            GuidedVoiceCatalog.resolveOutputRoute(
                phrase: "Got marker one.",
                isVoiceCoachEnabled: true,
                isRecordingVoiceNote: false,
                isVoiceOverRunning: true
            ),
            .voiceOverAnnouncement("Got marker one.")
        )
        XCTAssertEqual(
            GuidedVoiceCatalog.resolveOutputRoute(
                phrase: "Got marker two.",
                isVoiceCoachEnabled: true,
                isRecordingVoiceNote: false,
                isVoiceOverRunning: false
            ),
            .speechSynthesizer("Got marker two.")
        )
        XCTAssertNil(
            GuidedVoiceCatalog.resolveOutputRoute(
                phrase: "Got marker two.",
                isVoiceCoachEnabled: true,
                isRecordingVoiceNote: true,
                isVoiceOverRunning: false
            )
        )
    }

    /// `YS-76`: Always-on unit test verifying Pre-Flight Checklist and Stable-Frame Lens Smudge Detector:
    /// - `< 10 GB` free blocks start with `"Free up 10 GB to start."` (`YS-80` #4),
    /// - a thumb/smudge on the lens on a stable frame triggers the smudge warning card,
    /// - a clean lens at dusk while moving does NOT trigger the smudge warning card,
    /// - refusing Location falls back to the exposure probe and never blocks start.
    func testYS76_PreFlightChecklist_BlocksUnder10GBDetectsSmudgeOnStableFrameOnlyAndAllowsStartWhenLocationRefused() {
        // 1. Storage < 10 GB blocks Start Walk ("Free up 10 GB to start.")
        let lowSpaceReport = PreFlightChecklistEvaluator.evaluate(
            availableDiskBytes: 8_500_000_000,
            batteryLevelFraction: 0.85,
            thermalLevel: .nominal,
            exposureDurationS: 0.005,
            iso: 100.0,
            isLocationAuthorized: true
        )
        XCTAssertFalse(lowSpaceReport.canStartWalk, "YS-76: < 10 GB free must block Start Walk")
        XCTAssertEqual(lowSpaceReport.storageRow.state, .block)
        XCTAssertEqual(lowSpaceReport.storageRow.detail, "Free up 10 GB to start.")

        // 2. Thumb / heavy smudge on lens on a STABLE frame triggers the smudge card
        let w = 64
        let h = 48
        let thumbLuma = [UInt8](repeating: 42, count: w * h)
        let thumbProbe = LensSmudgeDetector.evaluate(
            lumaBytes: thumbLuma,
            width: w,
            height: h,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraVelocityMps: 0.01
        )
        XCTAssertTrue(thumbProbe.isStableFrame)
        XCTAssertTrue(thumbProbe.isSmudged, "YS-76: Uniform/blurred thumb frame on a stable camera must flag lens smudge")
        let thumbReport = PreFlightChecklistEvaluator.evaluate(
            availableDiskBytes: 20_000_000_000,
            batteryLevelFraction: 0.85,
            thermalLevel: .nominal,
            exposureDurationS: 0.010,
            iso: 200.0,
            isLocationAuthorized: true,
            lensProbe: thumbProbe
        )
        XCTAssertEqual(thumbReport.lensRow.state, .warn)
        XCTAssertEqual(thumbReport.lensRow.title, "Lens looks smudged")

        // 3. Clean lens at dusk WHILE MOVING (`turnRateRadS = 0.35`, `predictedBlurPx = 2.8`) must NOT trigger the smudge card
        let movingDuskProbe = LensSmudgeDetector.evaluate(
            lumaBytes: thumbLuma,
            width: w,
            height: h,
            turnRateRadS: 0.35,
            predictedBlurPx: 2.8,
            cameraVelocityMps: 0.60
        )
        XCTAssertFalse(movingDuskProbe.isStableFrame, "YS-76: Moving frame must not be classified as stable")
        XCTAssertFalse(movingDuskProbe.isSmudged, "YS-76: Clean lens at dusk while moving must never trigger the smudge card")

        // 4. Clean lens with sharp detail on a stable frame passes (`isSmudged == false`)
        var sharpLuma = [UInt8](repeating: 30, count: w * h)
        for v in 0..<h {
            for u in 0..<w where ((u / 2 + v / 2) % 2 == 0) {
                sharpLuma[v * w + u] = 210
            }
        }
        let sharpProbe = LensSmudgeDetector.evaluate(
            lumaBytes: sharpLuma,
            width: w,
            height: h,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2
        )
        XCTAssertTrue(sharpProbe.isStableFrame)
        XCTAssertFalse(sharpProbe.isSmudged)

        // 5. Refusing Location (`isLocationAuthorized == false`) falls back to the exposure probe and NEVER blocks Start Walk
        let refusedLocReport = PreFlightChecklistEvaluator.evaluate(
            availableDiskBytes: 18_000_000_000,
            batteryLevelFraction: 0.80,
            thermalLevel: .nominal,
            exposureDurationS: 0.055,
            iso: 1200.0,
            isLocationAuthorized: false
        )
        XCTAssertTrue(refusedLocReport.canStartWalk, "YS-76: Refusing Location must fall back to the exposure probe and never block")
        XCTAssertNotEqual(refusedLocReport.daylightRow.state, .block)
    }

    /// `YS-77`: Always-on unit test verifying Pre-Walk Centre-to-Centre Tape Measurement (`Marker 0 -> Marker 1`):
    /// - entered before the walk on the setup screen (`preWalkTapeMeters0To1`),
    /// - once both Marker 0 and Marker 1 are locked, automatically writes a standard `session.json["tape_checks"]` entry
    ///   from their locked 3D median positions with nothing typed during the walk,
    /// - at `4–8 m` spacing (`6.00 m`), residual is `<= 3.0 cm` (`1.2 cm`).
    func testYS77_PreWalkCentreToCentreTapeCheck_GeneratesSchemaValidEntryFromLockedMarker0And1WithResidualUnder3cm() {
        var engine = GuidedCaptureEngine(requireLookUpStep: true, preWalkTapeMeters0To1: 6.00)
        XCTAssertNil(engine.generatedMarker0To1TapeCheck, "YS-77: Tape check must not be generated before Markers 0 and 1 are locked")

        func makeSighting(id: Int, pos: SIMD3<Double>) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(400, 300),
                    SIMD2<Double>(600, 300),
                    SIMD2<Double>(600, 500),
                    SIMD2<Double>(400, 500),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: 180.0,
                sizeMMMeasured: 180.0,
                rangeMeters: 1.0,
                viewingAngleDeg: 14.0
            )
        }

        // Lock Marker 0 at (0, 0, 0)
        var t = 1.0
        var sIdx = 10
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
                confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(Double(k - 1) * 0.002, 0, 0))],
                rawSightings: [makeSighting(id: 0, pos: .zero)],
                unifiedLoopClosed: false
            )
        }
        XCTAssertTrue(engine.lockedMarkerIDs.contains(0))
        XCTAssertNil(engine.generatedMarker0To1TapeCheck, "YS-77: Still waiting for Marker 1 to lock")

        // Lock Marker 1 at (6.012, 0, 0) -> 6.012 m centre-to-centre (residual = 1.2 cm <= 3.0 cm)
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(6.012, 1.2, 0.5),
                confirmedSightings: [makeSighting(id: 1, pos: SIMD3<Double>(6.012 + Double(k - 1) * 0.002, 0, 0))],
                rawSightings: [makeSighting(id: 1, pos: SIMD3<Double>(6.012, 0, 0))],
                unifiedLoopClosed: false
            )
        }
        XCTAssertTrue(engine.lockedMarkerIDs.contains(1))
        guard let tapeEntry = engine.generatedMarker0To1TapeCheck else {
            XCTFail("YS-77: generatedMarker0To1TapeCheck must be populated as soon as Markers 0 and 1 are both locked")
            return
        }
        XCTAssertEqual(tapeEntry.id, 0)
        XCTAssertEqual(tapeEntry.role, "primary")
        XCTAssertEqual(tapeEntry.tapeM, 6.00, accuracy: 1e-6)
        XCTAssertEqual(tapeEntry.measuredM, 6.012, accuracy: 1e-4)
        XCTAssertLessThanOrEqual(tapeEntry.residualM, 0.030, "YS-77: Residual at 6 m spacing must be <= 3.0 cm")

        let dict = tapeEntry.toJSONDictionary()
        let requiredKeys: Set<String> = [
            "id", "role", "t_a_s", "t_b_s", "saved_index_a", "saved_index_b",
            "point_a_world_m", "point_b_world_m", "tape_m", "measured_m", "residual_m",
        ]
        XCTAssertEqual(Set(dict.keys), requiredKeys, "YS-77: Dictionary keys must match contract/session.schema.json:406-480")

        // When YARDSTICK_REPLAY_ROOT is set, also verify full Walk 5 replay with 180 mm marker size
        if let rootURL = Self.replayRootURL {
            let walk5URL = rootURL.appendingPathComponent("walk-5", isDirectory: true)
            if FileManager.default.fileExists(atPath: walk5URL.appendingPathComponent("frames.csv").path),
               let res = try? CaptureReplayHarness.replayCaptureDirectory(
                   at: walk5URL,
                   mode: .liveDetectorOnStills,
                   overridePrintedSizeMM: 180.0,
                   preWalkTapeMeters0To1: 7.456
               ),
               let walk5Tape = res.guidedEngine.generatedMarker0To1TapeCheck {
                XCTAssertEqual(walk5Tape.id, 0)
                XCTAssertEqual(walk5Tape.role, "primary")
                XCTAssertLessThanOrEqual(walk5Tape.residualM, 0.030, "YS-77: Walk 5 Marker 0->1 centre-to-centre residual must be <= 3.0 cm")
            }
        }
    }

    /// `YS-72`: Always-on unit test verifying conditional `Finish` and the live `Review Walk` sheet:
    /// - `Finish` stays unavailable until `YS-71` and Marker 0's closing hold both pass,
    /// - `WalkReviewEvaluator` caps displayed items at `<= 3` ordered worst first,
    /// - waiting `35 s` while the review sheet is open keeps the loop closed (`isFinishAvailable == true`),
    /// - any `"Fix now"` action (`targetMarkerID: 2` or `reopenLookUp: true`) immediately makes `Finish` unavailable
    ///   and requires a fresh 3-still closing hold at Marker 0 before saving.
    func testYS72_WalkReview_LimitsTo3ItemsWorstFirstRequiresReturnHoldAfterFixNowAndKeepsFinishUnavailableWithoutClose() {
        // 1. Verify WalkReviewEvaluator caps items at <= 3, worst first, when 6 conditions are present simultaneously
        let worstReport = WalkReviewEvaluator.evaluate(
            lockedMarkerIDs: [0, 1],
            isClosingMarker0Locked: false,
            unifiedLoopClosed: false,
            fixNowRequiresReturnToMarker0: false,
            endEarlyChosen: true,
            lookUpDone: false,
            lookUpSkipped: true,
            coveredGroundCellCount: 10,
            minExpectedCoverageCells: 36,
            missingCoverageNearMarkerID: 2,
            sizeMismatchMarkerID: 0,
            didGetDark: true
        )
        XCTAssertEqual(worstReport.verdictKind, .probablyWontBeAccepted)
        XCTAssertEqual(worstReport.headline, "This walk probably won't be accepted")
        XCTAssertTrue(worstReport.requiresSaveAnywayConfirmation)
        XCTAssertEqual(worstReport.items.count, 3, "YS-72: Review Walk must display at most 3 items")
        XCTAssertEqual(worstReport.items[0].id, "not_back_at_marker_0")
        XCTAssertEqual(worstReport.items[0].sentence, "You didn't finish at marker 0.")
        XCTAssertEqual(worstReport.items[0].buttonTitle, "Head Back")
        XCTAssertEqual(worstReport.items[1].id, "missing_marker_2")
        XCTAssertEqual(worstReport.items[1].sentence, "Marker 2 wasn't captured.")
        XCTAssertEqual(worstReport.items[1].buttonTitle, "Capture It")
        XCTAssertEqual(worstReport.items[2].id, "missing_marker_3")

        // 2. Complete a full 4-marker walk with closing hold at Marker 0
        var engine = GuidedCaptureEngine(requireLookUpStep: false)
        func makeSighting(id: Int, pos: SIMD3<Double>) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(400, 300),
                    SIMD2<Double>(600, 300),
                    SIMD2<Double>(600, 500),
                    SIMD2<Double>(400, 500),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: 180.0,
                sizeMMMeasured: 180.0,
                rangeMeters: 1.0,
                viewingAngleDeg: 12.0
            )
        }

        var t = 1.0
        var sIdx = 10
        let positions: [Int: SIMD3<Double>] = [
            0: .zero,
            1: SIMD3<Double>(5, 0, 0),
            2: SIMD3<Double>(5, 0, 5),
            3: SIMD3<Double>(0, 0, 5),
        ]
        for mid in [0, 1, 2, 3] {
            let p = positions[mid]!
            for k in 0..<3 {
                t += 0.55
                sIdx += 16
                _ = engine.step(
                    timestampS: t,
                    savedIndex: sIdx,
                    dt: 0.55,
                    trackingState: "normal",
                    pitchDeg: -30.0,
                    turnRateRadS: 0.02,
                    predictedBlurPx: 0.2,
                    cameraPositionWorld: p + SIMD3<Double>(0, 1.2, 0.5),
                    confirmedSightings: [makeSighting(id: mid, pos: p + SIMD3<Double>(Double(k - 1) * 0.002, 0, 0))],
                    rawSightings: [makeSighting(id: mid, pos: p)],
                    unifiedLoopClosed: false
                )
            }
        }
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: false), "YS-72: Finish must stay unavailable before closing hold")

        // Closing hold at Marker 0
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
                confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(Double(k - 1) * 0.002, 0, 0))],
                rawSightings: [makeSighting(id: 0, pos: .zero)],
                unifiedLoopClosed: k == 2
            )
        }
        XCTAssertTrue(engine.isFinishAvailable(unifiedLoopClosed: true))

        // 3. Simulate waiting 35 s on the live Review Walk sheet while recording continues at 30 Hz (YS-72)
        for _ in 0..<(35 * 30) {
            t += 1.0 / 30.0
            sIdx += 1
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 1.0 / 30.0,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.01,
                predictedBlurPx: 0.1,
                cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
                confirmedSightings: [],
                rawSightings: [],
                unifiedLoopClosed: true,
                didRunMarkerDetection: false
            )
        }
        XCTAssertTrue(engine.isFinishAvailable(unifiedLoopClosed: true), "YS-72: Waiting 35 s on Review Walk must keep loop closed and Finish available")

        // 4. Trigger "Fix now" for Marker 2 -> unlocks Marker 2 and requires a fresh return hold at Marker 0
        engine.beginFixNowRequiringReturnToMarker0(targetMarkerID: 2, at: t)
        XCTAssertFalse(engine.lockedMarkerIDs.contains(2))
        XCTAssertFalse(engine.isClosingMarker0Locked)
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: true), "YS-72: Fix now must immediately disable Finish")

        // Re-lock Marker 2
        let p2 = positions[2]!
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: p2 + SIMD3<Double>(0, 1.2, 0.5),
                confirmedSightings: [makeSighting(id: 2, pos: p2 + SIMD3<Double>(Double(k - 1) * 0.002, 0, 0))],
                rawSightings: [makeSighting(id: 2, pos: p2)],
                unifiedLoopClosed: false
            )
        }
        XCTAssertTrue(engine.lockedMarkerIDs.contains(2))
        XCTAssertFalse(engine.isFinishAvailable(unifiedLoopClosed: true), "YS-72: After re-locking Marker 2, Finish must STILL require a return hold at Marker 0")
        XCTAssertEqual(engine.phase, .closingMarker0)

        // Fresh return hold at Marker 0 re-enables Finish
        for k in 0..<3 {
            t += 0.55
            sIdx += 16
            _ = engine.step(
                timestampS: t,
                savedIndex: sIdx,
                dt: 0.55,
                trackingState: "normal",
                pitchDeg: -30.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(0, 1.2, 0.5),
                confirmedSightings: [makeSighting(id: 0, pos: SIMD3<Double>(Double(k - 1) * 0.002, 0, 0))],
                rawSightings: [makeSighting(id: 0, pos: .zero)],
                unifiedLoopClosed: true
            )
        }
        XCTAssertTrue(engine.isFinishAvailable(unifiedLoopClosed: true), "YS-72: Fresh return hold at Marker 0 after Fix now re-enables Finish")
    }

    /// `YS-80`: Always-on unit test verifying the native design wording table and all 4 signed corrections:
    /// 1. Wording knows only markers and coverage (`"near marker 2"` or `"in the middle of the yard"`, never `"by the shed"`),
    /// 2. Interruption reads `"Walk interrupted. Saved up to this point."`, never `"Walk paused"`,
    /// 3. Marker-size setting sets every marker ID uniformly,
    /// 4. Storage blocker reads `"Free up 10 GB to start."`,
    /// plus plain-language `CaptureDetailSummary` verdicts (`"Ready to share"`, `"Usable, with limits"`, `"Needs another walk"`).
    func testYS80_NativeDesignWordingAndFourSignedCorrections_ZeroAntiPatterns() {
        // Correction #1: Coverage hole wording knows only markers and coverage, never "by the shed"
        let nearMarkerReport = WalkReviewEvaluator.evaluate(
            lockedMarkerIDs: [0, 1, 2, 3],
            isClosingMarker0Locked: true,
            unifiedLoopClosed: true,
            fixNowRequiresReturnToMarker0: false,
            endEarlyChosen: false,
            lookUpDone: true,
            lookUpSkipped: false,
            coveredGroundCellCount: 20,
            minExpectedCoverageCells: 36,
            missingCoverageNearMarkerID: 2
        )
        XCTAssertEqual(nearMarkerReport.items.first?.sentence, "Some ground near marker 2 was missed.")
        XCTAssertFalse(nearMarkerReport.items.first?.sentence.contains("shed") ?? true)
        XCTAssertFalse(nearMarkerReport.mapAccessibilitySummary.contains("shed"))

        let middleYardReport = WalkReviewEvaluator.evaluate(
            lockedMarkerIDs: [0, 1, 2, 3],
            isClosingMarker0Locked: true,
            unifiedLoopClosed: true,
            fixNowRequiresReturnToMarker0: false,
            endEarlyChosen: false,
            lookUpDone: true,
            lookUpSkipped: false,
            coveredGroundCellCount: 20,
            minExpectedCoverageCells: 36,
            missingCoverageNearMarkerID: nil
        )
        XCTAssertEqual(middleYardReport.items.first?.sentence, "Some ground in the middle of the yard was missed.")
        XCTAssertFalse(middleYardReport.items.first?.sentence.contains("shed") ?? true)

        // Correction #2: Interruption reads "Walk interrupted. Saved up to this point.", never "Walk paused"
        var engine = GuidedCaptureEngine()
        engine.markWalkInterrupted(at: 12.5)
        XCTAssertEqual(engine.activeInstruction, "Walk interrupted. Saved up to this point.")
        XCTAssertNotEqual(engine.activeInstruction, "Walk paused")
        XCTAssertNil(engine.lastSpokenPhraseThisStep)

        // Correction #4: Storage blocker reads "Free up 10 GB to start."
        let lowSpace = PreFlightChecklistEvaluator.evaluate(
            availableDiskBytes: 6_000_000_000,
            batteryLevelFraction: 0.90,
            thermalLevel: .nominal,
            exposureDurationS: 0.004,
            iso: 100.0,
            isLocationAuthorized: true
        )
        XCTAssertEqual(lowSpace.storageRow.detail, "Free up 10 GB to start.")

        // Plain-language CaptureDetailSummary (§3f, §4b)
        let detailA = CaptureDetailSummary.build(
            provisionalGrade: "A",
            status: "stopped",
            distanceTolerance10mM: 0.028,
            lockedMarkerIDs: [0, 1, 2, 3],
            isLoopClosed: true,
            isLookUpDone: true,
            wasLookUpSkipped: false,
            primaryTapeMeters: 4.21
        )
        XCTAssertEqual(detailA.verdictTitle, "Ready to share")
        XCTAssertEqual(detailA.accuracySentence, "Accurate to about 3 cm.")
        XCTAssertEqual(detailA.markersValue, "All 4 ✓")
        XCTAssertEqual(detailA.backAtStartValue, "Yes ✓")
        XCTAssertEqual(detailA.lookedUpValue, "Yes ✓")
        XCTAssertEqual(detailA.tapeMeasurementValue, "4.21 m · ✓")

        let detailC = CaptureDetailSummary.build(
            provisionalGrade: "C",
            status: "stopped",
            distanceTolerance10mM: 0.14,
            lockedMarkerIDs: [0, 1, 2, 3],
            isLoopClosed: true,
            isLookUpDone: true,
            wasLookUpSkipped: true,
            primaryTapeMeters: nil
        )
        XCTAssertEqual(detailC.verdictTitle, "Usable, with limits")
        XCTAssertEqual(detailC.lookedUpValue, "Skipped ⚠︎")

        // In-view marker beacon (0.40 m above seen marker) and off-screen edge chevron (DESIGN-SPEC-NATIVE.md §3a ④ & ⑤)
        let camTransform = matrix_identity_double4x4
        let onScreenProj = MarkerScreenProjector.project(
            markerID: 1,
            markerWorldPos: SIMD3<Double>(0.0, -0.40, -2.0),
            isDone: true,
            cameraToWorld: camTransform
        )
        XCTAssertTrue(onScreenProj.isOnScreen)
        XCTAssertTrue(onScreenProj.isDone)
        XCTAssertNil(onScreenProj.edgeChevronSymbol)

        let offScreenRightProj = MarkerScreenProjector.project(
            markerID: 2,
            markerWorldPos: SIMD3<Double>(0.0, -3.0, -1.0),
            isDone: false,
            cameraToWorld: camTransform
        )
        XCTAssertFalse(offScreenRightProj.isOnScreen)
        XCTAssertEqual(offScreenRightProj.edgeChevronSymbol, "chevron.right")
    }

    /// `YS-81`: Always-on unit test verifying VoiceOver marker progress summaries, `Differentiate Without Color` badges,
    /// Dynamic Type accessibility text, and WCAG 2.1 outdoor sunlight contrast ratios (`>= 4.5:1` for text across bright sky, grass, and shade).
    func testYS81_AccessibilityAndOutdoorContrast_VoiceOverSummaryDifferentiateWithoutColorAndWCAGAAContrastInSun() {
        // 1. VoiceOver marker progress summary matches DESIGN-SPEC-NATIVE.md §6 line 552 ("0 and 2 done. 1 and 3 to go.")
        XCTAssertEqual(
            MarkerProgressAccessibility.voiceOverValue(lockedMarkerIDs: [0, 2], isFinishAvailable: false),
            "0 and 2 done. 1 and 3 to go."
        )
        XCTAssertEqual(
            MarkerProgressAccessibility.voiceOverValue(lockedMarkerIDs: [0, 1, 2, 3], isFinishAvailable: true),
            "All 4 markers done, back at the start."
        )
        XCTAssertEqual(
            MarkerProgressAccessibility.accessibilitySizeText(lockedMarkerIDs: [0, 2], isFinishAvailable: false),
            "2 of 4 markers"
        )
        XCTAssertEqual(
            MarkerProgressAccessibility.accessibilitySizeText(lockedMarkerIDs: [0, 1, 2, 3], isFinishAvailable: true),
            "All markers"
        )

        // 2. Differentiate Without Color badges ("Fix" vs "Check", DESIGN-SPEC-NATIVE.md §6 line 535)
        let warnItem = WalkReviewItem(
            id: "missing_marker_2",
            symbolName: "exclamationmark.triangle.fill",
            isWarning: true,
            sentence: "Marker 2 wasn't captured.",
            buttonTitle: "Capture It",
            action: .captureMissingMarker(2)
        )
        let infoItem = WalkReviewItem(
            id: "lookup_skipped",
            symbolName: "info.circle.fill",
            isWarning: false,
            sentence: "You didn't look up at the roof and trees.",
            buttonTitle: "Do It Now",
            action: .performLookUp
        )
        XCTAssertEqual(warnItem.differentiateWithoutColorBadge, "Fix")
        XCTAssertEqual(infoItem.differentiateWithoutColorBadge, "Check")

        // 3. Outdoor WCAG 2.1 contrast audit over bright midday sky (#FFFFFF), sunlit grass, and shade
        let standardAudit = OutdoorContrastAuditor.auditCaptureHUDContrast(increasedContrast: false)
        XCTAssertTrue(standardAudit.passesWCAGAA, "YS-81: Standard outdoor glass + 35% dim backing must exceed 4.5:1 WCAG AA contrast")
        XCTAssertGreaterThanOrEqual(standardAudit.brightSkyTextRatio, 4.5, "YS-81: Bright sky contrast must be >= 4.5:1")
        XCTAssertGreaterThanOrEqual(standardAudit.sunlitGrassTextRatio, 4.5, "YS-81: Sunlit grass contrast must be >= 4.5:1")
        XCTAssertGreaterThanOrEqual(standardAudit.shadeTextRatio, 4.5, "YS-81: Shade contrast must be >= 4.5:1")

        let highContrastAudit = OutdoorContrastAuditor.auditCaptureHUDContrast(increasedContrast: true)
        XCTAssertTrue(highContrastAudit.passesWCAGAA)
        XCTAssertGreaterThan(highContrastAudit.brightSkyTextRatio, standardAudit.brightSkyTextRatio)
    }

    /// `YS-82`: Always-on unit test verifying all 5 experiments (`exposureCapDuringHolds`, `hiResStillAtLock`,
    /// `hardwareButtonControl`, `arAnchorOnLocks`, `pauseAndResumeWithPaddedAudio`) default to `false`, log their
    /// state for `guidance.csv`, enforce the 2-second post-hold exposure-cap safety window, and require passing
    /// the `ExperimentABValidator` gates (`tracking normal >= 99%`, no rise in pose jumps or loop gap, walk validates).
    func testYS82_ExperimentsBehindFlags_DefaultOffLoggedAndABAcceptanceGatesEnforced() {
        // 1. All 5 experiments default to OFF and log "experiments:none"
        var flags = CaptureExperimentFlags.default
        XCTAssertTrue(flags.isAllDefaultOff, "YS-82: Every experiment must default to false")
        XCTAssertEqual(flags.guidanceLogSummary, "experiments:none")
        for id in CaptureExperimentID.allCases {
            XCTAssertFalse(flags.isEnabled(id))
        }

        flags.setEnabled(true, for: .exposureCapDuringHolds)
        flags.setEnabled(true, for: .arAnchorOnLocks)
        XCTAssertEqual(
            flags.guidanceLogSummary,
            "experiments:exposure_cap_during_holds+ar_anchor_on_locks"
        )

        // 2. Experiment 1: Exposure cap (<= 1/250 s = 0.004 s) + 2 s post-hold limited-tracking safety monitor
        let capped = ExposureCapExperimentController.computeCappedExposureAndISO(
            currentExposureDurationS: 0.016,
            currentISO: 200.0
        )
        XCTAssertTrue(capped.didCap)
        XCTAssertEqual(capped.cappedExposureS, 1.0 / 250.0, accuracy: 1e-6)
        XCTAssertEqual(capped.compensatedISO, 800.0, accuracy: 1e-3)

        var expCap = ExposureCapExperimentController()
        let holdFrame = expCap.step(
            timestampS: 10.0,
            isFlagEnabled: true,
            isMarkerHoldActive: true,
            trackingState: "normal"
        )
        XCTAssertTrue(holdFrame.shouldApplyCap)
        XCTAssertFalse(holdFrame.didViolateSafetyWindow)

        // 1.2 s after hold ends (< 2.0 s safety window): a limited-tracking transition must fail & auto-disable the experiment
        let postHoldLimited = expCap.step(
            timestampS: 11.2,
            isFlagEnabled: true,
            isMarkerHoldActive: false,
            trackingState: "limited"
        )
        XCTAssertFalse(postHoldLimited.shouldApplyCap)
        XCTAssertTrue(postHoldLimited.didViolateSafetyWindow, "YS-82: Limited tracking within 2 s after a hold must trigger safety violation")
        XCTAssertTrue(expCap.autoDisabledDueToLimitedTracking, "YS-82: Any failure keeps the experiment off")

        // 3. Experiment 2: Hi-res / maximum-quality still at marker lock
        XCTAssertEqual(HiResLockStillExperiment.jpegQuality(isFlagEnabled: false, didLockMarkerOnThisStill: true), 0.92, accuracy: 1e-6)
        XCTAssertEqual(HiResLockStillExperiment.jpegQuality(isFlagEnabled: true, didLockMarkerOnThisStill: true), 1.00, accuracy: 1e-6)

        // 4. Experiment 3: Hardware button (Camera Control / volume button) as Note and Finish
        XCTAssertEqual(
            HardwareButtonExperimentRouter.resolvePress(isFlagEnabled: false, isRecordingVoiceNote: false, isFinishAvailable: true),
            .ignoredFlagOff
        )
        XCTAssertEqual(
            HardwareButtonExperimentRouter.resolvePress(isFlagEnabled: true, isRecordingVoiceNote: false, isFinishAvailable: false),
            .startVoiceNote
        )
        XCTAssertEqual(
            HardwareButtonExperimentRouter.resolvePress(isFlagEnabled: true, isRecordingVoiceNote: true, isFinishAvailable: true),
            .stopVoiceNote
        )
        XCTAssertEqual(
            HardwareButtonExperimentRouter.resolvePress(isFlagEnabled: true, isRecordingVoiceNote: false, isFinishAvailable: true),
            .openReviewWalkFinish
        )

        // 5. Experiment 4: ARAnchor on locks
        let anchorMat = ARAnchorLockExperiment.makeAnchorTransform(
            positionWorld: SIMD3<Double>(1.25, -0.80, -3.50),
            quaternionXYZW: SIMD4<Double>(0.0, 0.0, 0.0, 1.0)
        )
        XCTAssertEqual(ARAnchorLockExperiment.anchorName(forMarkerID: 2, isClosingHold: false), "yardstick_marker_2")
        XCTAssertEqual(ARAnchorLockExperiment.anchorName(forMarkerID: 0, isClosingHold: true), "yardstick_marker_0_close")
        XCTAssertEqual(Double(anchorMat.columns.3.x), 1.25, accuracy: 1e-5)
        XCTAssertEqual(Double(anchorMat.columns.3.y), -0.80, accuracy: 1e-5)
        XCTAssertEqual(Double(anchorMat.columns.3.z), -3.50, accuracy: 1e-5)

        // 6. Experiment 5: Pause and resume with zero-PCM padded audio (16 kHz mono Int16) and declared gap
        let offPause = PauseResumeAudioPadder.evaluateResume(isFlagEnabled: false, pauseTimestampS: 20.0, resumeTimestampS: 24.5)
        XCTAssertFalse(offPause.canResume, "YS-82: When flag is off, interruption seals immediately and never pauses")

        let shortPause = PauseResumeAudioPadder.evaluateResume(isFlagEnabled: true, pauseTimestampS: 20.0, resumeTimestampS: 24.5)
        XCTAssertTrue(shortPause.canResume)
        XCTAssertEqual(shortPause.gapDurationS, 4.5, accuracy: 1e-6)
        XCTAssertEqual(shortPause.zeroPCMSampleCount, 72_000)
        XCTAssertEqual(shortPause.zeroPCMByteCount, 144_000)
        let padBytes = PauseResumeAudioPadder.makeZeroPCMPaddingData(sampleCount: shortPause.zeroPCMSampleCount)
        XCTAssertEqual(padBytes.count, 144_000)

        // 7. A/B Acceptance Gate (`ExperimentABValidator`)
        let walkA = ExperimentWalkMetrics(
            totalFrames: 1800,
            normalTrackingFrames: 1795,
            poseJumpCount: 0,
            loopClosureGapM: 0.032,
            validates: true,
            limitedTrackingTransitionsDuringOrWithin2sAfterHold: 0
        )
        let passingWalkB = ExperimentWalkMetrics(
            totalFrames: 1800,
            normalTrackingFrames: 1796,
            poseJumpCount: 0,
            loopClosureGapM: 0.029,
            validates: true,
            limitedTrackingTransitionsDuringOrWithin2sAfterHold: 0
        )
        let passVerdict = ExperimentABValidator.evaluate(
            experimentID: .exposureCapDuringHolds,
            baselineWalkA: walkA,
            candidateWalkB: passingWalkB
        )
        XCTAssertTrue(passVerdict.keepEnabled)
        XCTAssertTrue(passVerdict.failureReasons.isEmpty)

        let failingWalkB = ExperimentWalkMetrics(
            totalFrames: 1800,
            normalTrackingFrames: 1760, // 97.78% < 99%
            poseJumpCount: 1, // rose from 0 -> 1
            loopClosureGapM: 0.045, // rose from 0.032 -> 0.045
            validates: true,
            limitedTrackingTransitionsDuringOrWithin2sAfterHold: 1
        )
        let failVerdict = ExperimentABValidator.evaluate(
            experimentID: .exposureCapDuringHolds,
            baselineWalkA: walkA,
            candidateWalkB: failingWalkB
        )
        XCTAssertFalse(failVerdict.keepEnabled, "YS-82: Any A/B regression keeps the experiment off")
        XCTAssertFalse(failVerdict.trackingNormalPassed)
        XCTAssertFalse(failVerdict.noPoseJumpRisePassed)
        XCTAssertFalse(failVerdict.noLoopGapRisePassed)
        XCTAssertFalse(failVerdict.exposureHoldStabilityPassed)
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

        // YS-74: On Walk 5, all 4 markers (0, 1, 2, 3) lock on their opening holds, and the walk stops in
        // `.closingMarker0` ("Head back to marker 0") with `isFinishAvailable == false`!
        XCTAssertEqual(result.guidedEngine.lockedMarkerIDs, [0, 1, 2, 3], "YS-74: Walk 5 must lock Markers 0, 1, 2, 3")
        XCTAssertFalse(result.guidedEngine.isClosingMarker0Locked, "YS-74: Walk 5 must not lock closing Marker 0")
        XCTAssertFalse(
            result.guidedEngine.isFinishAvailable(unifiedLoopClosed: result.phoneLoopState.unifiedLoopClosed),
            "YS-74: Finish must remain unavailable on Walk 5"
        )
        XCTAssertEqual(result.guidedEngine.phase, .closingMarker0, "YS-74: Walk 5 must stop in .closingMarker0 phase")
        XCTAssertTrue(
            result.guidedEngine.instructionHistory.contains { $0.text == "Head back to marker 0" },
            "YS-74: Walk 5 must reach 'Head back to marker 0'"
        )
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

            // 2. Live OpenCV detector on stills check (YS-69 <= 0.3% size parity + YS-71 5/5 loop verdict parity + YS-74 determinism)
            let resLive = try CaptureReplayHarness.replayCaptureDirectory(
                at: walkURL,
                mode: .liveDetectorOnStills
            )
            let resLiveSecondRun = try CaptureReplayHarness.replayCaptureDirectory(
                at: walkURL,
                mode: .liveDetectorOnStills
            )
            XCTAssertEqual(
                resLive.guidedEngine.instructionHistory.map(\.text),
                resLiveSecondRun.guidedEngine.instructionHistory.map(\.text),
                "YS-74: Instruction sequence for \(exp.walkKey) must be 100% deterministic across runs"
            )
            // Across Walks 1–5, Walks 1–4 had 250 mm configured for 180 mm sheets (failing YS-74 ±3% size gate),
            // and Walk 5 never returned to Marker 0, so Finish is unavailable on all 5 legacy walks.
            XCTAssertFalse(
                resLive.guidedEngine.isFinishAvailable(unifiedLoopClosed: resLive.phoneLoopState.unifiedLoopClosed),
                "YS-74: Finish must remain unavailable on \(exp.walkKey)"
            )
            if exp.walkKey == "walk-5" {
                XCTAssertEqual(resLive.guidedEngine.lockedMarkerIDs, [0, 1, 2, 3])
                XCTAssertEqual(resLive.guidedEngine.phase, .closingMarker0)
                XCTAssertTrue(resLive.guidedEngine.instructionHistory.contains { $0.text == "Head back to marker 0" })
            }
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
