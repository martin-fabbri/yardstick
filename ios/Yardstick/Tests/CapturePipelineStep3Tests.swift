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

import CryptoKit
import Foundation
import simd
import XCTest

#if canImport(Yardstick)
@testable import Yardstick
#endif

final class CapturePipelineStep3Tests: XCTestCase {

    /// `YS-68`: Verify `WriterBacklogCounter` is atomic, lock-protected on the writer side, never requires
    /// `@MainActor` to decrement, and flags overload only when backlog exceeds `maxBacklog`.
    func testYS68_WriterBacklogCounter_IsAtomicAndNeverRequiresMainActor() {
        let counter = WriterBacklogCounter(maxBacklog: 8)
        XCTAssertEqual(counter.currentBacklog, 0)

        for i in 1...8 {
            let res = counter.acquireSlot()
            XCTAssertFalse(res.overloaded, "Slot \(i) within maxBacklog=8 must not be marked overloaded")
            XCTAssertEqual(res.currentBacklog, i)
        }

        // 9th slot acquires when pendingJobs == 8 (still not > 8)
        let ninth = counter.acquireSlot()
        XCTAssertFalse(ninth.overloaded)
        XCTAssertEqual(ninth.currentBacklog, 9)

        // 10th slot sees pendingJobs == 9 (> 8) -> overloaded!
        let tenth = counter.acquireSlot()
        XCTAssertTrue(tenth.overloaded, "Backlog > 8 must report overloaded")
        XCTAssertEqual(counter.overloadDropCount, 1)

        // Release all 10 slots concurrently from background threads while main thread is blocked
        let group = DispatchGroup()
        for _ in 0..<10 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                counter.releaseSlot()
                group.leave()
            }
        }
        group.wait()
        XCTAssertEqual(counter.currentBacklog, 0, "Backlog must return to 0 without @MainActor involvement")
        XCTAssertEqual(counter.peakBacklog, 10)
    }

    /// `YS-68`: Verify `SingleSlotDetectionScheduler` runs at most 1 job at a time, immediately drops
    /// incoming frames while busy, and drains cleanly via `scheduleAfterPendingWork` / `drainSynchronously`.
    func testYS68_SingleSlotDetectionScheduler_DropsWorkWhenBusyAndDrainsCleanly() {
        let scheduler = SingleSlotDetectionScheduler(label: "io.yardstick.test.detection")
        let gate = DispatchSemaphore(value: 0)
        let started = DispatchSemaphore(value: 0)

        let acceptedFirst = scheduler.submitIfIdle {
            started.signal()
            _ = gate.wait(timeout: .now() + 5.0)
        }
        XCTAssertTrue(acceptedFirst, "First job on idle scheduler must be accepted")
        _ = started.wait(timeout: .now() + 2.0)

        // Submit 5 more jobs while the first is still running; all 5 must be dropped immediately
        for _ in 0..<5 {
            let accepted = scheduler.submitIfIdle {
                XCTFail("Busy scheduler must drop incoming work immediately")
            }
            XCTAssertFalse(accepted)
        }

        let midStats = scheduler.stats
        XCTAssertTrue(midStats.isBusy)
        XCTAssertEqual(midStats.submittedCount, 1)
        XCTAssertEqual(midStats.droppedBusyCount, 5)

        let drainedExpectation = expectation(description: "Drained after pending work")
        scheduler.scheduleAfterPendingWork {
            drainedExpectation.fulfill()
        }
        gate.signal()
        wait(for: [drainedExpectation], timeout: 2.0)

        let finalStats = scheduler.stats
        XCTAssertFalse(finalStats.isBusy)
        XCTAssertEqual(finalStats.completedCount, 1)
        XCTAssertEqual(finalStats.droppedBusyCount, 5)
    }

    /// `YS-68`: Verify `StillCaptureGate` rate-limits marker-triggered stills to `≤ 1` per `0.35 s` and
    /// at most `5` per marker hold, keeping a 2-minute 4-marker walk (and Walk 5 replay) well under `120` stills.
    func testYS68_StillCaptureGate_LimitsWalk5ToUnder120StillsAndAtMost5PerHold() throws {
        var gate = StillCaptureGate(
            minMarkerStillIntervalS: 0.35,
            maxStillsPerMarkerHold: 5,
            holdResetGapS: 1.50,
            periodicStillMinIntervalS: 2.00,
            periodicStillMaxIntervalS: 3.00,
            minPeriodicSharpness: 12.0
        )

        // Simulate a 120-second 30 fps walk (3,600 frames) with 6-second stationary holds on Markers 0, 1, 2, 3, and 0 revisit
        let holdWindows: [(startS: Double, endS: Double, markerID: Int)] = [
            (2.0, 8.0, 0),
            (28.0, 34.0, 1),
            (56.0, 62.0, 2),
            (84.0, 90.0, 3),
            (110.0, 116.0, 0),
        ]

        var markerStillTimestampsByHold: [Int: [Double]] = [:]
        for frameIdx in 0..<3600 {
            let t = Double(frameIdx) / 30.0
            // ArUco runs every 3rd frame
            guard frameIdx % 3 == 0 else { continue }
            var seenIDs: [Int] = []
            var activeHoldIndex: Int? = nil
            for (hIdx, w) in holdWindows.enumerated() where t >= w.startS && t <= w.endS {
                seenIDs.append(w.markerID)
                activeHoldIndex = hIdx
            }
            let decision = gate.evaluateFrame(
                timestampS: t,
                detectedMarkerIDs: seenIDs,
                forceOperatorMarkerTap: false,
                sharpness: 15.0
            )
            if decision.shouldSave, decision.reason == "marker_hold", let hIdx = activeHoldIndex {
                markerStillTimestampsByHold[hIdx, default: []].append(t)
            }
        }

        for hIdx in 0..<holdWindows.count {
            let times = markerStillTimestampsByHold[hIdx] ?? []
            XCTAssertEqual(times.count, 5, "Hold \(hIdx) must cap at exactly 5 marker stills")
            for i in 1..<times.count {
                XCTAssertGreaterThanOrEqual(
                    times[i] - times[i - 1],
                    0.35 - 1e-6,
                    "Marker stills within hold \(hIdx) must be spaced by >= 0.35 s"
                )
            }
        }
        XCTAssertLessThan(
            gate.totalStillsSavedCount,
            120,
            "2-minute 4-marker walk must write fewer than 120 stills (got \(gate.totalStillsSavedCount))"
        )

        // Also verify against Walk 5 real capture data if YARDSTICK_REPLAY_ROOT is provided
        if let rootPath = ProcessInfo.processInfo.environment["YARDSTICK_REPLAY_ROOT"] {
            let walk5URL = URL(fileURLWithPath: rootPath, isDirectory: true)
                .appendingPathComponent("2026-10-31/walk-5/20261101T012134Z_yardstick", isDirectory: true)
            let framesURL = walk5URL.appendingPathComponent("frames.csv")
            let markersURL = walk5URL.appendingPathComponent("markers.csv")
            if FileManager.default.fileExists(atPath: framesURL.path),
               FileManager.default.fileExists(atPath: markersURL.path) {
                let markersText = try String(contentsOf: markersURL, encoding: .utf8)
                var sightingsByFrame: [Int: [Int]] = [:]
                for line in markersText.split(separator: "\n").dropFirst() {
                    let cols = line.split(separator: ",", omittingEmptySubsequences: false)
                    if cols.count >= 3, let sIdx = Int(cols[1]), let mid = Int(cols[2]) {
                        sightingsByFrame[sIdx, default: []].append(mid)
                    }
                }
                let framesText = try String(contentsOf: framesURL, encoding: .utf8)
                var walk5Gate = StillCaptureGate()
                for line in framesText.split(separator: "\n").dropFirst() {
                    let cols = line.split(separator: ",", omittingEmptySubsequences: false)
                    guard cols.count >= 3,
                          cols[2] == "false",
                          let t = Double(cols[0]),
                          let sIdx = Int(cols[1])
                    else { continue }
                    let ids = sightingsByFrame[sIdx] ?? []
                    if !ids.isEmpty || (sIdx % 3 == 0) {
                        _ = walk5Gate.evaluateFrame(
                            timestampS: t,
                            detectedMarkerIDs: ids,
                            forceOperatorMarkerTap: false,
                            sharpness: 15.0
                        )
                    }
                }
                XCTAssertLessThan(
                    walk5Gate.totalStillsSavedCount,
                    120,
                    "Walk 5 (originally 455 stills in 95 s) must be rate-limited to < 120 stills (got \(walk5Gate.totalStillsSavedCount))"
                )
            }
        }
    }

    /// `YS-84`: Verify `IncrementalByteDigest` matches `streamFile` byte-for-byte and that
    /// `collectManifestFileDigests` / `sealManifestAndPackZip` reuses `precomputedDigests` without re-reading precomputed files from disk.
    func testYS84_IncrementalByteDigest_MatchesStreamHashAndSealsWithoutReReadingPrecomputedFiles() throws {
        let fm = FileManager.default
        let tempBase = fm.temporaryDirectory.appendingPathComponent("ys84_test_\(UUID().uuidString)", isDirectory: true)
        let captureDir = tempBase.appendingPathComponent("20261101T040000Z_yardstick", isDirectory: true)
        try fm.createDirectory(at: captureDir.appendingPathComponent("depth"), withIntermediateDirectories: true)
        try fm.createDirectory(at: captureDir.appendingPathComponent("confidence"), withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempBase) }

        let registry = PrecomputedDigestRegistry()

        // Write multi-chunk test payload incrementally and verify IncrementalByteDigest matches streamFile
        var inc = IncrementalByteDigest()
        let framesURL = captureDir.appendingPathComponent("frames.csv")
        let headerData = Data("t,saved_index,dropped,drop_reason,tracking_state,tracking_reason\n".utf8)
        let row1Data = Data("0.000000,0,false,,normal,\n".utf8)
        let row2Data = Data("0.033333,1,false,,normal,\n".utf8)
        var combined = Data()
        combined.append(headerData)
        combined.append(row1Data)
        combined.append(row2Data)
        try combined.write(to: framesURL)

        inc.update(with: headerData)
        inc.update(with: row1Data)
        inc.update(with: row2Data)
        let incMeta = inc.finalize()
        registry.record(incMeta, forRelativePath: "frames.csv")

        let streamedMeta = try IncrementalByteDigest.streamFile(at: framesURL)
        XCTAssertEqual(incMeta, streamedMeta, "Incremental digest across multiple appends must match streamed file digest")

        let depthPayload = Data((0..<4096).map { UInt8($0 & 0xFF) })
        let depthURL = captureDir.appendingPathComponent("depth/000000.png")
        try depthPayload.write(to: depthURL)
        let preDepth = IncrementalByteDigest.compute(for: depthPayload)
        registry.record(preDepth, forRelativePath: "depth/000000.png")

        let streamedDepth = try IncrementalByteDigest.streamFile(at: depthURL)
        XCTAssertEqual(preDepth, streamedDepth)

        // Inject a sentinel digest for depth/000000.png to prove collectManifestFileDigests / sealManifestAndPackZip
        // uses precomputedDigests instead of re-reading the file from disk!
        var customDigests = registry.snapshot()
        let sentinelHash = String(repeating: "a", count: 64)
        customDigests["depth/000000.png"] = FileDigestMeta(
            sha256: sentinelHash,
            crc32: preDepth.crc32,
            sizeBytes: preDepth.sizeBytes
        )

        let collected = try IncrementalByteDigest.collectManifestFileDigests(
            in: captureDir,
            precomputedDigests: customDigests
        )
        let depthEntry = collected.filesArray.first { ($0["path"] as? String) == "depth/000000.png" }
        XCTAssertEqual(
            depthEntry?["sha256"] as? String,
            sentinelHash,
            "Manifest collection must reuse precomputed digest without re-reading depth/000000.png from disk"
        )
        let framesEntry = collected.filesArray.first { ($0["path"] as? String) == "frames.csv" }
        XCTAssertEqual(framesEntry?["sha256"] as? String, incMeta.sha256)
    }

    /// `YS-84`: Verify `DiskSpaceMonitor` warns `"Almost out of space. Finish soon."` on a forced low-space
    /// state before sealing can fail with `ENOSPC`.
    func testYS84_DiskSpaceMonitor_ForcedLowSpaceStateWarnsBeforeFailure() {
        let lowEval = DiskSpaceMonitor.evaluateMidWalkStorage(
            availableBytes: 1_800_000_000,
            recordedCaptureBytes: 900_000_000
        )
        XCTAssertTrue(lowEval.isLowSpace)
        XCTAssertEqual(lowEval.warningMessage, "Almost out of space. Finish soon.")

        // Also test when availableBytes > 2.5 GB floor, but < recordedCaptureBytes + 1.5 GB zip headroom
        let largeWalkLowEval = DiskSpaceMonitor.evaluateMidWalkStorage(
            availableBytes: 2_800_000_000,
            recordedCaptureBytes: 1_600_000_000
        )
        XCTAssertTrue(largeWalkLowEval.isLowSpace)
        XCTAssertEqual(largeWalkLowEval.warningMessage, "Almost out of space. Finish soon.")

        let okEval = DiskSpaceMonitor.evaluateMidWalkStorage(
            availableBytes: 8_000_000_000,
            recordedCaptureBytes: 900_000_000
        )
        XCTAssertFalse(okEval.isLowSpace)
        XCTAssertNil(okEval.warningMessage)
    }

    /// `YS-115`: Verify `ArUcoDetector` at the `minQuadAreaRatio = 0.0004` gate produces zero false positives
    /// on synthetic yard/grass/fence texture while detecting far-range marker pairs (`3.6 m`) and computing
    /// accurate camera-space pair measurements (`computeSingleFramePairMeasurement`).
    func testYS115_ArUcoDetector_LoweredAreaGate_ZeroFalsePositivesOnSyntheticYardTextureAndDetectsFarPair() {
        let width = 640
        let height = 480
        // 1. Synthetic high-frequency grass/fence/concrete texture must produce 0 false positives at 0.0004 gate
        var yardNoise = [UInt8](repeating: 128, count: width * height)
        var rng: UInt64 = 0x5EED_1150_CAFE_BABE
        for i in 0..<yardNoise.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let n = Int((rng >> 56) & 0x7F) - 64
            let stripe = ((i % width) % 17 < 3) ? -35 : 15
            yardNoise[i] = UInt8(clamping: 120 + n + stripe)
        }
        let falsePositives = ArUcoDetector.detectMarkers2D(
            lumaBytes: yardNoise,
            width: width,
            height: height,
            minQuadAreaRatio: ArUcoDetector.minQuadAreaRatio
        )
        XCTAssertEqual(falsePositives.count, 0, "YS-115: 0.0004 quad area gate must produce 0 false positives on yard texture")

        // 2. Render two small markers (28x28 px in 640x480 => area ratio = 784 / 307200 = 0.00255) and verify pair detection
        var pairFrame = [UInt8](repeating: 220, count: width * height)
        func stampMarker(id: Int, originX: Int, originY: Int, cellPx: Int) {
            guard let bits = ArUcoDictionary5x5.patternsByID[id]?.first else { return }
            let totalPx = 7 * cellPx
            for dy in -cellPx..<(totalPx + cellPx) {
                for dx in -cellPx..<(totalPx + cellPx) {
                    let x = originX + dx
                    let y = originY + dy
                    guard x >= 0, x < width, y >= 0, y < height else { continue }
                    if dx < 0 || dy < 0 || dx >= totalPx || dy >= totalPx {
                        pairFrame[y * width + x] = 245 // quiet zone
                    } else {
                        let r = dy / cellPx
                        let c = dx / cellPx
                        if r == 0 || r == 6 || c == 0 || c == 6 {
                            pairFrame[y * width + x] = 15
                        } else {
                            let bitIdx = 24 - ((r - 1) * 5 + (c - 1))
                            let isWhite = ((bits >> bitIdx) & 1) == 1
                            pairFrame[y * width + x] = isWhite ? 245 : 15
                        }
                    }
                }
            }
        }
        let secondID = ArUcoDictionary5x5.patternsByID[7] != nil ? 7 : 5
        stampMarker(id: 0, originX: 120, originY: 200, cellPx: 6)
        stampMarker(id: secondID, originX: 420, originY: 200, cellPx: 6)

        let detected2D = ArUcoDetector.detectMarkers2D(
            lumaBytes: pairFrame,
            width: width,
            height: height,
            minQuadAreaRatio: ArUcoDetector.minQuadAreaRatio
        )
        XCTAssertEqual(Set(detected2D.map(\.markerID)), Set([0, secondID]))

        // 3. Verify camera-space pair measurement between two simultaneous sightings at 3.6 m range
        let s0 = DetectedMarkerSighting(
            markerID: 0,
            cornersUV: [],
            positionWorld: SIMD3<Double>(0, 0, 0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 179.5,
            rangeMeters: 3.55,
            viewingAngleDeg: 18.0,
            positionCamera: SIMD3<Double>(-0.90, -1.10, -3.30),
            quaternionCameraXYZW: SIMD4<Double>(0, 0, 0, 1)
        )
        let sB = DetectedMarkerSighting(
            markerID: secondID,
            cornersUV: [],
            positionWorld: SIMD3<Double>(1.85, 0, 0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.2,
            rangeMeters: 3.65,
            viewingAngleDeg: 21.0,
            positionCamera: SIMD3<Double>(0.95, -1.10, -3.30),
            quaternionCameraXYZW: SIMD4<Double>(0, 0, 0, 1)
        )
        let pairMeas = ArUcoDetector.computeSingleFramePairMeasurement(sightingA: s0, sightingB: sB)
        XCTAssertNotNil(pairMeas)
        XCTAssertEqual(pairMeas?.idA, 0)
        XCTAssertEqual(pairMeas?.idB, secondID)
        XCTAssertEqual(pairMeas?.distM ?? 0.0, 1.85, accuracy: 1e-6)
        XCTAssertEqual(pairMeas?.dxCamM ?? 0.0, 1.85, accuracy: 1e-6)
    }

    /// `YS-115`: Verify the tossed-marker walk (`0...7`) completes via pair connectivity without requiring
    /// close-range solo locks (`L37` Condition 1), speaks the single placement rule (`"Toss them 2–3 big steps apart"`),
    /// and coaches when a marker is `> 3.5 m` from its nearest neighbour (Condition 2).
    func testYS115_TossedMarkerWalk_CompletesViaPairConnectivityWithoutSoloLocks() {
        var engine = GuidedCaptureEngine(requireLookUpStep: false, tossedMarkerMode: true)
        XCTAssertTrue(engine.activeInstruction.contains("2–3 big steps apart"))

        let ringPositions: [SIMD3<Double>] = (0..<8).map { i in
            let angle = Double(i) * (2.0 * .pi / 8.0)
            return SIMD3<Double>(2.6 * cos(angle), 0.0, 2.6 * sin(angle))
        }

        func makePairSighting(id: Int, pos: SIMD3<Double>, range: Double = 2.40) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(300, 300),
                    SIMD2<Double>(360, 300),
                    SIMD2<Double>(360, 360),
                    SIMD2<Double>(300, 360),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: 180.0,
                sizeMMMeasured: 180.0,
                rangeMeters: range,
                viewingAngleDeg: 20.0,
                positionCamera: pos,
                quaternionCameraXYZW: SIMD4<Double>(0, 0, 0, 1)
            )
        }

        // Walk around the 8 tossed markers seeing only pairs at 2.4 m range (outside the <= 1.5 m solo hold range!)
        var t = 1.0
        for i in 0..<8 {
            let a = i
            let b = (i + 1) % 8
            let sA = makePairSighting(id: a, pos: ringPositions[a])
            let sB = makePairSighting(id: b, pos: ringPositions[b])
            _ = engine.step(
                timestampS: t,
                savedIndex: i * 10,
                dt: 0.5,
                trackingState: "normal",
                pitchDeg: -28.0,
                turnRateRadS: 0.05,
                predictedBlurPx: 0.3,
                cameraPositionWorld: SIMD3<Double>(0, 1.3, 0),
                confirmedSightings: [sA, sB],
                rawSightings: [sA, sB],
                unifiedLoopClosed: i == 7
            )
            t += 1.0
        }

        XCTAssertEqual(engine.pairGraphState.connectedToOrigin, Set(0...7), "All 8 markers must be connected to Marker 0")
        XCTAssertEqual(engine.lockedMarkerIDs, Set(0...7), "Condition 1 (L37): All 8 markers must be marked done via pair connectivity without solo locks")
        XCTAssertGreaterThanOrEqual(engine.pairGraphState.cycleCount, 1, "Closing pair (7, 0) must form a cycle")
        XCTAssertTrue(engine.isFinishAvailable(unifiedLoopClosed: true), "Walk must be ready to finish once all tossed markers are connected in a closed cycle")

        // Verify coaching when an unconnected marker is > 3.5 m from every neighbour (Condition 2)
        var engineFar = GuidedCaptureEngine(requireLookUpStep: false, tossedMarkerMode: true)
        let s0 = makePairSighting(id: 0, pos: SIMD3<Double>(0, 0, 0))
        let s1 = makePairSighting(id: 1, pos: SIMD3<Double>(2.0, 0, 0))
        let sFar = makePairSighting(id: 4, pos: SIMD3<Double>(7.5, 0, 0)) // 5.5 m from marker 1 (> 3.5 m)
        _ = engineFar.step(
            timestampS: 1.0,
            savedIndex: 10,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.05,
            predictedBlurPx: 0.3,
            cameraPositionWorld: .zero,
            confirmedSightings: [s0, s1],
            rawSightings: [s0, s1],
            unifiedLoopClosed: false
        )
        let stepFar = engineFar.step(
            timestampS: 3.0,
            savedIndex: 30,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.05,
            predictedBlurPx: 0.3,
            cameraPositionWorld: SIMD3<Double>(6.0, 1.3, 0),
            confirmedSightings: [sFar],
            rawSightings: [sFar],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(stepFar.instructionKey, "pair_marker_too_far_4")
        XCTAssertTrue(stepFar.instruction.contains("over 3.5 m"))
    }

    /// `YS-115`: Verify `GradeModel` applies `disconnected_marker_graph` (cap C) and `pair_graph_no_closed_cycle` (cap B),
    /// and exempts a connected closed-cycle pair walk from `no_tape_check`.
    func testYS115_GradeModel_PairGraphCapsAndClosedCycleExemption() {
        var disconnected = GradeModel.Inputs()
        disconnected.pairGraphConnected = false
        disconnected.pairGraphCycleCount = 0
        disconnected.rawMarker0GapM = 0.02
        disconnected.walkLengthM = 24.0
        let resDisc = GradeModel.grade(disconnected)
        XCTAssertEqual(resDisc.grade, "C")
        XCTAssertTrue(resDisc.capsApplied.contains("disconnected_marker_graph"))

        var openTree = GradeModel.Inputs()
        openTree.pairGraphConnected = true
        openTree.pairGraphCycleCount = 0
        openTree.rawMarker0GapM = 0.02
        openTree.walkLengthM = 24.0
        openTree.tapeChecks = [GradeModel.TapeCheck(tapeM: 5.0, residualM: 0.01)]
        let resTree = GradeModel.grade(openTree)
        XCTAssertEqual(resTree.grade, "B")
        XCTAssertTrue(resTree.capsApplied.contains("pair_graph_no_closed_cycle"))

        var closedCycle = GradeModel.Inputs()
        closedCycle.pairGraphConnected = true
        closedCycle.pairGraphCycleCount = 1
        closedCycle.pairGraphOpenLoopResidualM = 0.018
        closedCycle.walkLengthM = 24.0
        let resClosed = GradeModel.grade(closedCycle)
        XCTAssertEqual(resClosed.grade, "A")
        XCTAssertFalse(resClosed.capsApplied.contains("no_tape_check"))
    }

    /// `YS-115` Items 1 & 2 (`C1`, `C2`, `C4`):
    /// - Any seen marker `1...7` while in `closing_marker_0` sends the walk back to `.markers1To3` (collecting),
    ///   names the unconnected marker (`"Walk to marker 4"`) before the close, and can be targeted and locked.
    /// - Once closing Marker 0 is locked and the look-up is done, the walk reaches `.readyToFinish` and never
    ///   falls back to `"Head back to marker 0"`.
    func testYS115_Item1And2_SeenMarkerInClosingPhaseReturnsToCollectingAndNeverLoopsBackAfterFinish() {
        var engine = GuidedCaptureEngine(requireLookUpStep: true, tossedMarkerMode: true)
        func makeCloseSighting(id: Int, pos: SIMD3<Double>) -> DetectedMarkerSighting {
            DetectedMarkerSighting(
                markerID: id,
                cornersUV: [
                    SIMD2<Double>(400, 400),
                    SIMD2<Double>(520, 400),
                    SIMD2<Double>(520, 520),
                    SIMD2<Double>(400, 520),
                ],
                positionWorld: pos,
                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
                sizeMMPrinted: 180.0,
                sizeMMMeasured: 180.0,
                rangeMeters: 1.25,
                viewingAngleDeg: 18.0,
                positionCamera: SIMD3<Double>(0, -0.5, -1.15),
                quaternionCameraXYZW: SIMD4<Double>(0, 0, 0, 1)
            )
        }

        // Lock markers 0, 1, 2, 3 via 3-still holds so engine enters `.closingMarker0`
        var t = 1.0
        for mid in 0...3 {
            let pos = SIMD3<Double>(Double(mid) * 2.0, 0, 0)
            let s = makeCloseSighting(id: mid, pos: pos)
            for sampleIdx in 0..<3 {
                _ = engine.step(
                    timestampS: t,
                    savedIndex: mid * 100 + sampleIdx * 10,
                    dt: 0.5,
                    trackingState: "normal",
                    pitchDeg: -28.0,
                    turnRateRadS: 0.02,
                    predictedBlurPx: 0.2,
                    cameraPositionWorld: pos + SIMD3<Double>(0, 1.25, 0),
                    confirmedSightings: [s],
                    rawSightings: [s],
                    unifiedLoopClosed: false
                )
                t += 0.55
            }
        }
        XCTAssertEqual(engine.phase, .closingMarker0)

        // Seeing Marker 4 in `.closingMarker0` (even in 2D first) must send the walk back to `.markers1To3`
        // and name `"Walk to marker 4"` before the close!
        t += 2.0
        let stepSeen4 = engine.step(
            timestampS: t,
            savedIndex: 500,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(7.5, 1.25, 0),
            confirmedSightings: [],
            rawSightings: [],
            seen2DMarkerIDs: [4],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.phase, .markers1To3, "C1/C4: Seeing a new marker 4 in closing_marker_0 must return to collecting (.markers1To3)")
        XCTAssertEqual(stepSeen4.instruction, "Walk to marker 4", "C2: Coach must name the unconnected marker before the close")

        // Lock Marker 4, then lock closing Marker 0, then complete look-up
        let pos4 = SIMD3<Double>(8.0, 0, 0)
        let s4 = makeCloseSighting(id: 4, pos: pos4)
        for sampleIdx in 0..<3 {
            t += 0.55
            _ = engine.step(
                timestampS: t,
                savedIndex: 520 + sampleIdx * 10,
                dt: 0.5,
                trackingState: "normal",
                pitchDeg: -28.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: pos4 + SIMD3<Double>(0, 1.25, 0),
                confirmedSightings: [s4],
                rawSightings: [s4],
                unifiedLoopClosed: false
            )
        }
        XCTAssertTrue(engine.lockedMarkerIDs.contains(4))
        XCTAssertEqual(engine.phase, .closingMarker0)

        // Closing hold on Marker 0
        let s0Close = makeCloseSighting(id: 0, pos: .zero)
        for sampleIdx in 0..<3 {
            t += 0.55
            _ = engine.step(
                timestampS: t,
                savedIndex: 600 + sampleIdx * 10,
                dt: 0.5,
                trackingState: "normal",
                pitchDeg: -28.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(0, 1.25, 0),
                confirmedSightings: [s0Close],
                rawSightings: [s0Close],
                unifiedLoopClosed: true
            )
        }
        XCTAssertTrue(engine.isClosingMarker0Locked)
        XCTAssertEqual(engine.phase, .houseWallLookUp)

        // Complete roof and tree look-up (12 steps for roof + 12 steps for trees at 0.25 s cap = 24 steps)
        for i in 0..<25 {
            t += 0.5
            _ = engine.step(
                timestampS: t,
                savedIndex: 700 + i,
                dt: 0.5,
                trackingState: "normal",
                pitchDeg: 22.0,
                turnRateRadS: 0.02,
                predictedBlurPx: 0.2,
                cameraPositionWorld: SIMD3<Double>(0, 1.25, 0),
                confirmedSightings: [],
                rawSightings: [],
                unifiedLoopClosed: true,
                hasValidDepthForLookUp: true,
                elevatedPointsAbove2m: 20,
                didRunMarkerDetection: false
            )
        }
        XCTAssertTrue(engine.lookUpTracker.isDone)
        XCTAssertEqual(engine.phase, .readyToFinish)

        // Even if a new stray marker sighting or `unifiedLoopClosed == false` arrives after `.readyToFinish`,
        // the walk must NEVER fall back to `.closingMarker0` or `"Head back to marker 0"` (Item 2 / C2)!
        t += 2.0
        let stepAfterFinish = engine.step(
            timestampS: t,
            savedIndex: 800,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: SIMD3<Double>(0, 1.25, 0),
            confirmedSightings: [],
            rawSightings: [],
            seen2DMarkerIDs: [6],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.phase, .readyToFinish, "C2: Once closing Marker 0 is locked and look-up is done, phase must never fall back to closingMarker0")
        XCTAssertEqual(stepAfterFinish.instruction, "All done. Tap Finish.")
    }

    /// `YS-115` Item 3 (`C3`): Verify the HUD (`hudDisplayedMarkerIDs`) shows every marker seen, up to 8 (`0...7`).
    func testYS115_Item3_HUDDisplaysAllSeenMarkersUpTo8() {
        var engine = GuidedCaptureEngine(requireLookUpStep: false, tossedMarkerMode: true)
        XCTAssertEqual(engine.hudDisplayedMarkerIDs, [0, 1, 2, 3])
        _ = engine.step(
            timestampS: 1.0,
            savedIndex: 10,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: .zero,
            confirmedSightings: [],
            rawSightings: [],
            seen2DMarkerIDs: [0, 4, 7],
            unifiedLoopClosed: false
        )
        XCTAssertEqual(engine.hudDisplayedMarkerIDs, Array(0...7))
        XCTAssertEqual(engine.hudDisplayedMarkerIDs.count, 8)
    }

    /// `YS-115` Item 4 (`C7`): Re-seeing Marker 0 at the end of a walk (with open-loop VIO drift) never wipes ground coverage.
    func testYS115_Item4_ReSeeingMarker0NeverWipesGroundCoverage() {
        var grid = GroundCoverageGrid()
        grid.anchorMarker0Frame(
            originWorld: SIMD3<Double>(0.0, -1.20, 0.0),
            xAxisWorld: SIMD3<Double>(1.0, 0.0, 0.0)
        )
        for ix in 0..<10 {
            for iz in 0..<10 {
                grid.markCoveredWorldPoint(
                    SIMD3<Double>(Double(ix) * 0.25 + 0.1, -1.20, Double(iz) * 0.25 + 0.1),
                    highConfidence: true
                )
            }
        }
        XCTAssertEqual(grid.coveredCellCount, 100)

        // Re-see Marker 0 at the end of the walk with 0.72 m VIO drift and rotated yaw
        grid.anchorMarker0Frame(
            originWorld: SIMD3<Double>(0.55, -1.18, -0.46),
            xAxisWorld: SIMD3<Double>(0.80, 0.0, 0.60)
        )
        XCTAssertEqual(grid.coveredCellCount, 100, "C7: Re-seeing Marker 0 must never wipe accumulated ground coverage")
    }

    /// `YS-115` Item 5 (`C8`): Marker 0 counts as connected to the origin from walk start, and no `"over 3.5 m"`
    /// warning ever fires for Marker 0 or during `house_wall_lookup`.
    func testYS115_Item5_Marker0ConnectedToOriginAndNoOver3_5mWarningDuringLookUp() {
        var engine = GuidedCaptureEngine(requireLookUpStep: true, tossedMarkerMode: true)
        XCTAssertTrue(engine.pairGraphState.connectedToOrigin.contains(0), "C8: Marker 0 must always be connected to origin")

        let s0 = DetectedMarkerSighting(
            markerID: 0,
            cornersUV: [SIMD2<Double>(100, 100), SIMD2<Double>(200, 100), SIMD2<Double>(200, 200), SIMD2<Double>(100, 200)],
            positionWorld: SIMD3<Double>(0, 0, 0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.0,
            rangeMeters: 1.25,
            viewingAngleDeg: 18.0
        )
        let s4Far = DetectedMarkerSighting(
            markerID: 4,
            cornersUV: [SIMD2<Double>(100, 100), SIMD2<Double>(200, 100), SIMD2<Double>(200, 200), SIMD2<Double>(100, 200)],
            positionWorld: SIMD3<Double>(6.0, 0, 0),
            quaternionXYZW: SIMD4<Double>(0, 0, 0, 1),
            sizeMMPrinted: 180.0,
            sizeMMMeasured: 180.0,
            rangeMeters: 2.50,
            viewingAngleDeg: 25.0
        )
        _ = engine.step(
            timestampS: 1.0,
            savedIndex: 10,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -28.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: .zero,
            confirmedSightings: [s0],
            rawSightings: [s0, s4Far],
            unifiedLoopClosed: false
        )
        XCTAssertNotEqual(engine.pairGraphState.isolatedTooFarMarkerID, 0, "C8: Marker 0 must never be flagged as isolatedTooFarMarkerID")

        // Force phase to `.houseWallLookUp` and verify `"over 3.5 m"` never preempts the roof look-up instruction
        engine.phase = .houseWallLookUp
        let stepLookUp = engine.step(
            timestampS: 3.0,
            savedIndex: 30,
            dt: 0.5,
            trackingState: "normal",
            pitchDeg: -10.0,
            turnRateRadS: 0.02,
            predictedBlurPx: 0.2,
            cameraPositionWorld: .zero,
            confirmedSightings: [],
            rawSightings: [s4Far],
            unifiedLoopClosed: true
        )
        XCTAssertFalse(stepLookUp.instruction.contains("over 3.5 m"), "C8: No over 3.5 m warning may fire during house_wall_lookup")
        XCTAssertTrue(stepLookUp.instruction.contains("Tilt up"))
    }

    /// `YS-115` Item 6 (`C5`, `C6`): Verify `ExposureAndStillnessGate.maxPairMarkerDistanceM == 4.20` and
    /// `ArUcoDetector.measureMarker3DFromGroundPlaneAndPrintedSize` recovers a 3D pose with uncertainty when LiDAR fails.
    func testYS115_Item6_GroundPlanePrintedSizeFallbackAndPairDistanceGate() {
        XCTAssertEqual(ExposureAndStillnessGate.maxPairMarkerDistanceM, 4.20, accuracy: 1e-6)
        let pairGateVerdict = ExposureAndStillnessGate.evaluateMarkerSighting(
            predictedBlurPx: 0.3,
            turnRateRadS: 0.05,
            rangeMeters3D: 3.70,
            horizontalRangeMeters: 3.45,
            maxDistanceMeters: ExposureAndStillnessGate.maxPairMarkerDistanceM
        )
        XCTAssertTrue(pairGateVerdict.passes, "C6: Pair-frame marker at 3.70 m must pass when maxPairMarkerDistanceM = 4.20 m is used")

        // Camera at (0, 1.35, 0) pitched down -22 deg looking along -Z at a 180 mm marker on the y = 0 ground plane at z = -3.40 m
        let pitchRad = -22.0 * .pi / 180.0
        let qCam = simd_quatd(angle: pitchRad, axis: SIMD3<Double>(1, 0, 0))
        var cMat = simd_double4x4(qCam)
        cMat.columns.3 = SIMD4<Double>(0, 1.35, 0, 1)
        let wToC = cMat.inverse

        let half = 0.090
        let worldCorners = [
            SIMD3<Double>(-half, 0.0, -3.40 - half),
            SIMD3<Double>( half, 0.0, -3.40 - half),
            SIMD3<Double>( half, 0.0, -3.40 + half),
            SIMD3<Double>(-half, 0.0, -3.40 + half),
        ]
        let fx = 1365.0, fy = 1365.0, cx = 960.0, cy = 720.0
        let cornersUV: [SIMD2<Double>] = worldCorners.map { pw in
            let pc4 = wToC * SIMD4<Double>(pw.x, pw.y, pw.z, 1.0)
            let z = -pc4.z
            let u = cx + fx * (pc4.x / z)
            let v = cy - fy * (pc4.y / z)
            return SIMD2<Double>(u, v)
        }

        let fallback = ArUcoDetector.measureMarker3DFromGroundPlaneAndPrintedSize(
            markerID: 6,
            cornersUV: cornersUV,
            printedSizeMM: 180.0,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            cameraToWorld: cMat
        )
        XCTAssertNotNil(fallback)
        if let fb = fallback {
            XCTAssertTrue(fb.isGroundPlanePrintedSizeFallback)
            XCTAssertEqual(fb.depthSampleCount, 0)
            XCTAssertGreaterThan(fb.positionUncertaintyMeters, 0.03)
            XCTAssertEqual(fb.positionWorld.x, 0.0, accuracy: 0.02)
            XCTAssertEqual(fb.positionWorld.z, -3.40, accuracy: 0.05)
        }
    }

    /// `YS-115` Field Walk `w3` Replay Test (`20261103T005121Z_yardstick`):
    /// Feeds `w3`'s 8,592 frames and 141 stills in order through the fixed engine and asserts:
    /// 1. Markers `4–7` are each targeted and locked/connected (`lockedMarkerIDs == Set(0...7)`);
    /// 2. The HUD count reaches `8` (`hudDisplayedMarkerIDs.count == 8`);
    /// 3. The walk reaches `"All done. Tap Finish."` with no loop back to `"Head back to marker 0"`;
    /// 4. Ground coverage isn't reset at the end (`coveredCellCount > 50`);
    /// 5. Frames at `saved_index = 3960` and `3966` give a `5–6` pair with both 3D positions about `3 m` apart (`3.0–3.3 m`).
    func testYS115_FieldWalkW3Replay_LocksAll8MarkersForms5To6PairAndFinishesWithoutLoopBack() throws {
        let env = ProcessInfo.processInfo.environment
        let replayRoot = env["YARDSTICK_REPLAY_ROOT"] ?? env["TEST_RUNNER_YARDSTICK_REPLAY_ROOT"] ?? "/nonexistent"
        let w3Path = replayRoot + "/2026-11-02-eight/w3/20261103T005121Z_yardstick"
        let w3URL = URL(fileURLWithPath: w3Path)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: w3URL.appendingPathComponent("frames.csv").path),
            "Field walk w3 capture directory not present on disk"
        )

        let result = try CaptureReplayHarness.replayCaptureDirectory(
            at: w3URL,
            mode: .liveDetectorOnStills,
            defaultPrintedSizeMM: 180.0
        )

        // 1. All 8 markers (0...7, including 4–7) are targeted and locked/connected
        XCTAssertEqual(
            result.guidedEngine.lockedMarkerIDs,
            Set(0...7),
            "All 8 markers (0...7) must be locked or connected on w3"
        )
        for mid in 4...7 {
            XCTAssertTrue(
                result.guidedEngine.lockedMarkerIDs.contains(mid),
                "Marker \(mid) must be locked or connected on w3"
            )
        }

        // 2. HUD count reaches 8
        XCTAssertEqual(result.guidedEngine.hudDisplayedMarkerIDs, Array(0...7))
        XCTAssertEqual(result.guidedEngine.hudDisplayedMarkerIDs.count, 8)

        // 3. Walk reaches "All done. Tap Finish." with no loop back to "Head back to marker 0"
        XCTAssertEqual(result.guidedEngine.phase, .readyToFinish)
        XCTAssertEqual(result.guidedEngine.activeInstruction, "All done. Tap Finish.")
        XCTAssertTrue(result.guidedEngine.isFinishAvailable(unifiedLoopClosed: result.phoneLoopState.unifiedLoopClosed))
        if let doneIdx = result.guidedEngine.instructionHistory.firstIndex(where: { $0.key == "all_done_finish" }) {
            let instructionsAfterDone = result.guidedEngine.instructionHistory[doneIdx...].map(\.key)
            XCTAssertFalse(
                instructionsAfterDone.contains("return_marker_0"),
                "Walk must never fall back to 'Head back to marker 0' after reaching 'All done. Tap Finish.'"
            )
        } else {
            XCTFail("Expected instructionHistory to contain 'all_done_finish'")
        }

        // 4. Coverage is not reset at the end when Marker 0 is re-seen
        XCTAssertGreaterThan(
            result.groundCoverageGrid.coveredCellCount,
            50,
            "Ground coverage must not be wiped when Marker 0 is re-seen at the end of w3"
        )

        // 5. Frames 3960 and 3966 produce a 5–6 pair with both 3D positions ~3.0–3.3 m apart
        let pairs3960 = result.replayedPairs.filter { $0.savedIndex == 3960 && $0.pair.idA == 5 && $0.pair.idB == 6 }
        let pairs3966 = result.replayedPairs.filter { $0.savedIndex == 3966 && $0.pair.idA == 5 && $0.pair.idB == 6 }
        XCTAssertEqual(pairs3960.count, 1, "Frame 3960 must produce a 5–6 pair")
        XCTAssertEqual(pairs3966.count, 1, "Frame 3966 must produce a 5–6 pair")
        if let p3960 = pairs3960.first?.pair {
            XCTAssertEqual(p3960.distM, 3.15, accuracy: 0.20, "5–6 pair distance in frame 3960 must be ~3.1–3.2 m")
            XCTAssertGreaterThan(p3960.sigmaDistM, 0.03, "Ground-plane printed-size fallback must record uncertainty")
        }
        if let p3966 = pairs3966.first?.pair {
            XCTAssertEqual(p3966.distM, 3.15, accuracy: 0.20, "5–6 pair distance in frame 3966 must be ~3.1–3.2 m")
            XCTAssertGreaterThan(p3966.sigmaDistM, 0.03, "Ground-plane printed-size fallback must record uncertainty")
        }
    }
}
