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
}
