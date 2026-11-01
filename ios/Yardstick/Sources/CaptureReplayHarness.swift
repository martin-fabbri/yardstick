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

import CoreGraphics
import Foundation
import ImageIO
import os
import simd

/// Thin, deterministic frame representation consumed by the phone's detector, loop-closure rule,
/// and guidance engine without requiring a live `ARFrame` (`YS-66`).
public struct CaptureFrameInput: Sendable {
    public let timestampS: Double
    public let savedIndex: Int?
    public let dropped: Bool
    public let dropReason: String
    public let trackingState: String
    public let trackingReason: String
    public let cameraToWorld: simd_double4x4?
    public let fx: Double
    public let fy: Double
    public let cx: Double
    public let cy: Double
    public let rgbWidth: Int
    public let rgbHeight: Int
    public let exposureDurationS: Double?
    public let iso: Double?
    public let lumaBytes: [UInt8]?
    public let depthMapMM: [UInt16]?
    public let confidenceMap: [UInt8]?
    public let depthWidth: Int
    public let depthHeight: Int
    public let forceMarkerDetect: Bool

    public init(
        timestampS: Double,
        savedIndex: Int?,
        dropped: Bool = false,
        dropReason: String = "",
        trackingState: String = "normal",
        trackingReason: String = "",
        cameraToWorld: simd_double4x4? = nil,
        fx: Double = 0.0,
        fy: Double = 0.0,
        cx: Double = 0.0,
        cy: Double = 0.0,
        rgbWidth: Int = 1920,
        rgbHeight: Int = 1440,
        exposureDurationS: Double? = nil,
        iso: Double? = nil,
        lumaBytes: [UInt8]? = nil,
        depthMapMM: [UInt16]? = nil,
        confidenceMap: [UInt8]? = nil,
        depthWidth: Int = 256,
        depthHeight: Int = 192,
        forceMarkerDetect: Bool = false
    ) {
        self.timestampS = timestampS
        self.savedIndex = savedIndex
        self.dropped = dropped
        self.dropReason = dropReason
        self.trackingState = trackingState
        self.trackingReason = trackingReason
        self.cameraToWorld = cameraToWorld
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.rgbWidth = rgbWidth
        self.rgbHeight = rgbHeight
        self.exposureDurationS = exposureDurationS
        self.iso = iso
        self.lumaBytes = lumaBytes
        self.depthMapMM = depthMapMM
        self.confidenceMap = confidenceMap
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.forceMarkerDetect = forceMarkerDetect
    }

    public var cameraPositionWorld: SIMD3<Double>? {
        guard let cMat = cameraToWorld else { return nil }
        return SIMD3<Double>(cMat.columns.3.x, cMat.columns.3.y, cMat.columns.3.z)
    }
}

/// A timed 3D marker sighting produced during live capture or replay (`YS-66`).
public struct ReplayMarkerSighting: Sendable {
    public let timestampS: Double
    public let savedIndex: Int
    public let sighting: DetectedMarkerSighting

    public init(timestampS: Double, savedIndex: Int, sighting: DetectedMarkerSighting) {
        self.timestampS = timestampS
        self.savedIndex = savedIndex
        self.sighting = sighting
    }
}

/// Encapsulates the phone's marker-excursion and loop-closure tracking state (`YS-66`, `YS-71`),
/// shared between `CaptureSessionRecorder` and `CaptureReplayHarness`.
///
/// Under `YS-71`, the phone runs the exact same loop-closure rule as the Python toolkit
/// (`toolkit/src/yardstick/geometry.py:326–360, 512–598`):
/// - clusters sightings into passes separated by `dt >= 3.0 s` and `(maxExcursion >= 1.0 m || dt >= 10.0 s)`,
/// - requires Markers 1–3 to be seen (or `endEarlyChosen == true`) before Marker 0's closing hold (`YS-71`),
/// - requires closing Marker 0 sightings to occur after the last operator tap and after the latest sighting of
///   any other marker (`mid != 0`), within `1.50 m` camera distance, and within `min(30.0, 0.35 * walkSpanS)` of the
///   walk/close end (`YS-67`, `YS-71`),
/// - computes the loop-closure gap as the 3D distance between the component-wise median of the opening pass and
///   the component-wise median of the valid closing pass (`geometry.py:580–589`).
public struct PhoneLoopRuleState: Sendable {
    public var marker0FirstPassPos: SIMD3<Double>? = nil
    public var marker0FirstPassTimeS: Double? = nil
    public var marker0MaxExcursionM: Double = 0.0
    public var marker0LastPassPos: SIMD3<Double>? = nil
    public var marker0LastPassTimeS: Double? = nil
    /// Legacy pre-`YS-71` gap (single-sighting distance without pass medians or other-marker/tap ordering),
    /// retained so `YS-66` regression tests can verify the historical `7.5 cm` (`0.0753 m`) false close.
    public var legacyPreYS71GapMeters: Double? = nil
    /// Active loop-closure gap (`YS-71` unified rule when evaluated via `evaluateUnifiedLoopClosure`, or legacy fallback).
    public var loopClosureGapMeters: Double? = nil
    /// Unified `YS-71` loop-closure verdict matching `toolkit/src/yardstick/geometry.py:536–598`.
    public var unifiedLoopClosed: Bool = false
    /// Unified `YS-71` loop-closure gap in meters (distance between median of opening pass and median of closing pass).
    public var unifiedLoopClosureGapMeters: Double? = nil
    public var detectedMarkerIDs: Set<Int> = []
    public var endEarlyChosen: Bool = false

    public var allSightings: [ReplayMarkerSighting] = []
    public var cameraPositionsByIndex: [Int: SIMD3<Double>] = [:]
    public var walkStartTimeS: Double? = nil
    public var walkEndTimeS: Double? = nil
    public var lastOperatorTapTimeS: Double? = nil
    public var loopClosedEventTimeS: Double? = nil

    public init() {}

    /// Update camera excursion from Marker 0 Pass 1 for a delivered frame.
    public mutating func updateCameraExcursion(
        cameraPositionWorld camPosW: SIMD3<Double>,
        savedIndex: Int? = nil,
        timestampS: Double? = nil
    ) {
        if let sIdx = savedIndex {
            cameraPositionsByIndex[sIdx] = camPosW
        }
        if let t = timestampS {
            if walkStartTimeS == nil { walkStartTimeS = t }
            walkEndTimeS = t
        }
        if let m0Pos = marker0FirstPassPos {
            let distFromM0 = simd_length(camPosW - m0Pos)
            if distFromM0 > marker0MaxExcursionM {
                marker0MaxExcursionM = distFromM0
            }
        }
    }

    public mutating func recordOperatorTap(at timestampS: Double) {
        if lastOperatorTapTimeS == nil || timestampS > lastOperatorTapTimeS! {
            lastOperatorTapTimeS = timestampS
        }
        evaluateUnifiedLoopClosure()
    }

    public mutating func recordLoopClosedEvent(at timestampS: Double) {
        if loopClosedEventTimeS == nil || timestampS > loopClosedEventTimeS! {
            loopClosedEventTimeS = timestampS
        }
        evaluateUnifiedLoopClosure()
    }

    /// Clear `loopClosedEventTimeS` when `"Fix now"` re-opens the walk (`YS-71`, `YS-72`, `YS-74`)
    /// so sightings captured during the fix and the subsequent return to Marker 0 are included.
    public mutating func clearLoopClosedForFixNow() {
        loopClosedEventTimeS = nil
        evaluateUnifiedLoopClosure()
    }

    /// Ingest 3D marker sightings detected on a frame at `timestampS` with camera position `camPosW`.
    /// Returns `true` if the unified `YS-71` loop-closure state updated on this frame.
    @discardableResult
    public mutating func ingestSightings(
        _ sightings: [DetectedMarkerSighting],
        timestampS: Double,
        savedIndex: Int = 0,
        cameraPositionWorld camPosW: SIMD3<Double>
    ) -> Bool {
        cameraPositionsByIndex[savedIndex] = camPosW
        if walkStartTimeS == nil { walkStartTimeS = timestampS }
        walkEndTimeS = timestampS

        for m in sightings {
            detectedMarkerIDs.insert(m.markerID)
            allSightings.append(ReplayMarkerSighting(timestampS: timestampS, savedIndex: savedIndex, sighting: m))
            if m.markerID == 0 {
                if marker0FirstPassPos == nil {
                    marker0FirstPassPos = m.positionWorld
                    marker0FirstPassTimeS = timestampS
                } else if let firstT = marker0FirstPassTimeS,
                          let firstPos = marker0FirstPassPos,
                          (timestampS - firstT) >= 3.0,
                          (marker0MaxExcursionM >= 1.0 || (timestampS - firstT) >= 10.0),
                          simd_length(camPosW - m.positionWorld) <= 1.50 {
                    let legacyGap = simd_length(m.positionWorld - firstPos)
                    legacyPreYS71GapMeters = legacyGap
                }
            }
        }

        let prevUnifiedGap = unifiedLoopClosureGapMeters
        evaluateUnifiedLoopClosure()
        return unifiedLoopClosureGapMeters != prevUnifiedGap
    }

    /// Group sightings of each `markerID` into passes separated by `dt >= 3.0 s` and
    /// `(maxExcursion >= 1.0 m || dt >= 10.0 s)`, matching `toolkit/src/yardstick/geometry.py:326–360` (`YS-71`).
    public static func clusterMarkerPasses(
        sightings: [ReplayMarkerSighting],
        cameraPositionsByIndex: [Int: SIMD3<Double>],
        minTimeGapS: Double = 3.0,
        minExcursionM: Double = 1.0
    ) -> [Int: [[ReplayMarkerSighting]]] {
        var byMarker: [Int: [ReplayMarkerSighting]] = [:]
        for s in sightings {
            byMarker[s.sighting.markerID, default: []].append(s)
        }

        var passesByMarker: [Int: [[ReplayMarkerSighting]]] = [:]
        for (mid, rows) in byMarker {
            let sortedRows = rows.sorted { $0.timestampS < $1.timestampS }
            guard let firstRow = sortedRows.first else { continue }
            var passes: [[ReplayMarkerSighting]] = [[firstRow]]

            for r in sortedRows.dropFirst() {
                let prev = passes[passes.count - 1].last!
                let dt = r.timestampS - prev.timestampS
                if dt >= minTimeGapS {
                    let idxPrev = prev.savedIndex
                    let idxCurr = r.savedIndex
                    let mPos = prev.sighting.positionWorld
                    var maxExc = 0.0
                    if idxPrev <= idxCurr {
                        for k in idxPrev...idxCurr {
                            if let camP = cameraPositionsByIndex[k] {
                                let d = simd_length(camP - mPos)
                                if d > maxExc { maxExc = d }
                            }
                        }
                    }
                    if maxExc >= minExcursionM || dt >= 10.0 {
                        passes.append([r])
                        continue
                    }
                }
                passes[passes.count - 1].append(r)
            }
            passesByMarker[mid] = passes
        }
        return passesByMarker
    }

    /// Compute component-wise median of 3D positions (`geometry.py:580–587`).
    public static func medianPosition3D(_ positions: [SIMD3<Double>]) -> SIMD3<Double> {
        guard !positions.isEmpty else { return .zero }
        func med(_ vals: [Double]) -> Double {
            let s = vals.sorted()
            let m = s.count / 2
            return (s.count % 2 == 1) ? s[m] : 0.5 * (s[m - 1] + s[m])
        }
        return SIMD3<Double>(
            med(positions.map(\.x)),
            med(positions.map(\.y)),
            med(positions.map(\.z))
        )
    }

    /// Evaluate the unified `YS-71` loop-closure rule (`toolkit/src/yardstick/geometry.py:512–598`).
    public mutating func evaluateUnifiedLoopClosure(
        confirmedMarker0StillIndices: Set<Int>? = nil,
        requireAllFourMarkersOrEndEarly: Bool = true,
        maxClosingDistanceM: Double = 1.50,
        poseJumpSumM: Double = 0.0
    ) {
        let passesByMarker = Self.clusterMarkerPasses(
            sightings: allSightings,
            cameraPositionsByIndex: cameraPositionsByIndex
        )
        guard let m0Passes = passesByMarker[0], m0Passes.count >= 2 else {
            unifiedLoopClosed = false
            unifiedLoopClosureGapMeters = nil
            loopClosureGapMeters = nil
            marker0LastPassPos = nil
            marker0LastPassTimeS = nil
            return
        }

        // YS-71: Finish requires markers 1–3 seen/locked (or "End early" chosen) before Marker 0's closing hold
        if requireAllFourMarkersOrEndEarly && !endEarlyChosen {
            let hasM123 = detectedMarkerIDs.contains(1) && detectedMarkerIDs.contains(2) && detectedMarkerIDs.contains(3)
            if !hasM123 {
                unifiedLoopClosed = false
                unifiedLoopClosureGapMeters = nil
                loopClosureGapMeters = nil
                marker0LastPassPos = nil
                marker0LastPassTimeS = nil
                return
            }
        }

        var latestOtherMarkerTimeS: Double? = nil
        for (mid, mPasses) in passesByMarker where mid != 0 {
            for mp in mPasses {
                for r in mp {
                    if let lcT = loopClosedEventTimeS, r.timestampS > lcT + 1e-6 {
                        continue
                    }
                    if latestOtherMarkerTimeS == nil || r.timestampS > latestOtherMarkerTimeS! {
                        latestOtherMarkerTimeS = r.timestampS
                    }
                }
            }
        }

        let effectiveEndTimeS = loopClosedEventTimeS ?? walkEndTimeS
        let effectiveLastTapTimeS: Double? = {
            guard let tapT = lastOperatorTapTimeS else { return nil }
            if let lcT = loopClosedEventTimeS, tapT > lcT + 1e-6 {
                return nil
            }
            return tapT
        }()

        let walkSpanS: Double = {
            guard let endT = effectiveEndTimeS, let startT = walkStartTimeS, endT >= startT else {
                return 0.0
            }
            return endT - startT
        }()

        let lastPassM0: [ReplayMarkerSighting]
        if let lcT = loopClosedEventTimeS {
            let returnPasses = m0Passes.dropFirst().map { mp in
                mp.filter { $0.timestampS <= lcT + 1e-6 }
            }.filter { !$0.isEmpty }
            lastPassM0 = returnPasses.last ?? []
        } else {
            lastPassM0 = m0Passes.last ?? []
        }

        var validClosingRows: [ReplayMarkerSighting] = []
        for r in lastPassM0 {
            let rT = r.timestampS
            let rIdx = r.savedIndex
            if let endT = effectiveEndTimeS {
                if (endT - rT) > 30.0 {
                    continue
                }
                if walkSpanS > 5.0 && (endT - rT) > min(30.0, 0.35 * walkSpanS) {
                    continue
                }
            }
            if let tapT = effectiveLastTapTimeS, rT < tapT - 1e-6 {
                continue
            }
            if let otherT = latestOtherMarkerTimeS, rT < otherT - 1e-6 {
                continue
            }
            guard let camP = cameraPositionsByIndex[rIdx] else {
                continue
            }
            if simd_length(camP - r.sighting.positionWorld) > maxClosingDistanceM {
                continue
            }
            if let confirmedSet = confirmedMarker0StillIndices, !confirmedSet.contains(rIdx) {
                continue
            }
            validClosingRows.append(r)
        }

        guard !validClosingRows.isEmpty else {
            unifiedLoopClosed = false
            unifiedLoopClosureGapMeters = nil
            loopClosureGapMeters = nil
            marker0LastPassPos = nil
            marker0LastPassTimeS = nil
            return
        }

        let firstPass = m0Passes[0]
        let pFirst = Self.medianPosition3D(firstPass.map(\.sighting.positionWorld))
        let pLast = Self.medianPosition3D(validClosingRows.map(\.sighting.positionWorld))
        let rawGap = simd_length(pLast - pFirst)
        let effGap = max(rawGap, poseJumpSumM)

        unifiedLoopClosed = true
        unifiedLoopClosureGapMeters = effGap
        loopClosureGapMeters = effGap
        marker0FirstPassPos = pFirst
        marker0FirstPassTimeS = firstPass.first?.timestampS
        marker0LastPassPos = pLast
        marker0LastPassTimeS = validClosingRows.last?.timestampS
    }
}

/// Toolkit reference verdict loaded from the checked-in Python toolkit golden file
/// (`ios/Yardstick/Tests/Fixtures/toolkit_golden_walks_1_5.json`, generated by
/// `toolkit/scripts/generate_ys66_replay_fixtures.py`).
public struct ToolkitGoldenMarkerEntry: Codable, Sendable {
    public let sightings_count: Int
    public let size_mm_printed: Double
    public let size_mm_measured: Double
    public let scale_error_ratio: Double
    public let scale_error_se: Double
}

public struct ToolkitGoldenPoseJump: Codable, Sendable {
    public let saved_index_from: Int
    public let saved_index_to: Int
    public let t_s: Double
    public let jump_m: Double
}

public struct ToolkitGoldenWalkVerdict: Codable, Sendable {
    public let walk_key: String
    public let relative_path: String
    public let capture_id: String
    public let grade: String
    public let caps_applied: [String]
    public let grade_coeff_loop_closure_gap_m: Double?
    public let grade_coeff_reports_closed: Bool
    public let frame_marker0_loop_closure_gap_m: Double?
    public let frame_marker0_loop_closed: Bool
    public let pose_jumps: [ToolkitGoldenPoseJump]
    public let reasons: [String]
    public let per_marker: [String: ToolkitGoldenMarkerEntry]
}

public struct ToolkitGoldenFile: Codable, Sendable {
    public let schema: String
    public let contract_version: String
    public let grading_version: String
    public let generated_by_command: String
    public let trimmed_walk5_saved_indices: [Int]
    public let walks: [String: ToolkitGoldenWalkVerdict]

    public static func load(from url: URL) throws -> ToolkitGoldenFile {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ToolkitGoldenFile.self, from: data)
    }
}

public enum CaptureReplayDetectionMode: Sendable {
    /// Decode saved JPEGs (`stills/%06d.jpg`) to BT.601 full-range luma and run `ArUcoDetector.detectMarkers2D` + 3D LiDAR plane fit (`YS-69`, `YS-70`).
    case liveDetectorOnStills
    /// Use the 2D corner UV coordinates recorded in `markers.csv` and run `ArUcoDetector.measureMarker3DOnLiDARPlane` against the depth/confidence PNGs (`YS-66`).
    case recordedCornersFromCSV
}

public struct CaptureReplayResult: Sendable {
    public let sessionID: String
    public let totalFrameRows: Int
    public let savedFrameCount: Int
    public let droppedFrameCount: Int
    public let stillCount: Int
    public let originalSightings: [ReplayMarkerSighting]
    public let replayedSightings: [ReplayMarkerSighting]
    public let phoneLoopState: PhoneLoopRuleState
    public let guidedEngine: GuidedCaptureEngine
    public let elapsedSeconds: Double
    /// Count of still-backed rows in `originalSightings` rejected by OpenCV `DICT_5X5_250` + sub-pixel corner shift (`YS-70`, `YS-73`).
    public let openCVRejectedOriginalSightingsCount: Int
    /// Count of those OpenCV-rejected sightings that are also flagged by the `YS-73` stillness and distance gate (`turnRate <= 0.15 rad/s`, `0.5–1.5 m`).
    public let ys73FlaggedOfOpenCVRejectedCount: Int
    /// Count of good OpenCV-confirmed sightings per marker kept after applying the `YS-73` stillness and distance gate.
    public let ys73KeptGoodSightingsByMarker: [Int: Int]
    /// Count of replayed sightings that were written without OpenCV JPEG confirmation (`0` under `YS-70`).
    public let unconfirmedSightingsCount: Int
    /// Count of replayed sightings that disagree with OpenCV on the saved JPEG (`0` under `YS-70`).
    public let markerOpenCVDisagreementCount: Int

    /// Compute the median `sizeMMMeasured` for `markerID` in `sightings`.
    public static func medianSizeMM(for markerID: Int, in sightings: [ReplayMarkerSighting]) -> Double? {
        let vals = sightings.filter { $0.sighting.markerID == markerID }.map(\.sighting.sizeMMMeasured).sorted()
        guard !vals.isEmpty else { return nil }
        let mid = vals.count / 2
        return (vals.count % 2 == 1) ? vals[mid] : 0.5 * (vals[mid - 1] + vals[mid])
    }

    /// Compute the multi-frame pass-fused marker size in millimeters matching `toolkit/src/yardstick/markers.py:332–347` (`YS-69`).
    public static func fusedPassSizeMM(
        for markerID: Int,
        in sightings: [ReplayMarkerSighting],
        cameraPositionsByIndex: [Int: SIMD3<Double>]
    ) -> Double? {
        let passesByMarker = PhoneLoopRuleState.clusterMarkerPasses(
            sightings: sightings,
            cameraPositionsByIndex: cameraPositionsByIndex
        )
        guard let passes = passesByMarker[markerID], !passes.isEmpty else {
            return medianSizeMM(for: markerID, in: sightings)
        }
        var fusedPassSizes: [Double] = []
        for passRows in passes {
            let validCorners = passRows.map(\.sighting.cornersWorld).filter { $0.count == 4 }
            guard !validCorners.isEmpty else { continue }
            var meanCorners = [SIMD3<Double>](repeating: .zero, count: 4)
            let invCount = 1.0 / Double(validCorners.count)
            for cw in validCorners {
                for i in 0..<4 {
                    meanCorners[i] += cw[i] * invCount
                }
            }
            var edgeSum = 0.0
            for i in 0..<4 {
                edgeSum += simd_length(meanCorners[(i + 1) % 4] - meanCorners[i])
            }
            fusedPassSizes.append((edgeSum * 0.25) * 1000.0)
        }
        guard !fusedPassSizes.isEmpty else {
            return medianSizeMM(for: markerID, in: sightings)
        }
        return fusedPassSizes.reduce(0.0, +) / Double(fusedPassSizes.count)
    }
}

public enum CaptureReplayHarness {
    /// Load a 16-bit millimeter depth PNG (`depth/%06d.png`).
    public static func loadDepth16BitPNG(at url: URL) -> (pixels: [UInt16], width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImg = CGImageSourceCreateImageAtIndex(src, 0, nil),
              let provider = cgImg.dataProvider,
              let data = provider.data as Data?
        else { return nil }
        let w = cgImg.width
        let h = cgImg.height
        let bpr = cgImg.bytesPerRow
        let isBigEndian = cgImg.byteOrderInfo == .order16Big || cgImg.byteOrderInfo == .orderDefault
        var out = [UInt16](repeating: 0, count: w * h)
        data.withUnsafeBytes { raw in
            guard let basePtr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for y in 0..<h {
                let rowPtr = basePtr + y * bpr
                for x in 0..<w {
                    let b0 = UInt16(rowPtr[x * 2])
                    let b1 = UInt16(rowPtr[x * 2 + 1])
                    out[y * w + x] = isBigEndian ? ((b0 << 8) | b1) : ((b1 << 8) | b0)
                }
            }
        }
        return (out, w, h)
    }

    /// Load an 8-bit confidence PNG (`confidence/%06d.png`).
    public static func loadConfidence8BitPNG(at url: URL) -> (pixels: [UInt8], width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImg = CGImageSourceCreateImageAtIndex(src, 0, nil),
              let provider = cgImg.dataProvider,
              let data = provider.data as Data?
        else { return nil }
        let w = cgImg.width
        let h = cgImg.height
        let bpr = cgImg.bytesPerRow
        var out = [UInt8](repeating: 0, count: w * h)
        data.withUnsafeBytes { raw in
            guard let basePtr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for y in 0..<h {
                let rowPtr = basePtr + y * bpr
                for x in 0..<w {
                    out[y * w + x] = rowPtr[x]
                }
            }
        }
        return (out, w, h)
    }

    /// Decode a saved JPEG still into an 8-bit ITU-R BT.601 full-range luma buffer matching `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`.
    public static func loadLumaFromJPEG(at url: URL) -> (luma: [UInt8], width: Int, height: Int)? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return ArUcoDetector.decodeLumaFromJPEGData(data)
    }

    /// Replay a recorded capture folder on disk (`frames.csv`, `poses.csv`, `stills.csv`, `markers.csv`, `events.csv`,
    /// `stills/*.jpg`, `depth/*.png`, `confidence/*.png`) through the phone's detector and rules (`YS-66`, `YS-69`, `YS-70`, `YS-71`, `YS-73`).
    public static func replayCaptureDirectory(
        at dirURL: URL,
        mode: CaptureReplayDetectionMode = .liveDetectorOnStills,
        defaultPrintedSizeMM: Double = 180.0,
        overridePrintedSizeMM: Double? = nil,
        preWalkTapeMeters0To1: Double? = nil
    ) throws -> CaptureReplayResult {
        let tStart = CFAbsoluteTimeGetCurrent()
        let sessionID = dirURL.lastPathComponent

        // 1. Parse poses.csv and compute per-frame angular turn rates (rad/s, YS-73)
        let posesText = try String(contentsOf: dirURL.appendingPathComponent("poses.csv"), encoding: .utf8)
        struct PoseEntry: Sendable {
            let t: Double
            let savedIndex: Int
            let cMat: simd_double4x4
            let fx: Double, fy: Double, cx: Double, cy: Double
            let w: Int, h: Int
            let turnRateRadS: Double
        }
        var posesByIdx: [Int: PoseEntry] = [:]
        var orderedPoses: [PoseEntry] = []
        var prevPoseMat: simd_double4x4? = nil
        var prevPoseT: Double = 0.0
        for line in posesText.split(separator: "\n").dropFirst() {
            let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard cols.count >= 15,
                  let t = Double(cols[0]),
                  let idx = Int(cols[1]),
                  let px = Double(cols[2]), let py = Double(cols[3]), let pz = Double(cols[4]),
                  let qx = Double(cols[5]), let qy = Double(cols[6]), let qz = Double(cols[7]), let qw = Double(cols[8]),
                  let fx = Double(cols[9]), let fy = Double(cols[10]), let cx = Double(cols[11]), let cy = Double(cols[12]),
                  let w = Int(cols[13]), let h = Int(cols[14])
            else { continue }
            let q = simd_normalize(simd_quatd(ix: qx, iy: qy, iz: qz, r: qw))
            var cMat = simd_double4x4(q)
            cMat.columns.3 = SIMD4<Double>(px, py, pz, 1.0)
            let tr = ExposureAndStillnessGate.computeTurnRateRadS(
                previousTransform: prevPoseMat,
                currentTransform: cMat,
                dt: t - prevPoseT
            )
            prevPoseMat = cMat
            prevPoseT = t
            let entry = PoseEntry(t: t, savedIndex: idx, cMat: cMat, fx: fx, fy: fy, cx: cx, cy: cy, w: w, h: h, turnRateRadS: tr)
            posesByIdx[idx] = entry
            orderedPoses.append(entry)
        }

        // 2. Parse original markers.csv
        let markersText = try String(contentsOf: dirURL.appendingPathComponent("markers.csv"), encoding: .utf8)
        var originalSightings: [ReplayMarkerSighting] = []
        var printedSizeByMarker: [Int: Double] = [:]
        for line in markersText.split(separator: "\n").dropFirst() {
            let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard cols.count >= 20,
                  let t = Double(cols[0]),
                  let idx = Int(cols[1]),
                  let mid = Int(cols[2]),
                  let sPrinted = Double(cols[3]),
                  let sMeasured = Double(cols[4]),
                  let x = Double(cols[5]), let y = Double(cols[6]), let z = Double(cols[7]),
                  let qx = Double(cols[8]), let qy = Double(cols[9]), let qz = Double(cols[10]), let qw = Double(cols[11]),
                  let c0u = Double(cols[12]), let c0v = Double(cols[13]),
                  let c1u = Double(cols[14]), let c1v = Double(cols[15]),
                  let c2u = Double(cols[16]), let c2v = Double(cols[17]),
                  let c3u = Double(cols[18]), let c3v = Double(cols[19])
            else { continue }
            printedSizeByMarker[mid] = overridePrintedSizeMM ?? sPrinted
            let posW = SIMD3<Double>(x, y, z)
            let camPosW: SIMD3<Double> = {
                guard let p = posesByIdx[idx] else { return .zero }
                return SIMD3<Double>(p.cMat.columns.3.x, p.cMat.columns.3.y, p.cMat.columns.3.z)
            }()
            let s = DetectedMarkerSighting(
                markerID: mid,
                cornersUV: [
                    SIMD2<Double>(c0u, c0v),
                    SIMD2<Double>(c1u, c1v),
                    SIMD2<Double>(c2u, c2v),
                    SIMD2<Double>(c3u, c3v),
                ],
                positionWorld: posW,
                quaternionXYZW: SIMD4<Double>(qx, qy, qz, qw),
                sizeMMPrinted: sPrinted,
                sizeMMMeasured: sMeasured,
                rangeMeters: simd_length(posW - camPosW)
            )
            originalSightings.append(ReplayMarkerSighting(timestampS: t, savedIndex: idx, sighting: s))
        }

        // 3. Parse stills.csv
        var stillPathByIdx: [Int: String] = [:]
        var stillExposureByIdx: [Int: Double] = [:]
        let stillsURL = dirURL.appendingPathComponent("stills.csv")
        if FileManager.default.fileExists(atPath: stillsURL.path),
           let stillsText = try? String(contentsOf: stillsURL, encoding: .utf8) {
            for line in stillsText.split(separator: "\n").dropFirst() {
                let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                guard cols.count >= 3, let idx = Int(cols[1]) else { continue }
                stillPathByIdx[idx] = cols[2]
                if cols.count >= 17, let expS = Double(cols[16]) {
                    stillExposureByIdx[idx] = expS
                }
            }
        }

        // 3b. Parse events.csv & guidance.csv for operator taps and loop_closed anchor (`YS-67`, `YS-71`)
        let operatorTapKinds: Set<String> = ["marker", "note", "tape_a", "tape_b"]
        var operatorTapTimes: [Double] = []
        var loopClosedTimes: [Double] = []
        let eventsURL = dirURL.appendingPathComponent("events.csv")
        if FileManager.default.fileExists(atPath: eventsURL.path),
           let eventsText = try? String(contentsOf: eventsURL, encoding: .utf8) {
            for line in eventsText.split(separator: "\n").dropFirst() {
                let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                guard cols.count >= 3, let t = Double(cols[0]) else { continue }
                let kind = cols[2].trimmingCharacters(in: .whitespacesAndNewlines)
                if operatorTapKinds.contains(kind) {
                    operatorTapTimes.append(t)
                }
                if kind == "loop_closed" {
                    loopClosedTimes.append(t)
                }
            }
        }
        let guidanceURL = dirURL.appendingPathComponent("guidance.csv")
        if FileManager.default.fileExists(atPath: guidanceURL.path),
           let guidanceText = try? String(contentsOf: guidanceURL, encoding: .utf8) {
            for line in guidanceText.split(separator: "\n").dropFirst() {
                let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                guard cols.count >= 3, let t = Double(cols[0]) else { continue }
                let ev = cols[2].trimmingCharacters(in: .whitespacesAndNewlines)
                if ev == "loop_closed" {
                    loopClosedTimes.append(t)
                }
            }
        }

        // 4. Parse frames.csv into ordered `CaptureFrameInput`s
        let framesText = try String(contentsOf: dirURL.appendingPathComponent("frames.csv"), encoding: .utf8)
        var frames: [CaptureFrameInput] = []
        var savedCount = 0
        var droppedCount = 0
        for line in framesText.split(separator: "\n").dropFirst() {
            let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard cols.count >= 6, let t = Double(cols[0]) else { continue }
            let sIdx = Int(cols[1])
            let isDropped = (cols[2].lowercased() == "true")
            if isDropped {
                droppedCount += 1
            } else {
                savedCount += 1
            }
            let pose = sIdx.flatMap { posesByIdx[$0] }
            frames.append(CaptureFrameInput(
                timestampS: t,
                savedIndex: sIdx,
                dropped: isDropped,
                dropReason: cols[3],
                trackingState: cols[4],
                trackingReason: cols[5],
                cameraToWorld: pose?.cMat,
                fx: pose?.fx ?? 0.0,
                fy: pose?.fy ?? 0.0,
                cx: pose?.cx ?? 0.0,
                cy: pose?.cy ?? 0.0,
                rgbWidth: pose?.w ?? 1920,
                rgbHeight: pose?.h ?? 1440
            ))
        }

        // 5. Run marker detection & 3D LiDAR plane measurement across candidate frames in parallel
        let detectionsLock = OSAllocatedUnfairLock(initialState: [Int: [DetectedMarkerSighting]]())
        let detected2DLock = OSAllocatedUnfairLock(initialState: [Int: Set<Int>]())
        let immutablePoses = posesByIdx
        let immutablePrintedSizes = printedSizeByMarker

        switch mode {
        case .liveDetectorOnStills:
            let sortedStillPairs = stillPathByIdx.sorted { $0.key < $1.key }
            DispatchQueue.concurrentPerform(iterations: sortedStillPairs.count) { i in
                let (sIdx, relPath) = sortedStillPairs[i]
                guard let pose = immutablePoses[sIdx],
                      let jpegData = try? Data(contentsOf: dirURL.appendingPathComponent(relPath)),
                      let (luma, w, h) = ArUcoDetector.decodeLumaFromJPEGData(jpegData),
                      let (dMM, dW, dH) = loadDepth16BitPNG(at: dirURL.appendingPathComponent(String(format: "depth/%06d.png", sIdx))),
                      let (cMap, _, _) = loadConfidence8BitPNG(at: dirURL.appendingPathComponent(String(format: "confidence/%06d.png", sIdx)))
                else { return }
                let dets2D = ArUcoDetector.detectMarkers2D(lumaBytes: luma, width: w, height: h, maxCornerShiftRatio: 0.03)
                if !dets2D.isEmpty {
                    let ids2D = Set(dets2D.map(\.markerID))
                    detected2DLock.withLock { state in
                        state[sIdx] = ids2D
                    }
                }
                var measured: [DetectedMarkerSighting] = []
                for d2 in dets2D {
                    let pSize = immutablePrintedSizes[d2.markerID] ?? defaultPrintedSizeMM
                    if let s3D = ArUcoDetector.measureMarker3DOnLiDARPlane(
                        markerID: d2.markerID,
                        cornersUV: d2.cornersUV,
                        printedSizeMM: pSize,
                        depthMapMM: dMM,
                        confidenceMap: cMap,
                        depthWidth: dW,
                        depthHeight: dH,
                        rgbWidth: pose.w,
                        rgbHeight: pose.h,
                        fx: pose.fx,
                        fy: pose.fy,
                        cx: pose.cx,
                        cy: pose.cy,
                        cameraToWorld: pose.cMat,
                        maxEdgeDisagreementRatio: 0.04,
                        maxPlaneTiltFromViewDeg: 60.0
                    ) {
                        measured.append(s3D)
                    }
                }
                if !measured.isEmpty {
                    let snapshot = measured
                    detectionsLock.withLock { state in
                        state[sIdx] = snapshot
                    }
                }
            }

        case .recordedCornersFromCSV:
            let groupedOrig = Dictionary(grouping: originalSightings, by: \.savedIndex)
            let sortedKeys = groupedOrig.keys.sorted()
            DispatchQueue.concurrentPerform(iterations: sortedKeys.count) { i in
                let sIdx = sortedKeys[i]
                guard let rows = groupedOrig[sIdx],
                      let pose = immutablePoses[sIdx],
                      let (dMM, dW, dH) = loadDepth16BitPNG(at: dirURL.appendingPathComponent(String(format: "depth/%06d.png", sIdx))),
                      let (cMap, _, _) = loadConfidence8BitPNG(at: dirURL.appendingPathComponent(String(format: "confidence/%06d.png", sIdx)))
                else { return }
                var measured: [DetectedMarkerSighting] = []
                for r in rows {
                    // In `.recordedCornersFromCSV` (YS-66 historical reproduction mode), disable the YS-69 4% edge gate
                    // and use the original 75° plane tilt so all 432 recorded rows reconstruct within 0.01 mm.
                    if let s3D = ArUcoDetector.measureMarker3DOnLiDARPlane(
                        markerID: r.sighting.markerID,
                        cornersUV: r.sighting.cornersUV,
                        printedSizeMM: r.sighting.sizeMMPrinted,
                        depthMapMM: dMM,
                        confidenceMap: cMap,
                        depthWidth: dW,
                        depthHeight: dH,
                        rgbWidth: pose.w,
                        rgbHeight: pose.h,
                        fx: pose.fx,
                        fy: pose.fy,
                        cx: pose.cx,
                        cy: pose.cy,
                        cameraToWorld: pose.cMat,
                        maxEdgeDisagreementRatio: 0.0,
                        maxPlaneTiltFromViewDeg: 75.0
                    ) {
                        measured.append(s3D)
                    }
                }
                if !measured.isEmpty {
                    let snapshot = measured
                    detectionsLock.withLock { state in
                        state[sIdx] = snapshot
                    }
                }
            }
        }
        let detectionsBySavedIndex = detectionsLock.withLock { $0 }
        let detected2DBySavedIndex = detected2DLock.withLock { $0 }

        // 6. Feed all frames sequentially in timestamp order through the phone's rules
        var phoneLoopState = PhoneLoopRuleState()
        phoneLoopState.walkStartTimeS = orderedPoses.first?.t
        phoneLoopState.walkEndTimeS = orderedPoses.last?.t
        phoneLoopState.loopClosedEventTimeS = loopClosedTimes.max()
        let filteredTaps = operatorTapTimes.filter { tapT in
            guard let lcT = phoneLoopState.loopClosedEventTimeS else { return true }
            return tapT <= lcT + 1e-6
        }
        phoneLoopState.lastOperatorTapTimeS = filteredTaps.max()

        let hasGuidanceStream = FileManager.default.fileExists(atPath: guidanceURL.path)
        var guidedEngine = GuidedCaptureEngine(
            requireLookUpStep: hasGuidanceStream,
            preWalkTapeMeters0To1: preWalkTapeMeters0To1
        )
        var replayedSightings: [ReplayMarkerSighting] = []
        var prevFrameT: Double? = nil

        for frame in frames {
            guard !frame.dropped,
                   let sIdx = frame.savedIndex,
                   let camPosW = frame.cameraPositionWorld
            else { continue }
            let dt = prevFrameT.map { max(1e-4, frame.timestampS - $0) } ?? (1.0 / 30.0)
            prevFrameT = frame.timestampS

            phoneLoopState.updateCameraExcursion(
                cameraPositionWorld: camPosW,
                savedIndex: sIdx,
                timestampS: frame.timestampS
            )
            let dets = detectionsBySavedIndex[sIdx] ?? []
            if !dets.isEmpty {
                phoneLoopState.ingestSightings(
                    dets,
                    timestampS: frame.timestampS,
                    savedIndex: sIdx,
                    cameraPositionWorld: camPosW
                )
                for d in dets {
                    replayedSightings.append(ReplayMarkerSighting(timestampS: frame.timestampS, savedIndex: sIdx, sighting: d))
                }
            }

            let pose = posesByIdx[sIdx]
            let cMat = pose?.cMat ?? matrix_identity_double4x4
            let lookRayY = Double(-cMat.columns.2.y)
            let pitchDeg = asin(min(1.0, max(-1.0, lookRayY))) * 180.0 / .pi
            let tr = pose?.turnRateRadS ?? 0.0
            let expS = stillExposureByIdx[sIdx] ?? (1.0 / 120.0)
            let meanF = pose.map { 0.5 * ($0.fx + $0.fy) } ?? 1440.0
            let predBlur = ExposureAndStillnessGate.computePredictedBlurPx(
                focalLengthPx: meanF,
                turnRateRadS: tr,
                exposureDurationS: expS
            )
            let steadyDets = dets.filter { s in
                let rngXZ = hypot(camPosW.x - s.positionWorld.x, camPosW.z - s.positionWorld.z)
                return ExposureAndStillnessGate.evaluateMarkerSighting(
                    predictedBlurPx: predBlur,
                    turnRateRadS: tr,
                    rangeMeters3D: s.rangeMeters,
                    horizontalRangeMeters: rngXZ
                ).passes
            }
            _ = guidedEngine.step(
                timestampS: frame.timestampS,
                savedIndex: sIdx,
                dt: dt,
                trackingState: frame.trackingState,
                trackingReason: frame.trackingReason,
                pitchDeg: pitchDeg,
                turnRateRadS: tr,
                predictedBlurPx: predBlur,
                cameraPositionWorld: camPosW,
                confirmedSightings: steadyDets,
                rawSightings: dets,
                unifiedLoopClosed: phoneLoopState.unifiedLoopClosed,
                hasValidDepthForLookUp: true,
                elevatedPointsAbove2m: pitchDeg >= 15.0 ? 10 : 0,
                didRunMarkerDetection: !dets.isEmpty
            )
        }

        // Finalize unified YS-71 loop closure evaluation at the end of the walk
        let confirmedM0Stills: Set<Int>? = (mode == .liveDetectorOnStills)
            ? Set(replayedSightings.filter { $0.sighting.markerID == 0 }.map(\.savedIndex))
            : nil
        phoneLoopState.evaluateUnifiedLoopClosure(
            confirmedMarker0StillIndices: confirmedM0Stills,
            requireAllFourMarkersOrEndEarly: true
        )
        if let lastFrame = frames.last(where: { !$0.dropped }),
           let sIdx = lastFrame.savedIndex,
           let camPosW = lastFrame.cameraPositionWorld {
            _ = guidedEngine.step(
                timestampS: lastFrame.timestampS + 0.01,
                savedIndex: sIdx,
                dt: 0.01,
                trackingState: lastFrame.trackingState,
                trackingReason: lastFrame.trackingReason,
                pitchDeg: -30.0,
                turnRateRadS: 0.0,
                predictedBlurPx: 0.0,
                cameraPositionWorld: camPosW,
                confirmedSightings: [],
                rawSightings: [],
                unifiedLoopClosed: phoneLoopState.unifiedLoopClosed,
                didRunMarkerDetection: false
            )
        }

        // 7. Evaluate YS-70 confirmation and YS-73 stillness & distance gate across sightings
        var openCVRejectedCount = 0
        var ys73FlaggedOfRejectedCount = 0
        var ys73KeptByMarker: [Int: Int] = [:]

        if mode == .liveDetectorOnStills {
            for orig in originalSightings {
                let sIdx = orig.savedIndex
                let mid = orig.sighting.markerID
                guard let pose = posesByIdx[sIdx] else { continue }
                let camPos = SIMD3<Double>(pose.cMat.columns.3.x, pose.cMat.columns.3.y, pose.cMat.columns.3.z)
                let mPos = orig.sighting.positionWorld
                let rng3D = simd_length(camPos - mPos)
                let rngXZ = hypot(camPos.x - mPos.x, camPos.z - mPos.z)
                let expS = stillExposureByIdx[sIdx] ?? (1.0 / 120.0)
                let predBlur = ExposureAndStillnessGate.computePredictedBlurPx(
                    focalLengthPx: 0.5 * (pose.fx + pose.fy),
                    turnRateRadS: pose.turnRateRadS,
                    exposureDurationS: expS
                )
                let verdict = ExposureAndStillnessGate.evaluateMarkerSighting(
                    predictedBlurPx: predBlur,
                    turnRateRadS: pose.turnRateRadS,
                    rangeMeters3D: rng3D,
                    horizontalRangeMeters: rngXZ
                )

                let hasStill = (stillPathByIdx[sIdx] != nil)
                let confirmed2D = detected2DBySavedIndex[sIdx]?.contains(mid) ?? false
                if hasStill && !confirmed2D {
                    openCVRejectedCount += 1
                    if !verdict.passes {
                        ys73FlaggedOfRejectedCount += 1
                    }
                } else if verdict.passes {
                    ys73KeptByMarker[mid, default: 0] += 1
                }
            }
        }

        // Under YS-70, in `.liveDetectorOnStills`, every replayed sighting comes from a saved still JPEG and is
        // confirmed by OpenCV on that JPEG (`unconfirmedSightingsCount == 0`, `markerOpenCVDisagreementCount == 0`).
        let unconfirmedCount = (mode == .liveDetectorOnStills)
            ? replayedSightings.filter { stillPathByIdx[$0.savedIndex] == nil }.count
            : 0
        let disagreementCount = (mode == .liveDetectorOnStills)
            ? replayedSightings.filter { !(detected2DBySavedIndex[$0.savedIndex]?.contains($0.sighting.markerID) ?? false) }.count
            : 0

        return CaptureReplayResult(
            sessionID: sessionID,
            totalFrameRows: frames.count,
            savedFrameCount: savedCount,
            droppedFrameCount: droppedCount,
            stillCount: stillPathByIdx.count,
            originalSightings: originalSightings,
            replayedSightings: replayedSightings,
            phoneLoopState: phoneLoopState,
            guidedEngine: guidedEngine,
            elapsedSeconds: CFAbsoluteTimeGetCurrent() - tStart,
            openCVRejectedOriginalSightingsCount: openCVRejectedCount,
            ys73FlaggedOfOpenCVRejectedCount: ys73FlaggedOfRejectedCount,
            ys73KeptGoodSightingsByMarker: ys73KeptByMarker,
            unconfirmedSightingsCount: unconfirmedCount,
            markerOpenCVDisagreementCount: disagreementCount
        )
    }
}
