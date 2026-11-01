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

/// Encapsulates the phone's marker-excursion and loop-closure tracking state (`YS-66`),
/// shared between `CaptureSessionRecorder` and `CaptureReplayHarness`.
public struct PhoneLoopRuleState: Sendable {
    public var marker0FirstPassPos: SIMD3<Double>? = nil
    public var marker0FirstPassTimeS: Double? = nil
    public var marker0MaxExcursionM: Double = 0.0
    public var marker0LastPassPos: SIMD3<Double>? = nil
    public var marker0LastPassTimeS: Double? = nil
    public var loopClosureGapMeters: Double? = nil
    public var detectedMarkerIDs: Set<Int> = []

    public init() {}

    /// Update camera excursion from Marker 0 Pass 1 for a delivered frame.
    public mutating func updateCameraExcursion(cameraPositionWorld camPosW: SIMD3<Double>) {
        if let m0Pos = marker0FirstPassPos {
            let distFromM0 = simd_length(camPosW - m0Pos)
            if distFromM0 > marker0MaxExcursionM {
                marker0MaxExcursionM = distFromM0
            }
        }
    }

    /// Ingest 3D marker sightings detected on a frame at `timestampS` with camera position `camPosW`.
    /// Returns `true` if `loopClosureGapMeters` was updated on this frame.
    @discardableResult
    public mutating func ingestSightings(
        _ sightings: [DetectedMarkerSighting],
        timestampS: Double,
        cameraPositionWorld camPosW: SIMD3<Double>
    ) -> Bool {
        var updatedLoop = false
        for m in sightings {
            detectedMarkerIDs.insert(m.markerID)
            if m.markerID == 0 {
                if marker0FirstPassPos == nil {
                    marker0FirstPassPos = m.positionWorld
                    marker0FirstPassTimeS = timestampS
                } else if let firstT = marker0FirstPassTimeS,
                          let firstPos = marker0FirstPassPos,
                          (timestampS - firstT) >= 3.0,
                          (marker0MaxExcursionM >= 1.0 || (timestampS - firstT) >= 10.0),
                          simd_length(camPosW - m.positionWorld) <= 1.50 {
                    marker0LastPassPos = m.positionWorld
                    marker0LastPassTimeS = timestampS
                    let gap = simd_length(m.positionWorld - firstPos)
                    loopClosureGapMeters = gap
                    updatedLoop = true
                }
            }
        }
        return updatedLoop
    }
}

/// Toolkit reference verdict loaded from the checked-in Python toolkit golden file
/// (`ios/Yardstick/Tests/Fixtures/toolkit_golden_walks_1_5.json`, generated by
/// `toolkit/scripts/generate_ys66_replay_fixtures.py`).
/// Note (`YS-66` / `YS-71`): Before `YS-71`, toolkit verdicts come strictly from the Python toolkit's
/// golden output rather than a duplicate Swift port.
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
    /// Decode saved JPEGs (`stills/%06d.jpg`) to BT.601 full-range luma and run `ArUcoDetector.detectMarkers2D` + 3D LiDAR plane fit.
    case liveDetectorOnStills
    /// Use the 2D corner UV coordinates recorded in `markers.csv` and run `ArUcoDetector.measureMarker3DOnLiDARPlane` against the depth/confidence PNGs.
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
    public let elapsedSeconds: Double

    /// Compute the median `sizeMMMeasured` for `markerID` in `sightings`.
    public static func medianSizeMM(for markerID: Int, in sightings: [ReplayMarkerSighting]) -> Double? {
        let vals = sightings.filter { $0.sighting.markerID == markerID }.map(\.sighting.sizeMMMeasured).sorted()
        guard !vals.isEmpty else { return nil }
        let mid = vals.count / 2
        return (vals.count % 2 == 1) ? vals[mid] : 0.5 * (vals[mid - 1] + vals[mid])
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
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        let w = cgImg.width
        let h = cgImg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                  data: &rgba,
                  width: w,
                  height: h,
                  bitsPerComponent: 8,
                  bytesPerRow: w * 4,
                  space: cs,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              )
        else { return nil }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var luma = [UInt8](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let r = Int(rgba[i * 4])
            let g = Int(rgba[i * 4 + 1])
            let b = Int(rgba[i * 4 + 2])
            luma[i] = UInt8(clamping: (77 * r + 150 * g + 29 * b + 128) >> 8)
        }
        return (luma, w, h)
    }

    /// Replay a recorded capture folder on disk (`frames.csv`, `poses.csv`, `stills.csv`, `markers.csv`, `events.csv`,
    /// `stills/*.jpg`, `depth/*.png`, `confidence/*.png`) through the phone's detector and rules (`YS-66`).
    public static func replayCaptureDirectory(
        at dirURL: URL,
        mode: CaptureReplayDetectionMode = .liveDetectorOnStills,
        defaultPrintedSizeMM: Double = 180.0
    ) throws -> CaptureReplayResult {
        let tStart = CFAbsoluteTimeGetCurrent()
        let sessionID = dirURL.lastPathComponent

        // 1. Parse poses.csv
        let posesText = try String(contentsOf: dirURL.appendingPathComponent("poses.csv"), encoding: .utf8)
        struct PoseEntry: Sendable {
            let t: Double
            let savedIndex: Int
            let cMat: simd_double4x4
            let fx: Double, fy: Double, cx: Double, cy: Double
            let w: Int, h: Int
        }
        var posesByIdx: [Int: PoseEntry] = [:]
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
            posesByIdx[idx] = PoseEntry(t: t, savedIndex: idx, cMat: cMat, fx: fx, fy: fy, cx: cx, cy: cy, w: w, h: h)
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
            printedSizeByMarker[mid] = sPrinted
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
        let stillsURL = dirURL.appendingPathComponent("stills.csv")
        if FileManager.default.fileExists(atPath: stillsURL.path),
           let stillsText = try? String(contentsOf: stillsURL, encoding: .utf8) {
            for line in stillsText.split(separator: "\n").dropFirst() {
                let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                guard cols.count >= 3, let idx = Int(cols[1]) else { continue }
                stillPathByIdx[idx] = cols[2]
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
        let immutablePoses = posesByIdx
        let immutablePrintedSizes = printedSizeByMarker

        switch mode {
        case .liveDetectorOnStills:
            let sortedStillPairs = stillPathByIdx.sorted { $0.key < $1.key }
            DispatchQueue.concurrentPerform(iterations: sortedStillPairs.count) { i in
                let (sIdx, relPath) = sortedStillPairs[i]
                guard let pose = immutablePoses[sIdx],
                      let (luma, w, h) = loadLumaFromJPEG(at: dirURL.appendingPathComponent(relPath)),
                      let (dMM, dW, dH) = loadDepth16BitPNG(at: dirURL.appendingPathComponent(String(format: "depth/%06d.png", sIdx))),
                      let (cMap, _, _) = loadConfidence8BitPNG(at: dirURL.appendingPathComponent(String(format: "confidence/%06d.png", sIdx)))
                else { return }
                let dets2D = ArUcoDetector.detectMarkers2D(lumaBytes: luma, width: w, height: h)
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
                        cameraToWorld: pose.cMat
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
                        cameraToWorld: pose.cMat
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

        // 6. Feed all frames sequentially in timestamp order through the phone's rules
        var phoneLoopState = PhoneLoopRuleState()
        var replayedSightings: [ReplayMarkerSighting] = []
        for frame in frames {
            guard !frame.dropped,
                  let sIdx = frame.savedIndex,
                  let camPosW = frame.cameraPositionWorld
            else { continue }
            phoneLoopState.updateCameraExcursion(cameraPositionWorld: camPosW)
            if let dets = detectionsBySavedIndex[sIdx] {
                phoneLoopState.ingestSightings(dets, timestampS: frame.timestampS, cameraPositionWorld: camPosW)
                for d in dets {
                    replayedSightings.append(ReplayMarkerSighting(timestampS: frame.timestampS, savedIndex: sIdx, sighting: d))
                }
            }
        }

        return CaptureReplayResult(
            sessionID: sessionID,
            totalFrameRows: frames.count,
            savedFrameCount: savedCount,
            droppedFrameCount: droppedCount,
            stillCount: stillPathByIdx.count,
            originalSightings: originalSightings,
            replayedSightings: replayedSightings,
            phoneLoopState: phoneLoopState,
            elapsedSeconds: CFAbsoluteTimeGetCurrent() - tStart
        )
    }
}

