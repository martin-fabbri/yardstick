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

/// Toolkit reference loop-closure verdict (`toolkit/src/yardstick/geometry.py:500–575`),
/// used by `YS-66` to compare the phone's verdict against the toolkit's on recorded walks.
public struct ToolkitLoopClosureVerdict: Sendable {
    public let closed: Bool
    public let reason: String
    public let spreadMeters: Double?
    public let passCount: Int
    public let openingSightings: Int
    public let closingSightings: Int
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
    public let toolkitLoopVerdict: ToolkitLoopClosureVerdict
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

    /// Compute the toolkit's reference marker loop-closure verdict (`toolkit/src/yardstick/geometry.py:326–360, 500–575`).
    public static func computeToolkitLoopClosure(
        sightings: [ReplayMarkerSighting],
        frames: [CaptureFrameInput],
        events: [(t: Double, kind: String)]
    ) -> ToolkitLoopClosureVerdict {
        let savedFrames = frames.filter { !$0.dropped && $0.savedIndex != nil }
        guard let walkStartT = savedFrames.first?.timestampS,
              let walkEndT = savedFrames.last?.timestampS
        else {
            return ToolkitLoopClosureVerdict(closed: false, reason: "no_frames", spreadMeters: nil, passCount: 0, openingSightings: 0, closingSightings: 0)
        }

        var cameraPosByIndex: [Int: SIMD3<Double>] = [:]
        for f in savedFrames {
            if let sIdx = f.savedIndex, let camP = f.cameraPositionWorld {
                cameraPosByIndex[sIdx] = camP
            }
        }

        let lastTapTS: Double? = events
            .filter { $0.kind != "done" && !$0.kind.isEmpty }
            .map(\.t)
            .max()

        let latestOtherMarkerTS: Double? = sightings
            .filter { $0.sighting.markerID != 0 }
            .map(\.timestampS)
            .max()

        let m0Sightings = sightings
            .filter { $0.sighting.markerID == 0 }
            .sorted { $0.timestampS < $1.timestampS }

        guard m0Sightings.count >= 2 else {
            return ToolkitLoopClosureVerdict(closed: false, reason: "insufficient_marker_0_sightings", spreadMeters: nil, passCount: m0Sightings.isEmpty ? 0 : 1, openingSightings: m0Sightings.count, closingSightings: 0)
        }

        // Group Marker 0 sightings into passes (`cluster_marker_passes` in `geometry.py:326–360`:
        // `dt >= 3.0` AND (`max_exc >= 1.0 || dt >= 10.0`))
        var passes: [[ReplayMarkerSighting]] = [[m0Sightings[0]]]
        for s in m0Sightings.dropFirst() {
            let prev = passes[passes.count - 1].last!
            let dt = s.timestampS - prev.timestampS
            if dt >= 3.0 {
                let idxPrev = prev.savedIndex
                let idxCurr = s.savedIndex
                let mPos = prev.sighting.positionWorld
                var maxExc = 0.0
                if idxPrev <= idxCurr {
                    for k in idxPrev...idxCurr {
                        if let camP = cameraPosByIndex[k] {
                            maxExc = max(maxExc, simd_length(camP - mPos))
                        }
                    }
                }
                if maxExc >= 1.0 || dt >= 10.0 {
                    passes.append([s])
                    continue
                }
            }
            passes[passes.count - 1].append(s)
        }

        guard passes.count >= 2 else {
            return ToolkitLoopClosureVerdict(
                closed: false,
                reason: "single_pass_only",
                spreadMeters: nil,
                passCount: passes.count,
                openingSightings: passes.first?.count ?? 0,
                closingSightings: 0
            )
        }

        let firstPass = passes[0]
        let lastPass = passes[passes.count - 1]
        let walkSpanS = (walkEndT >= walkStartT) ? (walkEndT - walkStartT) : 0.0

        // Filter closing pass (`geometry.py:528–551`)
        let validClosing = lastPass.filter { r in
            let rT = r.timestampS
            if (walkEndT - rT) > 30.0 { return false }
            if walkSpanS > 5.0 && (walkEndT - rT) > min(30.0, 0.35 * walkSpanS) { return false }
            if let tapT = lastTapTS, rT < tapT - 1e-6 { return false }
            if let otherT = latestOtherMarkerTS, rT < otherT - 1e-6 { return false }
            guard let camP = cameraPosByIndex[r.savedIndex] else { return false }
            if simd_length(camP - r.sighting.positionWorld) > 1.50 { return false }
            return true
        }

        guard validClosing.count >= 1 else {
            return ToolkitLoopClosureVerdict(
                closed: false,
                reason: "closing_pass_filtered_out",
                spreadMeters: nil,
                passCount: passes.count,
                openingSightings: firstPass.count,
                closingSightings: validClosing.count
            )
        }

        func medianVec3(_ rows: [ReplayMarkerSighting]) -> SIMD3<Double> {
            let xs = rows.map(\.sighting.positionWorld.x).sorted()
            let ys = rows.map(\.sighting.positionWorld.y).sorted()
            let zs = rows.map(\.sighting.positionWorld.z).sorted()
            func med(_ a: [Double]) -> Double {
                let m = a.count / 2
                return (a.count % 2 == 1) ? a[m] : 0.5 * (a[m - 1] + a[m])
            }
            return SIMD3<Double>(med(xs), med(ys), med(zs))
        }

        let openMed = medianVec3(firstPass)
        let closeMed = medianVec3(validClosing)
        let spread = simd_length(closeMed - openMed)
        return ToolkitLoopClosureVerdict(
            closed: true,
            reason: "ok",
            spreadMeters: spread,
            passCount: passes.count,
            openingSightings: firstPass.count,
            closingSightings: validClosing.count
        )
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

        // 3. Parse events.csv
        var events: [(t: Double, kind: String)] = []
        let eventsURL = dirURL.appendingPathComponent("events.csv")
        if FileManager.default.fileExists(atPath: eventsURL.path),
           let eventsText = try? String(contentsOf: eventsURL, encoding: .utf8) {
            for line in eventsText.split(separator: "\n").dropFirst() {
                let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
                guard cols.count >= 3, let t = Double(cols[0]) else { continue }
                events.append((t: t, kind: cols[2]))
            }
        }

        // 4. Parse stills.csv
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

        // 5. Parse frames.csv into ordered `CaptureFrameInput`s
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

        // 6. Run marker detection & 3D LiDAR plane measurement across candidate frames in parallel
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

        // 7. Feed all frames sequentially in timestamp order through the phone's rules
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

        // 8. Compute toolkit reference verdict on the same sightings
        let toolkitVerdict = computeToolkitLoopClosure(
            sightings: replayedSightings,
            frames: frames,
            events: events
        )

        return CaptureReplayResult(
            sessionID: sessionID,
            totalFrameRows: frames.count,
            savedFrameCount: savedCount,
            droppedFrameCount: droppedCount,
            stillCount: stillPathByIdx.count,
            originalSightings: originalSightings,
            replayedSightings: replayedSightings,
            phoneLoopState: phoneLoopState,
            toolkitLoopVerdict: toolkitVerdict,
            elapsedSeconds: CFAbsoluteTimeGetCurrent() - tStart
        )
    }
}

