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
import CoreVideo
import Foundation
import Vision
import simd

/// Clean-room ArUco `DICT_5X5_250` marker detector and LiDAR depth plane corner projector (`YS-30`, `YS-33`, `Q007`, `Q027`).
///
/// Uses the bit patterns in `ArUcoDictionaryData` (generated from OpenCV's published `DICT_5X5_250`
/// definition via `toolkit/scripts/generate_aruco_dict.py`), detects 2D square marker quads in the camera
/// luma plane, decodes the 7×7 grid (1-cell black border + 5×5 inner code), and refines 3D marker pose
/// and measured edge length on the interior LiDAR depth plane (10% inward border erosion).
public struct DetectedMarkerSighting: Sendable {
    public let markerID: Int
    public let cornersUV: [SIMD2<Double>]  // 4 corners in RGB pixel coordinates (TL, TR, BR, BL)
    public let positionWorld: SIMD3<Double>
    public let quaternionXYZW: SIMD4<Double>
    public let sizeMMPrinted: Double
    public let sizeMMMeasured: Double
    public let rangeMeters: Double
}

public enum ArUcoDetector {
    /// Rotate a 25-bit 5x5 grid 90 degrees clockwise.
    public static func rotateBits90CW(_ bits: UInt32) -> UInt32 {
        var out: UInt32 = 0
        for r in 0..<5 {
            for c in 0..<5 {
                let srcBitIndex = 24 - (r * 5 + c)
                let bit = (bits >> srcBitIndex) & 1
                let dstR = c
                let dstC = 4 - r
                let dstBitIndex = 24 - (dstR * 5 + dstC)
                out |= (bit << dstBitIndex)
            }
        }
        return out
    }

    /// Match a 25-bit 5x5 inner payload against `ArUcoDictionary5x5.patternsByID` across all 4 rotations (Hamming distance <= 1).
    /// Returns `(markerID, rotationCount90CW)` or `nil`.
    public static func matchCode5x5(_ observedBits: UInt32, maxHamming: Int = 1) -> (Int, Int)? {
        var bestID = -1
        var bestRot = 0
        var bestDist = maxHamming + 1

        for (id, rotList) in ArUcoDictionary5x5.patternsByID {
            for (rot, pattern) in rotList.enumerated() {
                let dist = (observedBits ^ pattern).nonzeroBitCount
                if dist < bestDist {
                    bestDist = dist
                    bestID = id
                    bestRot = rot
                    if dist == 0 {
                        return (bestID, bestRot)
                    }
                }
            }
        }
        if bestID >= 0 && bestDist <= maxHamming {
            return (bestID, bestRot)
        }
        return nil
    }

    /// Detect 2D ArUco `DICT_5X5_250` markers in an 8-bit luma plane (`width` x `height`).
    /// Returns an array of `(markerID, canonicalCornersUV)` where `canonicalCornersUV` are ordered `(TL, TR, BR, BL)`.
    public static func detectMarkers2D(
        lumaBytes: [UInt8],
        width: Int,
        height: Int
    ) -> [(markerID: Int, cornersUV: [SIMD2<Double>])] {
        guard lumaBytes.count == width * height, width >= 64, height >= 64 else {
            return []
        }

        var candidateQuads: [[SIMD2<Double>]] = []

        // 1. Vision rectangle detector for fast quadrilateral candidates
        if let cgImage = makeGrayCGImage(lumaBytes: lumaBytes, width: width, height: height) {
            let req = VNDetectRectanglesRequest()
            req.maximumObservations = 16
            req.minimumAspectRatio = 0.35
            req.maximumAspectRatio = 1.0
            req.quadratureTolerance = 30.0
            req.minimumSize = 0.025
            req.minimumConfidence = 0.4
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([req])
            if let results = req.results {
                for obs in results {
                    // Vision normalized coordinates have origin at bottom-left; convert to top-left pixel (u, v)
                    let tl = SIMD2<Double>(Double(obs.topLeft.x) * Double(width), (1.0 - Double(obs.topLeft.y)) * Double(height))
                    let tr = SIMD2<Double>(Double(obs.topRight.x) * Double(width), (1.0 - Double(obs.topRight.y)) * Double(height))
                    let br = SIMD2<Double>(Double(obs.bottomRight.x) * Double(width), (1.0 - Double(obs.bottomRight.y)) * Double(height))
                    let bl = SIMD2<Double>(Double(obs.bottomLeft.x) * Double(width), (1.0 - Double(obs.bottomLeft.y)) * Double(height))
                    candidateQuads.append([tl, tr, br, bl])
                    // Also add an 80% inset candidate in case Vision detected the outer white paper border around a marker
                    let center = (tl + tr + br + bl) * 0.25
                    candidateQuads.append([
                        center + 0.80 * (tl - center),
                        center + 0.80 * (tr - center),
                        center + 0.80 * (br - center),
                        center + 0.80 * (bl - center),
                    ])
                }
            }
        }

        // 2. Decode each candidate quad against the 7x7 ArUco grid (1-cell black border + 5x5 code)
        var detectedByID: [Int: [SIMD2<Double>]] = [:]
        for quad in candidateQuads {
            if let (mid, canonicalCorners) = decodeQuad7x7(quad: quad, lumaBytes: lumaBytes, width: width, height: height) {
                detectedByID[mid] = canonicalCorners
            }
        }

        return detectedByID.keys.sorted().compactMap { mid in
            guard let c = detectedByID[mid] else { return nil }
            return (markerID: mid, cornersUV: c)
        }
    }

    /// Sample a 7x7 ArUco grid inside `quad` (`[TL, TR, BR, BL]`), verify the 1-cell black border and outer quiet zone,
    /// and match the inner 5x5 bit pattern against `ArUcoDictionary5x5`.
    private static func decodeQuad7x7(
        quad: [SIMD2<Double>],
        lumaBytes: [UInt8],
        width: Int,
        height: Int
    ) -> (Int, [SIMD2<Double>])? {
        guard quad.count == 4 else { return nil }

        func sampleLuma(s: Double, t: Double) -> Double? {
            let pTop = quad[0] * (1.0 - s) + quad[1] * s
            let pBot = quad[3] * (1.0 - s) + quad[2] * s
            let p = pTop * (1.0 - t) + pBot * t
            let u = Int(p.x.rounded())
            let v = Int(p.y.rounded())
            guard u >= 0, u < width, v >= 0, v < height else { return nil }
            return Double(lumaBytes[v * width + u])
        }

        func sampleCellMean(row: Int, col: Int) -> Double? {
            var sum = 0.0
            var count = 0
            for dy in [-0.20, 0.0, 0.20] {
                for dx in [-0.20, 0.0, 0.20] {
                    let s = (Double(col) + 0.5 + dx) / 7.0
                    let t = (Double(row) + 0.5 + dy) / 7.0
                    if let val = sampleLuma(s: s, t: t) {
                        sum += val
                        count += 1
                    }
                }
            }
            return count > 0 ? (sum / Double(count)) : nil
        }

        var grid = Array(repeating: Array(repeating: 0.0, count: 7), count: 7)
        var borderVals: [Double] = []
        var innerVals: [Double] = []

        for r in 0..<7 {
            for c in 0..<7 {
                guard let val = sampleCellMean(row: r, col: c) else { return nil }
                grid[r][c] = val
                if r == 0 || r == 6 || c == 0 || c == 6 {
                    borderVals.append(val)
                } else {
                    innerVals.append(val)
                }
            }
        }

        // Sample quiet zone outside the 7x7 border
        var quietVals: [Double] = []
        for idx in 0..<7 {
            if let vTop = sampleCellMean(row: -1, col: idx) { quietVals.append(vTop) }
            if let vBot = sampleCellMean(row: 7, col: idx) { quietVals.append(vBot) }
            if let vLeft = sampleCellMean(row: idx, col: -1) { quietVals.append(vLeft) }
            if let vRight = sampleCellMean(row: idx, col: 7) { quietVals.append(vRight) }
        }

        let borderMean = borderVals.reduce(0.0, +) / Double(max(1, borderVals.count))
        let maxInner = innerVals.max() ?? 255.0
        let quietMean = quietVals.isEmpty ? maxInner : (quietVals.reduce(0.0, +) / Double(quietVals.count))
        let brightRef = max(quietMean, maxInner)

        // Contrast gate: border must be significantly darker than white cells / quiet zone
        guard brightRef - borderMean >= 25.0 else {
            return nil
        }

        let thresh = 0.5 * (brightRef + borderMean)
        let darkBorderCount = borderVals.filter { $0 <= thresh }.count
        guard darkBorderCount >= 22 else {
            return nil
        }

        var observedBits: UInt32 = 0
        for r in 1...5 {
            for c in 1...5 {
                let bit: UInt32 = (grid[r][c] > thresh) ? 1 : 0
                let bitIndex = 24 - ((r - 1) * 5 + (c - 1))
                observedBits |= (bit << bitIndex)
            }
        }

        guard let (markerID, rotCount) = matchCode5x5(observedBits, maxHamming: 1) else {
            return nil
        }

        // Rotate corners so index 0 is canonical Top-Left of the unrotated marker
        let shift = (4 - (rotCount % 4)) % 4
        let canonicalCorners = (0..<4).map { i in quad[(i + shift) % 4] }
        return (markerID, canonicalCorners)
    }

    private static func makeGrayCGImage(lumaBytes: [UInt8], width: Int, height: Int) -> CGImage? {
        let data = Data(lumaBytes) as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)
        else {
            return nil
        }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: cs,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    /// Compute variance of discrete Laplacian on a downsampled luma grid as a fast sharpness metric (`YS-13`).
    public static func computeLaplacianSharpness(lumaBytes: [UInt8], width: Int, height: Int, step: Int = 4) -> Double {
        guard lumaBytes.count == width * height, width > 2 * step, height > 2 * step else {
            return 0.0
        }
        var sum = 0.0
        var sumSq = 0.0
        var count = 0.0
        var v = step
        while v < height - step {
            let rowOffset = v * width
            var u = step
            while u < width - step {
                let c = Int(lumaBytes[rowOffset + u])
                let n = Int(lumaBytes[(v - step) * width + u])
                let s = Int(lumaBytes[(v + step) * width + u])
                let w = Int(lumaBytes[rowOffset + (u - step)])
                let e = Int(lumaBytes[rowOffset + (u + step)])
                let lap = Double(n + s + w + e - 4 * c)
                sum += lap
                sumSq += lap * lap
                count += 1.0
                u += step
            }
            v += step
        }
        guard count > 1.0 else { return 0.0 }
        let mean = sum / count
        return (sumSq / count) - (mean * mean)
    }

    /// Project 4 RGB marker corners onto the 3D plane fitted to interior LiDAR depth pixels (10% inward border erosion, `YS-33`, `Q027`).
    public static func measureMarker3DOnLiDARPlane(
        markerID: Int,
        cornersUV: [SIMD2<Double>],
        printedSizeMM: Double,
        depthMapMM: [UInt16],
        confidenceMap: [UInt8],
        depthWidth: Int,
        depthHeight: Int,
        rgbWidth: Int,
        rgbHeight: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraToWorld: simd_double4x4
    ) -> DetectedMarkerSighting? {
        guard cornersUV.count == 4, depthMapMM.count == depthWidth * depthHeight else {
            return nil
        }

        let scaleX = Double(depthWidth) / Double(rgbWidth)
        let scaleY = Double(depthHeight) / Double(rgbHeight)
        let fxD = fx * scaleX
        let fyD = fy * scaleY
        let cxD = cx * scaleX
        let cyD = cy * scaleY

        // Erode quadrilateral 10% inward toward center in depth coordinates
        let cornersD = cornersUV.map { SIMD2<Double>($0.x * scaleX, $0.y * scaleY) }
        let centerD = (cornersD[0] + cornersD[1] + cornersD[2] + cornersD[3]) * 0.25
        let erodedD = cornersD.map { centerD + 0.90 * ($0 - centerD) }

        let minU = max(0, Int(erodedD.map(\.x).min() ?? 0))
        let maxU = min(depthWidth - 1, Int(ceil(erodedD.map(\.x).max() ?? 0)))
        let minV = max(0, Int(erodedD.map(\.y).min() ?? 0))
        let maxV = min(depthHeight - 1, Int(ceil(erodedD.map(\.y).max() ?? 0)))

        var interiorCamPts: [SIMD3<Double>] = []
        if minU <= maxU && minV <= maxV {
            for v in minV...maxV {
                for u in minU...maxU {
                    let pt = SIMD2<Double>(Double(u) + 0.5, Double(v) + 0.5)
                    if pointInQuad(pt, quad: erodedD) {
                        let idx = v * depthWidth + u
                        let dMM = depthMapMM[idx]
                        let conf = confidenceMap[idx]
                        if dMM > 0 && conf >= 1 {
                            let z = Double(dMM) / 1000.0
                            let x = (Double(u) - cxD) * z / fxD
                            let y = -(Double(v) - cyD) * z / fyD
                            interiorCamPts.append(SIMD3<Double>(x, y, -z))
                        }
                    }
                }
            }
        }

        guard interiorCamPts.count >= 4 else {
            return nil
        }

        var centroidCam = SIMD3<Double>.zero
        for p in interiorCamPts {
            centroidCam += p
        }
        centroidCam /= Double(interiorCamPts.count)
        let rangeM = simd_length(centroidCam)
        guard rangeM >= 0.60 && rangeM <= 2.00 else {
            return nil
        }

        // Plane normal facing camera in camera coordinates
        let normalCam = simd_normalize(-centroidCam)
        let dCam = -simd_dot(normalCam, centroidCam)

        // Intersect the 4 full-res RGB corner rays with the fitted LiDAR depth plane
        var cornersCam: [SIMD3<Double>] = []
        for cUV in cornersUV {
            let ray = SIMD3<Double>((cUV.x - cx) / fx, -(cUV.y - cy) / fy, -1.0)
            let denom = simd_dot(normalCam, ray)
            if abs(denom) < 1e-9 {
                return nil
            }
            let tRay = -dCam / denom
            if tRay <= 0 {
                return nil
            }
            cornersCam.append(ray * tRay)
        }

        var edgeSumM = 0.0
        for i in 0..<4 {
            edgeSumM += simd_length(cornersCam[(i + 1) % 4] - cornersCam[i])
        }
        let measuredMM = (edgeSumM / 4.0) * 1000.0

        // Transform corners and center into ARKit world coordinates
        let cornersWorld = cornersCam.map { p -> SIMD3<Double> in
            let hp = cameraToWorld * SIMD4<Double>(p.x, p.y, p.z, 1.0)
            return SIMD3<Double>(hp.x, hp.y, hp.z)
        }
        let centerWorld = (cornersWorld[0] + cornersWorld[1] + cornersWorld[2] + cornersWorld[3]) * 0.25

        // Build orthonormal marker rotation frame in world coordinates:
        // +X_m along top edge (TL -> TR), +Y_m along left edge (BL -> TL), +Z_m = X_m x Y_m
        let xAxis = simd_normalize(cornersWorld[1] - cornersWorld[0])
        let yRaw = simd_normalize(cornersWorld[0] - cornersWorld[3])
        let zAxis = simd_normalize(simd_cross(xAxis, yRaw))
        let yAxis = simd_normalize(simd_cross(zAxis, xAxis))
        let rotMat = simd_double3x3(columns: (xAxis, yAxis, zAxis))
        let quat = simd_quatd(rotMat)

        return DetectedMarkerSighting(
            markerID: markerID,
            cornersUV: cornersUV,
            positionWorld: centerWorld,
            quaternionXYZW: SIMD4<Double>(quat.imag.x, quat.imag.y, quat.imag.z, quat.real),
            sizeMMPrinted: printedSizeMM,
            sizeMMMeasured: measuredMM,
            rangeMeters: rangeM
        )
    }

    private static func pointInQuad(_ p: SIMD2<Double>, quad: [SIMD2<Double>]) -> Bool {
        var positive = 0
        var negative = 0
        for i in 0..<4 {
            let a = quad[i]
            let b = quad[(i + 1) % 4]
            let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            if cross >= 0 {
                positive += 1
            } else {
                negative += 1
            }
        }
        return positive == 4 || negative == 4
    }
}
