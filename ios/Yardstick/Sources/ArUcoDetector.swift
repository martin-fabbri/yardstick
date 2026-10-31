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
/// luma plane, decodes the 7×7 grid (1-cell black border + 5×5 inner code), fits a 3D plane via SVD/Jacobi
/// eigendecomposition to at least 16 interior LiDAR depth pixels (range 0.50–3.50 m), and projects the 4 RGB
/// corner rays onto the fitted plane.
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

    /// Return the cyclic shift applied to clockwise quad corners `[TL, TR, BR, BL]` when the observed
    /// 5x5 bit pattern matches the dictionary after `rotCount` 90-degree clockwise image rotations.
    ///
    /// When a marker in the image is rotated 90 degrees clockwise (`rotCount == 1`), its canonical
    /// Top-Left corner moves clockwise from `quad[0]` (image TL) to `quad[1]` (image TR), so the
    /// canonical corner sequence is `(0..<4).map { quad[($0 + rotCount) % 4] }`.
    public static func canonicalCornerShift(forRotation90CW rotCount: Int) -> Int {
        return ((rotCount % 4) + 4) % 4
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

    /// Order 4 2D points clockwise in top-left image coordinates `(u right, v down)`, starting at top-left (`min(u + v)`).
    public static func orderQuadClockwiseTL(_ pts: [SIMD2<Double>]) -> [SIMD2<Double>] {
        guard pts.count == 4 else { return pts }
        let center = (pts[0] + pts[1] + pts[2] + pts[3]) * 0.25
        // In top-left pixel coords (u right, v down), atan2(v - cy, u - cx) increases clockwise!
        let sortedCW = pts.sorted { a, b in
            atan2(a.y - center.y, a.x - center.x) < atan2(b.y - center.y, b.x - center.x)
        }
        var startIdx = 0
        var bestSum = Double.infinity
        for (i, p) in sortedCW.enumerated() {
            let s = p.x + p.y
            if s < bestSum {
                bestSum = s
                startIdx = i
            }
        }
        return (0..<4).map { sortedCW[(startIdx + $0) % 4] }
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
            req.maximumObservations = 24
            req.minimumAspectRatio = 0.30
            req.maximumAspectRatio = 1.0
            req.quadratureTolerance = 35.0
            req.minimumSize = 0.02
            req.minimumConfidence = 0.35
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([req])
            if let results = req.results {
                for obs in results {
                    // Vision normalized coordinates have origin at bottom-left; convert to top-left pixel (u, v)
                    let rawPts = [
                        SIMD2<Double>(Double(obs.topLeft.x) * Double(width), (1.0 - Double(obs.topLeft.y)) * Double(height)),
                        SIMD2<Double>(Double(obs.topRight.x) * Double(width), (1.0 - Double(obs.topRight.y)) * Double(height)),
                        SIMD2<Double>(Double(obs.bottomRight.x) * Double(width), (1.0 - Double(obs.bottomRight.y)) * Double(height)),
                        SIMD2<Double>(Double(obs.bottomLeft.x) * Double(width), (1.0 - Double(obs.bottomLeft.y)) * Double(height)),
                    ]
                    let ordered = orderQuadClockwiseTL(rawPts)
                    let center = (ordered[0] + ordered[1] + ordered[2] + ordered[3]) * 0.25
                    // Test direct quad as well as inset candidates in case Vision latched onto the outer white paper sheet
                    for scale in [1.0, 0.85, 0.80, 0.75] {
                        candidateQuads.append(ordered.map { center + scale * ($0 - center) })
                    }
                }
            }
        }

        // 2. Decode each candidate quad against the 7x7 ArUco grid (1-cell black border + 5x5 code)
        var detectedByID: [Int: [SIMD2<Double>]] = [:]
        for quad in candidateQuads {
            if let (mid, canonicalCorners) = decodeQuad7x7(quad: quad, lumaBytes: lumaBytes, width: width, height: height) {
                if detectedByID[mid] == nil {
                    detectedByID[mid] = canonicalCorners
                }
            }
        }

        return detectedByID.keys.sorted().compactMap { mid in
            guard let c = detectedByID[mid] else { return nil }
            return (markerID: mid, cornersUV: c)
        }
    }

    /// Sample a 7x7 ArUco grid inside `quad` (`[TL, TR, BR, BL]`), verify the 1-cell black border and outer quiet zone,
    /// and match the inner 5x5 bit pattern against `ArUcoDictionary5x5`.
    public static func decodeQuad7x7(
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

        // Rotate corners so index 0 is canonical Top-Left of the marker (shift = rotCount % 4)
        let shift = canonicalCornerShift(forRotation90CW: rotCount)
        let canonicalCorners = (0..<4).map { i in quad[(i + shift) % 4] }
        return (markerID, canonicalCorners)
    }

    private static func makeGrayCGImage(lumaBytes: [UInt8], width: Int, height: Int) -> CGImage? {
        let data = Data(lumaBytes) as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.linearGray)
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

    /// Fit a 3D plane normal to `points` centered at `centroid` using 3x3 symmetric Jacobi eigendecomposition.
    /// Returns the unit eigenvector corresponding to the smallest eigenvalue, oriented so `dot(n, -centroid) > 0`.
    public static func fitPlaneNormalPCA(points: [SIMD3<Double>], centroid: SIMD3<Double>) -> SIMD3<Double> {
        let fallback = simd_normalize(-centroid)
        guard points.count >= 3 else { return fallback }

        var c00 = 0.0, c01 = 0.0, c02 = 0.0
        var c11 = 0.0, c12 = 0.0, c22 = 0.0
        for p in points {
            let d = p - centroid
            c00 += d.x * d.x
            c01 += d.x * d.y
            c02 += d.x * d.z
            c11 += d.y * d.y
            c12 += d.y * d.z
            c22 += d.z * d.z
        }

        var a = [
            [c00, c01, c02],
            [c01, c11, c12],
            [c02, c12, c22],
        ]
        var v = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        // 15 Jacobi sweeps diagonalize a 3x3 symmetric matrix to machine precision
        for _ in 0..<15 {
            let pairs = [(0, 1), (0, 2), (1, 2)]
            var maxOff = 0.0
            for (p, q) in pairs {
                let apq = abs(a[p][q])
                if apq > maxOff { maxOff = apq }
                guard apq > 1e-15 else { continue }
                let diff = a[q][q] - a[p][p]
                let t: Double
                if abs(diff) < 1e-15 {
                    t = (a[p][q] >= 0.0) ? 1.0 : -1.0
                } else {
                    let phi = diff / (2.0 * a[p][q])
                    let sign = (phi >= 0.0) ? 1.0 : -1.0
                    t = sign / (abs(phi) + sqrt(1.0 + phi * phi))
                }
                let c = 1.0 / sqrt(1.0 + t * t)
                let s = t * c
                let app = a[p][p]
                let aqq = a[q][q]
                let apqVal = a[p][q]
                a[p][p] = c * c * app - 2.0 * s * c * apqVal + s * s * aqq
                a[q][q] = s * s * app + 2.0 * s * c * apqVal + c * c * aqq
                a[p][q] = 0.0
                a[q][p] = 0.0

                for r in 0..<3 where r != p && r != q {
                    let arp = a[r][p]
                    let arq = a[r][q]
                    a[r][p] = c * arp - s * arq
                    a[p][r] = a[r][p]
                    a[r][q] = s * arp + c * arq
                    a[q][r] = a[r][q]
                }
                for r in 0..<3 {
                    let vrp = v[r][p]
                    let vrq = v[r][q]
                    v[r][p] = c * vrp - s * vrq
                    v[r][q] = s * vrp + c * vrq
                }
            }
            if maxOff < 1e-15 { break }
        }

        let eigs = [a[0][0], a[1][1], a[2][2]]
        var minIdx = 0
        if eigs[1] < eigs[minIdx] { minIdx = 1 }
        if eigs[2] < eigs[minIdx] { minIdx = 2 }

        var normal = SIMD3<Double>(v[0][minIdx], v[1][minIdx], v[2][minIdx])
        let len = simd_length(normal)
        guard len > 1e-9 else { return fallback }
        normal /= len
        if simd_dot(normal, -centroid) < 0.0 {
            normal = -normal
        }
        return normal
    }

    /// Project 4 RGB marker corners onto the 3D plane fitted to interior LiDAR depth pixels (10% inward border erosion, >= 16 points, range 0.50–3.50 m, `YS-30`, `YS-33`, `Q027`).
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

        let cornersD = cornersUV.map { SIMD2<Double>($0.x * scaleX, $0.y * scaleY) }
        let centerD = (cornersD[0] + cornersD[1] + cornersD[2] + cornersD[3]) * 0.25

        func sampleDepthPoints(scaleFactor: Double) -> [SIMD3<Double>] {
            let scaledQuad = cornersD.map { centerD + scaleFactor * ($0 - centerD) }
            let minU = max(0, Int(scaledQuad.map(\.x).min() ?? 0))
            let maxU = min(depthWidth - 1, Int(ceil(scaledQuad.map(\.x).max() ?? 0)))
            let minV = max(0, Int(scaledQuad.map(\.y).min() ?? 0))
            let maxV = min(depthHeight - 1, Int(ceil(scaledQuad.map(\.y).max() ?? 0)))
            var pts: [SIMD3<Double>] = []
            guard minU <= maxU, minV <= maxV else { return pts }
            for v in minV...maxV {
                for u in minU...maxU {
                    let pt = SIMD2<Double>(Double(u) + 0.5, Double(v) + 0.5)
                    if pointInQuad(pt, quad: scaledQuad) {
                        let idx = v * depthWidth + u
                        let dMM = depthMapMM[idx]
                        let conf = confidenceMap[idx]
                        if dMM > 0 && conf >= 1 {
                            let z = Double(dMM) / 1000.0
                            let x = (Double(u) - cxD) * z / fxD
                            let y = -(Double(v) - cyD) * z / fyD
                            pts.append(SIMD3<Double>(x, y, -z))
                        }
                    }
                }
            }
            return pts
        }

        // Start with 10% inward border erosion; if marker is far (up to 3.5 m) and has < 16 pixels,
        // expand to 1.0x or 1.35x surrounding plane patch so the plane fit always uses >= 16 LiDAR points.
        var interiorCamPts = sampleDepthPoints(scaleFactor: 0.90)
        if interiorCamPts.count < 16 {
            interiorCamPts = sampleDepthPoints(scaleFactor: 1.00)
        }
        if interiorCamPts.count < 16 {
            interiorCamPts = sampleDepthPoints(scaleFactor: 1.35)
        }
        guard interiorCamPts.count >= 16 else {
            return nil
        }

        var centroidCam = SIMD3<Double>.zero
        for p in interiorCamPts {
            centroidCam += p
        }
        centroidCam /= Double(interiorCamPts.count)
        let rangeM = simd_length(centroidCam)
        // Range gate: 0.50 m to 3.50 m so ground markers viewed from chest height (~2.6 m) are accepted
        guard rangeM >= 0.50 && rangeM <= 3.50 else {
            return nil
        }

        // Fit 3D plane to interior LiDAR points via PCA/Jacobi eigendecomposition (Blocker 2)
        var normalCam = fitPlaneNormalPCA(points: interiorCamPts, centroid: centroidCam)
        let viewDir = simd_normalize(-centroidCam)
        // Guard against grazing-angle degenerate plane fits (> 75 deg from view ray)
        if abs(simd_dot(normalCam, viewDir)) < cos(75.0 * .pi / 180.0) {
            normalCam = viewDir
        }
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

        // Transform corners, center, and fitted plane normal into ARKit world coordinates
        let cornersWorld = cornersCam.map { p -> SIMD3<Double> in
            let hp = cameraToWorld * SIMD4<Double>(p.x, p.y, p.z, 1.0)
            return SIMD3<Double>(hp.x, hp.y, hp.z)
        }
        let centerWorld = (cornersWorld[0] + cornersWorld[1] + cornersWorld[2] + cornersWorld[3]) * 0.25
        let normalWorld4 = cameraToWorld * SIMD4<Double>(normalCam.x, normalCam.y, normalCam.z, 0.0)
        let zAxis = simd_normalize(SIMD3<Double>(normalWorld4.x, normalWorld4.y, normalWorld4.z))

        // Build orthonormal marker rotation frame in world coordinates with +Z_m along the fitted plane normal:
        // +X_m along top edge (TL -> TR) projected into plane, +Y_m = +Z_m x +X_m
        let topEdge = cornersWorld[1] - cornersWorld[0]
        let xProj = topEdge - simd_dot(topEdge, zAxis) * zAxis
        guard simd_length(xProj) > 1e-9 else { return nil }
        let xAxis = simd_normalize(xProj)
        let yAxis = simd_normalize(simd_cross(zAxis, xAxis))
        let rotMat = simd_double3x3(columns: (xAxis, yAxis, zAxis))
        let quat = simd_normalize(simd_quatd(rotMat))

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
