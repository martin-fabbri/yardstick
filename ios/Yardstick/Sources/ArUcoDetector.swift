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
//
// Portions of the ArUco DICT_5X5_250 adaptive-threshold contour detector and
// sub-pixel corner refiner follow OpenCV's `objdetect` / `imgproc` algorithms
// (Copyright (C) 2000-2024, Intel Corporation, Willow Garage Inc., Itseez Inc.,
// OpenCV Foundation; Apache License, Version 2.0).

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import simd

/// ArUco `DICT_5X5_250` marker detector and LiDAR depth plane corner projector (`YS-30`, `YS-33`, `YS-69`, `YS-70`).
///
/// The Vision rectangle detector and its `[1.0, 0.85, 0.80, 0.75]` inset ladder are removed (`YS-69`).
/// Both the phone and the Python toolkit run the same OpenCV `DICT_5X5_250` pipeline:
/// - box-mean adaptive thresholding across window sizes `[3, 13, 23]` with `C = 7.0`,
/// - Suzuki85 outer boundary contour following and closed Ramer-Douglas-Peucker polygon approximation (`0.03 * perimeter`),
/// - perspective homography unwarping (`28×28`, `4 px/cell`) with Otsu thresholding (`minOtsuStdDev = 5.0`),
/// - sub-pixel corner refinement (`cornerSubPix`, window `5`, `maxIter = 30`, `eps = 0.1`),
/// - rejection of any quad whose refined corners move `> 3%` of mean 2D edge length (`YS-69`),
/// - 3D LiDAR plane projection with rejection of any quad whose 3D edge lengths disagree by `> 4%` (`YS-69`),
/// - confirmation on the encoded JPEG bytes before writing any row to `markers.csv` or any marker-triggered still to `stills.csv` (`YS-70`).
public struct DetectedMarkerSighting: Sendable {
    public let markerID: Int
    public let cornersUV: [SIMD2<Double>]      // 4 corners in RGB pixel coordinates (TL, TR, BR, BL)
    public let cornersWorld: [SIMD3<Double>]   // 4 corners in ARKit world coordinates (TL, TR, BR, BL)
    public let positionWorld: SIMD3<Double>
    public let quaternionXYZW: SIMD4<Double>
    public let sizeMMPrinted: Double
    public let sizeMMMeasured: Double
    public let rangeMeters: Double
    public let viewingAngleDeg: Double

    public init(
        markerID: Int,
        cornersUV: [SIMD2<Double>],
        cornersWorld: [SIMD3<Double>] = [],
        positionWorld: SIMD3<Double>,
        quaternionXYZW: SIMD4<Double>,
        sizeMMPrinted: Double,
        sizeMMMeasured: Double,
        rangeMeters: Double,
        viewingAngleDeg: Double = 0.0
    ) {
        self.markerID = markerID
        self.cornersUV = cornersUV
        self.cornersWorld = cornersWorld
        self.positionWorld = positionWorld
        self.quaternionXYZW = quaternionXYZW
        self.sizeMMPrinted = sizeMMPrinted
        self.sizeMMMeasured = sizeMMMeasured
        self.rangeMeters = rangeMeters
        self.viewingAngleDeg = viewingAngleDeg
    }
}

public enum ArUcoDetector {
    /// OpenCV Apache-2.0 attribution text displayed in Settings ▸ About (`YS-70`).
    public static let openCVApacheAttributionNotice = """
    ArUco Marker Detection & Sub-Pixel Corner Refinement (`DICT_5X5_250`)
    Portions adapted from OpenCV (`core`, `imgproc`, `calib3d`, `objdetect`).
    Copyright © 2000–2024 Intel Corporation, Willow Garage Inc., Itseez Inc., and OpenCV Foundation.
    Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
    """

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

    /// Verify that sub-pixel corner refinement (`cornerSubPix` window 5) did not shift any corner by more
    /// than `maxCornerShiftRatio` (3%) of the raw quad's mean 2D edge length (`YS-69`).
    public static func checkRefinedCornerShift(
        rawCorners: [SIMD2<Double>],
        refinedCorners: [SIMD2<Double>],
        maxCornerShiftRatio: Double = 0.03
    ) -> Bool {
        guard rawCorners.count == 4, refinedCorners.count == 4 else { return false }
        var edgeSum = 0.0
        for i in 0..<4 {
            edgeSum += simd_length(rawCorners[(i + 1) % 4] - rawCorners[i])
        }
        let meanEdge = edgeSum * 0.25
        guard meanEdge > 1e-6 else { return false }
        var maxShift = 0.0
        for i in 0..<4 {
            let shift = simd_length(refinedCorners[i] - rawCorners[i])
            if shift > maxShift {
                maxShift = shift
            }
        }
        return (maxShift / meanEdge) <= maxCornerShiftRatio
    }

    /// Refine 2D corner positions to sub-pixel accuracy on the outer black-to-white marker border using
    /// OpenCV's `cornerSubPix` orthogonal gradient-dot-product iteration (`winSize = 5`, `maxIter = 30`, `eps = 0.1`, `YS-69`).
    public static func refineCornersSubPix(
        corners: [SIMD2<Double>],
        lumaBytes: [UInt8],
        width: Int,
        height: Int,
        winSize: Int = 5,
        maxIter: Int = 30,
        eps: Double = 0.1
    ) -> [SIMD2<Double>] {
        let maskSize = 2 * winSize + 1
        let patchSize = maskSize + 2
        let halfPatch = winSize + 1

        // Precompute 1D Gaussian weights matching OpenCV cornerSubPix: exp(-((i - winSize) / winSize)^2)
        var mask1D = [Double](repeating: 0.0, count: maskSize)
        let invWin = 1.0 / Double(winSize)
        for i in 0..<maskSize {
            let d = Double(i - winSize) * invWin
            mask1D[i] = exp(-d * d)
        }
        var weightMask = [Double](repeating: 0.0, count: maskSize * maskSize)
        for y in 0..<maskSize {
            for x in 0..<maskSize {
                weightMask[y * maskSize + x] = mask1D[y] * mask1D[x]
            }
        }

        func sampleSubPix(x: Double, y: Double) -> Double {
            let cx = min(Double(width - 1), max(0.0, x))
            let cy = min(Double(height - 1), max(0.0, y))
            let x0 = Int(floor(cx))
            let y0 = Int(floor(cy))
            let x1 = min(width - 1, x0 + 1)
            let y1 = min(height - 1, y0 + 1)
            let ax = cx - Double(x0)
            let ay = cy - Double(y0)
            let v00 = Double(lumaBytes[y0 * width + x0])
            let v01 = Double(lumaBytes[y0 * width + x1])
            let v10 = Double(lumaBytes[y1 * width + x0])
            let v11 = Double(lumaBytes[y1 * width + x1])
            return (1.0 - ay) * ((1.0 - ax) * v00 + ax * v01) + ay * ((1.0 - ax) * v10 + ax * v11)
        }

        var refined: [SIMD2<Double>] = []
        refined.reserveCapacity(corners.count)
        var patch = [Double](repeating: 0.0, count: patchSize * patchSize)

        for pt in corners {
            let cT = pt
            var cI = pt
            guard cT.x >= Double(halfPatch) && cT.x < Double(width - halfPatch) &&
                  cT.y >= Double(halfPatch) && cT.y < Double(height - halfPatch)
            else {
                refined.append(pt)
                continue
            }

            for _ in 0..<maxIter {
                for py in 0..<patchSize {
                    let sy = cI.y + Double(py - halfPatch)
                    let rowBase = py * patchSize
                    for px in 0..<patchSize {
                        let sx = cI.x + Double(px - halfPatch)
                        patch[rowBase + px] = sampleSubPix(x: sx, y: sy)
                    }
                }

                var a = 0.0, b = 0.0, c = 0.0
                var bb1 = 0.0, bb2 = 0.0
                for py in 0..<maskSize {
                    let pRow = (py + 1) * patchSize
                    let pRowUp = py * patchSize
                    let pRowDn = (py + 2) * patchSize
                    let wRow = py * maskSize
                    let dy = Double(py - winSize)
                    let pyCoord = cI.y + dy
                    for px in 0..<maskSize {
                        let gx = patch[pRow + px + 2] - patch[pRow + px]
                        let gy = patch[pRowDn + px + 1] - patch[pRowUp + px + 1]
                        let w = weightMask[wRow + px]
                        let gxx = gx * gx * w
                        let gxy = gx * gy * w
                        let gyy = gy * gy * w
                        a += gxx
                        b += gxy
                        c += gyy
                        let pxCoord = cI.x + Double(px - winSize)
                        bb1 += gxx * pxCoord + gxy * pyCoord
                        bb2 += gxy * pxCoord + gyy * pyCoord
                    }
                }

                let det = a * c - b * b
                guard abs(det) > 1e-10 else { break }
                let invDet = 1.0 / det
                let nx = (c * bb1 - b * bb2) * invDet
                let ny = (a * bb2 - b * bb1) * invDet
                let cI2 = SIMD2<Double>(nx, ny)
                let shiftSq = simd_length_squared(cI2 - cI)
                cI = cI2
                if abs(cI.x - cT.x) > Double(winSize) || abs(cI.y - cT.y) > Double(winSize) {
                    cI = cT
                    break
                }
                if shiftSq <= eps * eps {
                    break
                }
            }
            refined.append(cI)
        }
        return refined
    }

    /// Detect 2D ArUco `DICT_5X5_250` markers in an 8-bit luma plane (`width` x `height`) using OpenCV's
    /// adaptive-threshold contour detector (`winSize in [3, 13, 23]`, `C = 7.0`), `approxPolyDP` (`0.03 * perim`),
    /// perspective homography unwarping (`28×28`, Otsu threshold), and sub-pixel corner refinement (`winSize = 5`),
    /// rejecting any quad whose refined corners shift by `> maxCornerShiftRatio` (3%, `YS-69`).
    public static func detectMarkers2D(
        lumaBytes: [UInt8],
        width: Int,
        height: Int,
        maxCornerShiftRatio: Double = 0.03
    ) -> [(markerID: Int, cornersUV: [SIMD2<Double>])] {
        guard lumaBytes.count == width * height, width >= 64, height >= 64 else {
            return []
        }

        // 1. Build 1-based integral image for O(1) adaptive box-mean thresholding
        let strideI = width + 1
        var integral = [Int](repeating: 0, count: (width + 1) * (height + 1))
        for v in 0..<height {
            var rowSum = 0
            let srcBase = v * width
            let prevBase = v * strideI
            let dstBase = (v + 1) * strideI
            for u in 0..<width {
                rowSum &+= Int(lumaBytes[srcBase + u])
                integral[dstBase + u + 1] = integral[prevBase + u + 1] &+ rowSum
            }
        }

        let maxDim = Double(max(width, height))
        let minPerim = max(16, Int(maxDim * 0.03))
        let maxPerim = Int(maxDim * 4.0)
        let dedupDist = max(10.0, Double(minPerim) * 0.05)

        struct CandidateQuad {
            var corners: [SIMD2<Double>]
            var perimeter: Double
        }
        var allCandidates: [CandidateQuad] = []

        var bin = [UInt8](repeating: 0, count: width * height)
        var visitedEdge = [UInt8](repeating: 0, count: width * height)

        // Counter-clockwise 8-neighborhood offsets matching Suzuki85 / OpenCV `findContours`
        let ccwDx = [-1, -1, 0, 1, 1, 1, 0, -1]
        let ccwDy = [0, 1, 1, 1, 0, -1, -1, -1]

        // 2. Multi-scale adaptive threshold across OpenCV's default ArUco window sizes [3, 13, 23]
        for win in [3, 13, 23] {
            let rad = win / 2
            for v in 0..<height {
                let y0 = max(0, v - rad)
                let y1 = min(height - 1, v + rad) + 1
                let rowY0 = y0 * strideI
                let rowY1 = y1 * strideI
                let hBox = y1 - y0
                let dstRow = v * width
                for u in 0..<width {
                    let x0 = max(0, u - rad)
                    let x1 = min(width - 1, u + rad) + 1
                    let area = hBox * (x1 - x0)
                    let sum = integral[rowY1 + x1] - integral[rowY0 + x1] - integral[rowY1 + x0] + integral[rowY0 + x0]
                    // THRESH_BINARY_INV with C = 7.0: foreground (1) when luma <= round(mean) - 7
                    let meanRounded = (sum + (area >> 1)) / area
                    bin[dstRow + u] = (Int(lumaBytes[dstRow + u]) <= meanRounded - 7) ? 1 : 0
                }
            }

            visitedEdge.withUnsafeMutableBufferPointer { buf in
                buf.initialize(repeating: 0)
            }

            var windowQuads: [CandidateQuad] = []
            for v in 1..<(height - 1) {
                let rowBase = v * width
                for u in 1..<(width - 1) {
                    let idx = rowBase + u
                    guard bin[idx] == 1 && bin[idx - 1] == 0 && visitedEdge[idx] == 0 else {
                        continue
                    }

                    // Trace outer 8-connected boundary starting from left background neighbor (dir = 0)
                    var contour: [(x: Int, y: Int)] = []
                    contour.reserveCapacity(256)
                    var cx = u
                    var cy = v
                    var dir = 0
                    var firstStepDir = -1
                    let maxSteps = min(width * height / 2, maxPerim * 2)

                    for step in 0..<maxSteps {
                        contour.append((cx, cy))
                        if cx > 0 && bin[cy * width + cx - 1] == 0 {
                            visitedEdge[cy * width + cx] = 1
                        }
                        var foundNext = false
                        var nextDir = 0
                        var nx = cx
                        var ny = cy
                        for k in 0..<8 {
                            let d = (dir + k) & 7
                            let tx = cx + ccwDx[d]
                            let ty = cy + ccwDy[d]
                            if tx >= 0 && tx < width && ty >= 0 && ty < height && bin[ty * width + tx] == 1 {
                                nx = tx
                                ny = ty
                                nextDir = d
                                foundNext = true
                                break
                            }
                        }
                        if !foundNext {
                            break
                        }
                        if step == 0 {
                            firstStepDir = nextDir
                        } else if cx == u && cy == v && nextDir == firstStepDir {
                            contour.removeLast()
                            break
                        }
                        cx = nx
                        cy = ny
                        dir = ((nextDir + 4) & 7 + 1) & 7
                    }

                    guard contour.count >= minPerim && contour.count <= maxPerim else {
                        continue
                    }

                    let eps = Double(contour.count) * 0.03
                    let poly = approxPolyDPClosed(contour: contour, epsilon: eps)
                    guard poly.count == 4 && isConvexQuad(poly) else {
                        continue
                    }

                    var borderOK = true
                    for p in poly {
                        if p.x < 3.0 || p.x >= Double(width - 3) || p.y < 3.0 || p.y >= Double(height - 3) {
                            borderOK = false
                            break
                        }
                    }
                    guard borderOK else { continue }

                    let ordered = orderQuadClockwiseTL(poly)
                    var perim = 0.0
                    var minEdge = Double.infinity
                    for i in 0..<4 {
                        let e = simd_length(ordered[(i + 1) % 4] - ordered[i])
                        perim += e
                        if e < minEdge { minEdge = e }
                    }
                    guard minEdge >= max(4.0, perim * 0.05) else {
                        continue
                    }

                    // Deduplicate nearby contours (outer black border vs inner hole): keep larger perimeter
                    var matchedExisting = false
                    for k in 0..<windowQuads.count {
                        let d = meanCornerDistance(ordered, windowQuads[k].corners)
                        if d < dedupDist {
                            if perim > windowQuads[k].perimeter {
                                windowQuads[k] = CandidateQuad(corners: ordered, perimeter: perim)
                            }
                            matchedExisting = true
                            break
                        }
                    }
                    if !matchedExisting {
                        windowQuads.append(CandidateQuad(corners: ordered, perimeter: perim))
                    }
                }
            }

            for q in windowQuads {
                var matchedGlobal = false
                for k in 0..<allCandidates.count {
                    if meanCornerDistance(q.corners, allCandidates[k].corners) < dedupDist {
                        if q.perimeter > allCandidates[k].perimeter {
                            allCandidates[k] = q
                        }
                        matchedGlobal = true
                        break
                    }
                }
                if !matchedGlobal {
                    allCandidates.append(q)
                }
            }
        }

        // Sort candidates by largest perimeter first so outer marker borders are decoded first
        allCandidates.sort { $0.perimeter > $1.perimeter }

        var detectedByID: [Int: (corners: [SIMD2<Double>], perim: Double)] = [:]
        for cand in allCandidates {
            guard let (mid, canonicalRaw) = decodeQuadOpenCV(
                quad: cand.corners,
                lumaBytes: lumaBytes,
                width: width,
                height: height
            ) else {
                continue
            }

            let refined = refineCornersSubPix(
                corners: canonicalRaw,
                lumaBytes: lumaBytes,
                width: width,
                height: height,
                winSize: 5,
                maxIter: 30,
                eps: 0.1
            )
            guard checkRefinedCornerShift(
                rawCorners: canonicalRaw,
                refinedCorners: refined,
                maxCornerShiftRatio: maxCornerShiftRatio
            ) else {
                continue
            }

            if let existing = detectedByID[mid] {
                if cand.perimeter > existing.perim {
                    detectedByID[mid] = (refined, cand.perimeter)
                }
            } else {
                detectedByID[mid] = (refined, cand.perimeter)
            }
        }

        return detectedByID.keys.sorted().compactMap { mid in
            guard let entry = detectedByID[mid] else { return nil }
            return (markerID: mid, cornersUV: entry.corners)
        }
    }

    /// Closed Ramer-Douglas-Peucker polygon approximation matching OpenCV's `cv::approxPolyDP(..., closed=true)`.
    private static func approxPolyDPClosed(contour: [(x: Int, y: Int)], epsilon: Double) -> [SIMD2<Double>] {
        let n = contour.count
        guard n >= 4 else { return [] }

        // 3-pass diametral furthest-point split matching OpenCV's closed `approxPolyDP`
        var idxA = 0
        var idxB = 0
        var maxDistSq = 0
        for _ in 0..<3 {
            maxDistSq = 0
            let ax = contour[idxA].x
            let ay = contour[idxA].y
            for i in 0..<n {
                let dx = contour[i].x - ax
                let dy = contour[i].y - ay
                let dSq = dx * dx + dy * dy
                if dSq > maxDistSq {
                    maxDistSq = dSq
                    idxB = i
                }
            }
            swap(&idxA, &idxB)
        }

        if Double(maxDistSq) <= epsilon * epsilon {
            return [SIMD2<Double>(Double(contour[idxA].x), Double(contour[idxA].y))]
        }

        func sliceContour(from start: Int, to end: Int) -> [SIMD2<Double>] {
            var out: [SIMD2<Double>] = []
            var k = start
            while true {
                out.append(SIMD2<Double>(Double(contour[k].x), Double(contour[k].y)))
                if k == end { break }
                k = (k + 1) % n
            }
            return out
        }

        func rdpOpen(_ pts: [SIMD2<Double>], epsSq: Double) -> [SIMD2<Double>] {
            guard pts.count > 2 else { return pts }
            let a = pts[0]
            let b = pts[pts.count - 1]
            let ab = b - a
            let abLenSq = simd_length_squared(ab)
            var bestIdx = -1
            var bestDistSq = -1.0

            for i in 1..<(pts.count - 1) {
                let p = pts[i]
                let dSq: Double
                if abLenSq < 1e-12 {
                    dSq = simd_length_squared(p - a)
                } else {
                    let cross = ab.x * (p.y - a.y) - ab.y * (p.x - a.x)
                    dSq = (cross * cross) / abLenSq
                }
                if dSq > bestDistSq {
                    bestDistSq = dSq
                    bestIdx = i
                }
            }

            if bestDistSq > epsSq && bestIdx > 0 {
                let left = rdpOpen(Array(pts[0...bestIdx]), epsSq: epsSq)
                let right = rdpOpen(Array(pts[bestIdx..<pts.count]), epsSq: epsSq)
                return Array(left.dropLast()) + right
            } else {
                return [a, b]
            }
        }

        let epsSq = epsilon * epsilon
        let arc1 = rdpOpen(sliceContour(from: idxA, to: idxB), epsSq: epsSq)
        let arc2 = rdpOpen(sliceContour(from: idxB, to: idxA), epsSq: epsSq)
        return Array(arc1.dropLast()) + Array(arc2.dropLast())
    }

    private static func isConvexQuad(_ pts: [SIMD2<Double>]) -> Bool {
        guard pts.count == 4 else { return false }
        var pos = 0
        var neg = 0
        for i in 0..<4 {
            let a = pts[i]
            let b = pts[(i + 1) % 4]
            let c = pts[(i + 2) % 4]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if cross > 1e-3 {
                pos += 1
            } else if cross < -1e-3 {
                neg += 1
            }
        }
        return pos == 4 || neg == 4
    }

    private static func meanCornerDistance(_ a: [SIMD2<Double>], _ b: [SIMD2<Double>]) -> Double {
        guard a.count == 4, b.count == 4 else { return .infinity }
        var best = Double.infinity
        for shift in 0..<4 {
            var sum = 0.0
            for i in 0..<4 {
                sum += simd_length(a[i] - b[(i + shift) % 4])
            }
            let mean = sum * 0.25
            if mean < best { best = mean }
        }
        return best
    }

    /// Decode a candidate quad `[TL, TR, BR, BL]` by perspective homography unwarping (`28×28`, `4 px/cell`),
    /// Otsu thresholding (`minOtsuStdDev = 5.0`), 24-cell black border verification, and `DICT_5X5_250` bit matching (`YS-69`).
    public static func decodeQuadOpenCV(
        quad: [SIMD2<Double>],
        lumaBytes: [UInt8],
        width: Int,
        height: Int
    ) -> (Int, [SIMD2<Double>])? {
        guard quad.count == 4,
              let H = computeSquareToQuadHomography(squareSize: 28.0, quad: quad)
        else {
            return nil
        }

        let warpSize = 28
        var patch = [UInt8](repeating: 0, count: warpSize * warpSize)
        var histogram = [Int](repeating: 0, count: 256)
        var totalSum = 0.0
        var totalSumSq = 0.0

        for y in 0..<warpSize {
            let sy = Double(y) + 0.5
            let rowBase = y * warpSize
            for x in 0..<warpSize {
                let sx = Double(x) + 0.5
                let denom = H[6] * sx + H[7] * sy + 1.0
                guard abs(denom) > 1e-9 else { return nil }
                let u = (H[0] * sx + H[1] * sy + H[2]) / denom
                let v = (H[3] * sx + H[4] * sy + H[5]) / denom
                guard u >= 0.0 && u < Double(width - 1) && v >= 0.0 && v < Double(height - 1) else {
                    return nil
                }
                let u0 = Int(floor(u))
                let v0 = Int(floor(v))
                let ax = u - Double(u0)
                let ay = v - Double(v0)
                let i00 = v0 * width + u0
                let val = (1.0 - ay) * ((1.0 - ax) * Double(lumaBytes[i00]) + ax * Double(lumaBytes[i00 + 1]))
                        + ay * ((1.0 - ax) * Double(lumaBytes[i00 + width]) + ax * Double(lumaBytes[i00 + width + 1]))
                let byteVal = UInt8(clamping: Int(val.rounded()))
                patch[rowBase + x] = byteVal
                histogram[Int(byteVal)] += 1
                let dVal = Double(byteVal)
                totalSum += dVal
                totalSumSq += dVal * dVal
            }
        }

        let nPix = Double(warpSize * warpSize)
        let mean = totalSum / nPix
        let variance = max(0.0, (totalSumSq / nPix) - (mean * mean))
        guard sqrt(variance) >= 5.0 else {
            return nil
        }

        // Otsu threshold across the 28×28 warped patch
        var sumB = 0.0
        var wB = 0
        var maxBetweenVar = -1.0
        var otsuThresh = 127
        let nTotal = warpSize * warpSize
        for t in 0..<256 {
            wB += histogram[t]
            if wB == 0 { continue }
            let wF = nTotal - wB
            if wF == 0 { break }
            sumB += Double(t * histogram[t])
            let mB = sumB / Double(wB)
            let mF = (totalSum - sumB) / Double(wF)
            let between = Double(wB) * Double(wF) * (mB - mF) * (mB - mF)
            if between > maxBetweenVar {
                maxBetweenVar = between
                otsuThresh = t
            }
        }

        // Sample the inner 2×2 pixels of each 4×4 cell (margin = 1 px on each side of the 4×4 cell)
        for r in 0..<7 {
            for c in 0..<7 {
                if r == 0 || r == 6 || c == 0 || c == 6 {
                    var whiteCount = 0
                    for dy in 1...2 {
                        for dx in 1...2 {
                            if Int(patch[(r * 4 + dy) * warpSize + (c * 4 + dx)]) > otsuThresh {
                                whiteCount += 1
                            }
                        }
                    }
                    // Every border cell's inner 2×2 core must be strictly black (0 white pixels out of 4)
                    if whiteCount > 0 {
                        return nil
                    }
                }
            }
        }

        var observedBits: UInt32 = 0
        for r in 1...5 {
            for c in 1...5 {
                var whiteCount = 0
                for dy in 1...2 {
                    for dx in 1...2 {
                        if Int(patch[(r * 4 + dy) * warpSize + (c * 4 + dx)]) > otsuThresh {
                            whiteCount += 1
                        }
                    }
                }
                let bit: UInt32 = (whiteCount >= 2) ? 1 : 0
                let bitIndex = 24 - ((r - 1) * 5 + (c - 1))
                observedBits |= (bit << bitIndex)
            }
        }

        guard let (markerID, rotCount) = matchCode5x5(observedBits, maxHamming: 0) else {
            return nil
        }

        let shift = canonicalCornerShift(forRotation90CW: rotCount)
        let canonicalCorners = (0..<4).map { i in quad[(i + shift) % 4] }
        return (markerID, canonicalCorners)
    }

    /// Backwards-compatible wrapper that calls `decodeQuadOpenCV`.
    public static func decodeQuad7x7(
        quad: [SIMD2<Double>],
        lumaBytes: [UInt8],
        width: Int,
        height: Int
    ) -> (Int, [SIMD2<Double>])? {
        return decodeQuadOpenCV(quad: quad, lumaBytes: lumaBytes, width: width, height: height)
    }

    /// Compute the 3x3 projective homography `[h0, h1, h2, h3, h4, h5, h6, h7]` mapping
    /// `(0,0) -> quad[0]`, `(S,0) -> quad[1]`, `(S,S) -> quad[2]`, `(0,S) -> quad[3]`.
    private static func computeSquareToQuadHomography(squareSize S: Double, quad: [SIMD2<Double>]) -> [Double]? {
        guard quad.count == 4, S > 0 else { return nil }
        let x0 = quad[0].x, y0 = quad[0].y
        let x1 = quad[1].x, y1 = quad[1].y
        let x2 = quad[2].x, y2 = quad[2].y
        let x3 = quad[3].x, y3 = quad[3].y

        let dx1 = x1 - x2
        let dx2 = x3 - x2
        let sx = x0 - x1 + x2 - x3
        let dy1 = y1 - y2
        let dy2 = y3 - y2
        let sy = y0 - y1 + y2 - y3

        let det = dx1 * dy2 - dx2 * dy1
        guard abs(det) > 1e-12 else { return nil }
        let g = (sx * dy2 - sy * dx2) / det
        let h = (dx1 * sy - dy1 * sx) / det

        let h2 = x0
        let h5 = y0
        let h0 = (x1 - x0 + g * x1) / S
        let h1 = (x3 - x0 + h * x3) / S
        let h3 = (y1 - y0 + g * y1) / S
        let h4 = (y3 - y0 + h * y3) / S
        let h6 = g / S
        let h7 = h / S
        return [h0, h1, h2, h3, h4, h5, h6, h7]
    }

    /// Decode an encoded JPEG buffer (`Data`) into an 8-bit ITU-R BT.601 full-range luma array (`YS-70`).
    public static func decodeLumaFromJPEGData(_ jpegData: Data) -> (luma: [UInt8], width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithData(jpegData as CFData, nil),
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

    /// Re-verify marker sightings directly on the encoded JPEG bytes (`stills/%06d.jpg`) before any row is
    /// written to `markers.csv` or any marker-triggered still is written to `stills.csv` (`YS-70`).
    public static func confirmSightingsInJPEGData(
        _ jpegData: Data,
        printedSizesByMarker: [Int: Double],
        defaultPrintedSizeMM: Double = 250.0,
        depthMapMM: [UInt16],
        confidenceMap: [UInt8],
        depthWidth: Int,
        depthHeight: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        cameraToWorld: simd_double4x4,
        maxCornerShiftRatio: Double = 0.03,
        maxEdgeDisagreementRatio: Double = 0.04
    ) -> [DetectedMarkerSighting] {
        guard let (luma, w, h) = decodeLumaFromJPEGData(jpegData) else {
            return []
        }
        let dets2D = detectMarkers2D(
            lumaBytes: luma,
            width: w,
            height: h,
            maxCornerShiftRatio: maxCornerShiftRatio
        )
        var confirmed: [DetectedMarkerSighting] = []
        for d2 in dets2D {
            let pSize = printedSizesByMarker[d2.markerID] ?? defaultPrintedSizeMM
            if let s3D = measureMarker3DOnLiDARPlane(
                markerID: d2.markerID,
                cornersUV: d2.cornersUV,
                printedSizeMM: pSize,
                depthMapMM: depthMapMM,
                confidenceMap: confidenceMap,
                depthWidth: depthWidth,
                depthHeight: depthHeight,
                rgbWidth: w,
                rgbHeight: h,
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                cameraToWorld: cameraToWorld,
                maxEdgeDisagreementRatio: maxEdgeDisagreementRatio
            ) {
                confirmed.append(s3D)
            }
        }
        return confirmed
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
    /// Returns the unit eigenvector corresponding to the smallest eigenvalue, oriented so `normal.z > 0` (towards camera in ARKit `-Z` forward coords).
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
        // Orient normal towards camera (+Z in ARKit camera frame, matching toolkit `markers.py:157`)
        if normal.z < 0.0 {
            normal = -normal
        }
        return normal
    }

    /// Project 4 RGB marker corners onto the 3D plane fitted to interior LiDAR depth pixels (10% inward border erosion,
    /// `>= 16` points, range `0.50–3.50 m`, `<= 4%` 3D edge disagreement, `YS-30`, `YS-33`, `YS-69`).
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
        cameraToWorld: simd_double4x4,
        maxEdgeDisagreementRatio: Double = 0.04,
        maxPlaneTiltFromViewDeg: Double = 60.0
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
        guard rangeM >= 0.50 && rangeM <= 3.50 else {
            return nil
        }

        var normalCam = fitPlaneNormalPCA(points: interiorCamPts, centroid: centroidCam)
        let viewDir = simd_normalize(-centroidCam)
        let rawCosViewAngle = min(1.0, max(-1.0, abs(simd_dot(normalCam, viewDir))))
        let viewingAngleDeg = acos(rawCosViewAngle) * 180.0 / .pi
        // Guard against grazing-angle degenerate plane fits (matching toolkit `markers.py:160`: cos(60 deg))
        // Note: `viewingAngleDeg` is computed BEFORE this fallback so grazing-angle sightings (> 60 deg)
        // never masquerade as 0.0 deg fronto-parallel sightings in `GuidedCaptureEngine` (`YS-69`, `YS-74`).
        if rawCosViewAngle < cos(maxPlaneTiltFromViewDeg * .pi / 180.0) {
            normalCam = viewDir
        }
        let dCam = -simd_dot(normalCam, centroidCam)

        // Intersect the 4 full-res RGB corner rays with the fitted LiDAR depth plane
        var cornersCam: [SIMD3<Double>] = []
        cornersCam.reserveCapacity(4)
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

        var edgesM = [Double](repeating: 0.0, count: 4)
        var edgeSumM = 0.0
        for i in 0..<4 {
            let e = simd_length(cornersCam[(i + 1) % 4] - cornersCam[i])
            edgesM[i] = e
            edgeSumM += e
        }
        let meanEdgeM = edgeSumM * 0.25
        guard meanEdgeM > 1e-9 else { return nil }

        // Reject any quad whose 3D edge lengths disagree by > maxEdgeDisagreementRatio (4%, YS-69)
        if maxEdgeDisagreementRatio > 0.0 {
            let maxEdge = edgesM.max() ?? meanEdgeM
            let minEdge = edgesM.min() ?? meanEdgeM
            if ((maxEdge - minEdge) / meanEdgeM) > maxEdgeDisagreementRatio {
                return nil
            }
        }
        let measuredMM = meanEdgeM * 1000.0

        // Transform corners, center, and fitted plane normal into ARKit world coordinates
        let cornersWorld = cornersCam.map { p -> SIMD3<Double> in
            let hp = cameraToWorld * SIMD4<Double>(p.x, p.y, p.z, 1.0)
            return SIMD3<Double>(hp.x, hp.y, hp.z)
        }
        let centerWorld = (cornersWorld[0] + cornersWorld[1] + cornersWorld[2] + cornersWorld[3]) * 0.25
        let normalWorld4 = cameraToWorld * SIMD4<Double>(normalCam.x, normalCam.y, normalCam.z, 0.0)
        let zAxis = simd_normalize(SIMD3<Double>(normalWorld4.x, normalWorld4.y, normalWorld4.z))

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
            cornersWorld: cornersWorld,
            positionWorld: centerWorld,
            quaternionXYZW: SIMD4<Double>(quat.imag.x, quat.imag.y, quat.imag.z, quat.real),
            sizeMMPrinted: printedSizeMM,
            sizeMMMeasured: measuredMM,
            rangeMeters: rangeM,
            viewingAngleDeg: viewingAngleDeg
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
