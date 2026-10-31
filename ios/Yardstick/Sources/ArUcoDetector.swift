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
import simd

/// Clean-room ArUco `DICT_5X5_250` marker detector and LiDAR depth plane corner projector (`YS-30`, `YS-33`, `Q007`, `Q027`).
///
/// Uses the bit patterns in `ArUcoDictionaryData` (generated from OpenCV's published `DICT_5X5_250`
/// definition via `toolkit/scripts/generate_aruco_dict.py`) and refines 3D marker pose and measured
/// edge length on the interior LiDAR depth plane (10% inward border erosion).
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
