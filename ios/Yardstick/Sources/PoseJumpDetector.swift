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
import simd

/// One non-physical pose jump (ARKit relocalization snap or multi-frame ramp), as in `poses.csv` indices.
public struct DetectedPoseJump: Sendable, Equatable {
    public let savedIndexFrom: Int
    public let savedIndexTo: Int
    public let timestampS: Double
    public let jumpM: Double
}

/// Swift port of the toolkit's `detect_pose_jumps` (`toolkit/src/yardstick/geometry.py`, YS-32, Q026), so the phone
/// and the toolkit compute the same loop gap (`gap = max(raw gap, sum of jumps)`, YS-32 / YS-88).
/// Keep the two in step: the shared vectors live in `PoseJumpDetectorTests`.
public enum PoseJumpDetector {
    public static func detect(
        timestamps ts: [Double],
        positions pos: [SIMD3<Double>],
        savedIndices sIdx: [Int],
        jumpThresholdM: Double = 0.08,
        maxStepDtS: Double = 2.50,
        maxRampDtS: Double = 0.80
    ) -> [DetectedPoseJump] {
        let n = min(ts.count, pos.count, sIdx.count)
        guard n >= 2 else { return [] }
        let m = n - 1
        var dp = [SIMD3<Double>](repeating: .zero, count: m)
        var dt = [Double](repeating: 0, count: m)
        var vel = [SIMD3<Double>](repeating: .zero, count: m)
        var speeds = [Double](repeating: 0, count: m)
        for k in 0..<m {
            dp[k] = pos[k + 1] - pos[k]
            dt[k] = ts[k + 1] - ts[k]
            let vdt = dt[k] > 1e-6 ? dt[k] : 1e-6
            vel[k] = dp[k] / vdt
            speeds[k] = simd_length(vel[k])
        }

        func scaled(_ v: SIMD3<Double>, to s: Double) -> SIMD3<Double> {
            let nl = simd_length(v)
            return nl > 1e-6 ? (v / nl) * s : v
        }

        // 1. Single-step snaps against the speed-preserving midpoint-time velocity prediction.
        var rawRes = [Double](repeating: 0, count: m)
        for k in 0..<m {
            if dt[k] <= 0.0 || dt[k] > maxStepDtS { continue }
            var vHat: SIMD3<Double>
            if m == 1 {
                vHat = .zero
            } else if k > 0 && k < m - 1 {
                let denom = dt[k - 1] + 2.0 * dt[k] + dt[k + 1]
                let wL = (dt[k] + dt[k + 1]) / denom
                let wR = (dt[k] + dt[k - 1]) / denom
                let vLin = wL * vel[k - 1] + wR * vel[k + 1]
                let sHat = wL * speeds[k - 1] + wR * speeds[k + 1]
                vHat = scaled(vLin, to: sHat)
            } else if k == 0 {
                if m >= 3 {
                    let tau01 = dt[0] + dt[1]
                    let tau12 = max(dt[1] + dt[2], 1e-6)
                    let vLin = vel[1] - (tau01 / tau12) * (vel[2] - vel[1])
                    vHat = scaled(vLin, to: speeds[1])
                } else {
                    vHat = vel[1]
                }
            } else {
                if m >= 3 {
                    let tauA = dt[m - 1] + dt[m - 2]
                    let tauB = max(dt[m - 2] + dt[m - 3], 1e-6)
                    let vLin = vel[m - 2] + (tauA / tauB) * (vel[m - 2] - vel[m - 3])
                    vHat = scaled(vLin, to: speeds[m - 2])
                } else {
                    vHat = vel[m - 2]
                }
            }
            rawRes[k] = simd_length(dp[k] - vHat * dt[k])
        }

        var usedSteps = Set<Int>()
        var jumps: [DetectedPoseJump] = []
        for k in 0..<m where rawRes[k] > jumpThresholdM {
            let leftR = k > 0 ? rawRes[k - 1] : 0.0
            let rightR = k + 1 < m ? rawRes[k + 1] : 0.0
            if rawRes[k] >= leftR && rawRes[k] >= rightR {
                usedSteps.insert(k)
                jumps.append(DetectedPoseJump(
                    savedIndexFrom: sIdx[k], savedIndexTo: sIdx[k + 1],
                    timestampS: (ts[k + 1] * 1e6).rounded() / 1e6, jumpM: (rawRes[k] * 1e6).rounded() / 1e6))
            }
        }

        // 2. Multi-frame ramps (2…20 frames within maxRampDtS) where pre- and post-ramp velocities agree.
        var candidates: [(score: Double, resM: Double, j: Int, i: Int, t: Double)] = []
        let maxW = min(20, m)
        if maxW >= 2 {
            for w in 2...maxW {
                var j = 1
                while j < n - w - 1 {
                    let i = j + w
                    defer { j += 1 }
                    if ((j - 1)...i).contains(where: { usedSteps.contains($0) }) { continue }
                    let dtWin = ts[i] - ts[j]
                    if dtWin <= 0.0 || dtWin > maxRampDtS { continue }
                    let vPre = vel[j - 1]
                    let vPost = vel[i]
                    if simd_length(vPost - vPre) * dtWin > 0.5 * jumpThresholdM { continue }
                    let vHat = 0.5 * (vPre + vPost)
                    let diffVec = (pos[i] - pos[j]) - vHat * dtWin
                    let resM = simd_length(diffVec)
                    if resM > jumpThresholdM {
                        let uRamp = diffVec / resM
                        var minExcess = Double.infinity
                        for st in j..<i {
                            minExcess = min(minExcess, simd_dot(dp[st] - vHat * dt[st], uRamp))
                        }
                        if minExcess <= 0.002 { continue }
                        candidates.append((resM - 1e-4 * Double(w), resM, j, i, ts[i]))
                    }
                }
            }
        }
        // Python's sort is stable; keep insertion order on equal scores.
        let ordered = candidates.enumerated().sorted { a, b in
            a.element.score != b.element.score ? a.element.score > b.element.score : a.offset < b.offset
        }.map(\.element)
        for c in ordered {
            let stepRange = Set(c.j..<c.i)
            if !stepRange.isDisjoint(with: usedSteps) { continue }
            usedSteps.formUnion(stepRange)
            jumps.append(DetectedPoseJump(
                savedIndexFrom: sIdx[c.j], savedIndexTo: sIdx[c.i],
                timestampS: (c.t * 1e6).rounded() / 1e6, jumpM: (c.resM * 1e6).rounded() / 1e6))
        }
        return jumps.sorted { $0.savedIndexFrom < $1.savedIndexFrom }
    }
}
