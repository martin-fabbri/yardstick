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

import XCTest
import simd

/// YS-88: the phone's pose-jump detector matches the toolkit's `detect_pose_jumps` on the toolkit's own vectors
/// (`toolkit/tests/test_yardstick.py::test_round4_brisk_walk_snap_across_gap_and_multiframe_ramp`) and on walk 7.
final class PoseJumpDetectorTests: XCTestCase {
    private func run(_ rows: [(t: Double, x: Double, y: Double, z: Double)]) -> [DetectedPoseJump] {
        // The toolkit reads poses.csv written with 6 decimals; round the same way.
        let r6 = { (v: Double) in Double(String(format: "%.6f", v))! }
        return PoseJumpDetector.detect(
            timestamps: rows.map { r6($0.t) },
            positions: rows.map { SIMD3(r6($0.x), r6($0.y), r6($0.z)) },
            savedIndices: Array(0..<rows.count)
        )
    }

    func testYS88_ToolkitVectors_BriskWalkGapSnapAndRamp() {
        let brisk = (0..<150).map { i -> (t: Double, x: Double, y: Double, z: Double) in
            let t = Double(i) / 30.0
            return (t, 1.4 * t, 0.9, 0.25 * sin(t))
        }
        XCTAssertEqual(run(brisk), [], "A clean 1.4 m/s walk has no jumps")

        let gap = (0..<40).map { i -> (t: Double, x: Double, y: Double, z: Double) in
            let t = i < 15 ? Double(i) / 30.0 : Double(i + 4) / 30.0
            return (t, 1.1 * t + (i >= 15 ? 1.0 : 0.0), 0.9, 0.0)
        }
        let jg = run(gap)
        XCTAssertEqual(jg.count, 1)
        XCTAssertEqual(jg.first?.jumpM ?? 0, 1.0, accuracy: 0.02)

        let ramp = (0..<45).map { i -> (t: Double, x: Double, y: Double, z: Double) in
            let t = Double(i) / 30.0
            let r = i <= 10 ? 0.0 : (i <= 25 ? 0.30 * Double(i - 10) / 15.0 : 0.30)
            return (t, 1.0 * t + r, 0.9, 0.0)
        }
        let jr = run(ramp)
        XCTAssertEqual(jr.count, 1)
        XCTAssertEqual(jr.first?.jumpM ?? 0, 0.30, accuracy: 0.02)
    }

    /// Walk 7 (2026-11-01): the toolkit found two jumps (0.108140 m at saved index 8339, 0.093554 m at 9164).
    func testYS88_Walk7RealPoses_MatchToolkitJumpsExactly() throws {
        let root = ProcessInfo.processInfo.environment["YARDSTICK_REPLAY_ROOT"] ?? "/Users/martinfabbri/Workspace/scrap/yardstick-captures"
        let url = URL(fileURLWithPath: root)
            .appendingPathComponent("2026-11-01/walk-7/20261102T004449Z_yardstick/poses.csv")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw XCTSkip("walk 7 poses.csv not present at \(url.path)")
        }
        var lines = text.split(separator: "\n")
        let header = lines.removeFirst().split(separator: ",").map(String.init)
        let it = header.firstIndex(of: "t")!, iIdx = header.firstIndex(of: "saved_index")!
        let ix = header.firstIndex(of: "px")!, iy = header.firstIndex(of: "py")!, iz = header.firstIndex(of: "pz")!
        var ts: [Double] = [], ps: [SIMD3<Double>] = [], idx: [Int] = []
        for l in lines {
            let c = l.split(separator: ",", omittingEmptySubsequences: false)
            ts.append(Double(c[it])!); idx.append(Int(c[iIdx])!)
            ps.append(SIMD3(Double(c[ix])!, Double(c[iy])!, Double(c[iz])!))
        }
        let jumps = PoseJumpDetector.detect(timestamps: ts, positions: ps, savedIndices: idx)
        XCTAssertEqual(jumps.map(\.savedIndexFrom), [8339, 9164])
        XCTAssertEqual(jumps.map(\.savedIndexTo), [8341, 9165])
        XCTAssertEqual(jumps.first?.jumpM ?? 0, 0.108140, accuracy: 1e-6)
        XCTAssertEqual(jumps.last?.jumpM ?? 0, 0.093554, accuracy: 1e-6)
        XCTAssertEqual(jumps.reduce(0) { $0 + $1.jumpM }, 0.201694, accuracy: 1e-6, "Toolkit walk 7 loop gap = 0.201694 m")
    }
}
