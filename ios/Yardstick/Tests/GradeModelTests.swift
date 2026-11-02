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

/// YS-44 / YS-111: the phone's grade equals the toolkit's on the shared vectors in
/// `toolkit/tests/fixtures/grade_parity_vectors.json` (expected values written by the toolkit itself),
/// including walks 6 (C) and 7 (fail), which the old phone formula graded B and C.
final class GradeModelTests: XCTestCase {
    private struct Vector: Decodable {
        let name: String
        let tapes: [[AnyCodableValue]]
        let scale_eff: Double
        let gap: Double?
        let jumps: [Double]
        let walk: Double
        let limited: Double
        let normal_ok: Bool?
        let expected_grade: String
        let expected_distance_m: Double
        let expected_caps: [String]
    }

    private enum AnyCodableValue: Decodable {
        case s(String), d(Double)
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let v = try? c.decode(Double.self) { self = .d(v) } else { self = .s(try c.decode(String.self)) }
        }
        var string: String { if case .s(let v) = self { return v }; return "" }
        var double: Double { if case .d(let v) = self { return v }; return .nan }
    }

    private func loadVectors() throws -> [Vector] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // Yardstick
            .deletingLastPathComponent()  // ios
            .deletingLastPathComponent()  // repo
            .appendingPathComponent("toolkit/tests/fixtures/grade_parity_vectors.json")
        return try JSONDecoder().decode([Vector].self, from: Data(contentsOf: url))
    }

    func testPhoneGradeEqualsToolkitOnSharedVectors() throws {
        let vectors = try loadVectors()
        XCTAssertGreaterThanOrEqual(vectors.count, 11)
        for v in vectors {
            var inp = GradeModel.Inputs()
            inp.tapeChecks = v.tapes.map { t in
                let tape = t[1].double
                let u = t.count > 3 ? t[3].double : 0.0
                return GradeModel.TapeCheck(
                    tapeM: tape, residualM: abs(t[2].double - tape), isHeldOut: t[0].string == "held_out", tapeUncertaintyM: u)
            }
            inp.scaleErrorEff = v.scale_eff
            inp.rawMarker0GapM = v.gap
            inp.poseJumpSumM = v.jumps.reduce(0, +)
            inp.walkLengthM = v.walk
            inp.limitedTrackingShare = v.limited
            inp.marker0NormalOk = v.normal_ok ?? true
            let r = GradeModel.grade(inp)
            XCTAssertEqual(r.grade, v.expected_grade, v.name)
            XCTAssertEqual(r.distanceTolerance10mM, v.expected_distance_m, accuracy: 1e-6, v.name)
            XCTAssertEqual(r.capsApplied, v.expected_caps, v.name)
        }
    }

    func testScaleErrorEffMatchesToolkitAggregation() {
        // Single sighting → SE 0.5 %: 1 % error − 2·0.5 % = 0.
        XCTAssertEqual(GradeModel.scaleErrorEff(sizesByMarker: [0: (180, [181.8])]), 0.0, accuracy: 1e-12)
        // Worst marker wins: marker 2 reads 1.0 % big with tight spread.
        let eff = GradeModel.scaleErrorEff(sizesByMarker: [
            0: (180, [180.1, 179.9, 180.0]),
            2: (180, [181.8, 181.8, 181.8, 181.8]),
        ])
        XCTAssertEqual(eff, 0.01, accuracy: 1e-9)
    }
}
