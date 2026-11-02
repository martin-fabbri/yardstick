import Foundation
import XCTest

#if canImport(Yardstick)
@testable import Yardstick
#endif

final class VisitPlanDecodingTests: XCTestCase {

    private func fixturesRootURL() -> URL {
        // .../yardstick/ios/Yardstick/Tests/VisitPlanDecodingTests.swift
        // Going up 5 levels reaches .../Workspace/github/
        let thisFile = URL(fileURLWithPath: #filePath)
        let githubRoot = thisFile
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // Yardstick
            .deletingLastPathComponent() // ios
            .deletingLastPathComponent() // yardstick
            .deletingLastPathComponent() // github
        return githubRoot
            .appendingPathComponent("lotfit-workbench")
            .appendingPathComponent("intake")
            .appendingPathComponent("contracts")
            .appendingPathComponent("fixtures")
    }

    func testValidVisitPlanFixturesDecodeAndValidate() throws {
        let validDir = fixturesRootURL().appendingPathComponent("valid")
        let files = try FileManager.default.contentsOfDirectory(
            at: validDir,
            includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasPrefix("VisitPlan.") && $0.pathExtension == "json" }

        XCTAssertFalse(files.isEmpty, "Expected at least one valid VisitPlan.*.json fixture in \(validDir.path)")

        for fileURL in files {
            let data = try Data(contentsOf: fileURL)
            let plan = try VisitPlan.decodeAndValidate(from: data)

            XCTAssertEqual(plan.schema, "lotfit.visit-plan/1")
            XCTAssertEqual(plan.contract, "lotfit.visit-plan/1")
            XCTAssertEqual(plan.anchor.primary.lengthM, 17.13, accuracy: 1e-6)
            XCTAssertEqual(plan.markers.count, 4)
            XCTAssertEqual(plan.markers.first(where: { $0.id == 0 })?.placement, .flatGround)
            XCTAssertEqual(plan.expectedReadings.first?.reference, .recordOutline)
        }
    }

    func testToleratesUnknownExtraFieldsOnValidFixture() throws {
        let validURL = fixturesRootURL()
            .appendingPathComponent("valid")
            .appendingPathComponent("VisitPlan.eug-martin.json")
        let data = try Data(contentsOf: validURL)
        var jsonObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        jsonObject["future_v1_2_extra_field"] = ["unknown": true, "count": 42]
        let augmentedData = try JSONSerialization.data(withJSONObject: jsonObject)

        let plan = try VisitPlan.decodeAndValidate(from: augmentedData)
        XCTAssertEqual(plan.planId, "019a4500-0000-7000-8000-000000000301")
    }

    func testInvalidVisitPlanFixturesAreRejectedWithExpectedRule() throws {
        let invalidDir = fixturesRootURL().appendingPathComponent("invalid")
        let files = try FileManager.default.contentsOfDirectory(
            at: invalidDir,
            includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasPrefix("VisitPlan.") && $0.pathExtension == "json" }

        XCTAssertGreaterThanOrEqual(
            files.count,
            4,
            "Expected at least 4 invalid VisitPlan.*.json fixtures in \(invalidDir.path)"
        )

        for fileURL in files {
            let filename = fileURL.lastPathComponent
            // VisitPlan.<rule>.json
            let parts = filename.split(separator: ".")
            guard parts.count == 3 else {
                XCTFail("Unexpected fixture filename format: \(filename)")
                continue
            }
            let expectedRule = String(parts[1])
            let data = try Data(contentsOf: fileURL)

            XCTAssertThrowsError(
                try VisitPlan.decodeAndValidate(from: data),
                "Expected invalid fixture \(filename) to fail decoding or contract validation"
            ) { error in
                if let validationError = error as? VisitPlanValidationError {
                    XCTAssertEqual(
                        validationError.ruleName,
                        expectedRule,
                        "Fixture \(filename) failed with wrong rule: \(validationError)"
                    )
                } else if error is DecodingError {
                    // Structural / missing-required-field fixtures (e.g. missing_expected_reading_reference)
                    // fail directly in Swift's Codable decoder.
                    XCTAssertTrue(
                        expectedRule.hasPrefix("missing_") || expectedRule.hasPrefix("invalid_"),
                        "Unexpected DecodingError for rule \(expectedRule) in \(filename): \(error)"
                    )
                } else {
                    XCTFail("Unexpected error type for \(filename): \(error)")
                }
            }
        }
    }
}
