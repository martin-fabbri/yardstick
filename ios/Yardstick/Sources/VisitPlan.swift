import Foundation

// MARK: - lotfit.visit-plan/1 Codable Contract Model

public enum VisitPlanValidationError: Error, LocalizedError, Equatable, Sendable {
    case invalidSchemaVersion(String)
    case invalidContractVersion(String)
    case ruleViolation(rule: String, detail: String)

    public var ruleName: String {
        switch self {
        case .invalidSchemaVersion:
            return "invalid_schema_version"
        case .invalidContractVersion:
            return "invalid_contract_version"
        case .ruleViolation(let rule, _):
            return rule
        }
    }

    public var errorDescription: String? {
        switch self {
        case .invalidSchemaVersion(let got):
            return "[invalid_schema_version] Expected schema == \"lotfit.visit-plan/1\", got \"\(got)\""
        case .invalidContractVersion(let got):
            return "[invalid_contract_version] Expected contract == \"lotfit.visit-plan/1\", got \"\(got)\""
        case .ruleViolation(let rule, let detail):
            return "[\(rule)] \(detail)"
        }
    }
}

public struct VisitPlan: Codable, Equatable, Sendable {
    public static let expectedSchema = "lotfit.visit-plan/1"
    public static let expectedContract = "lotfit.visit-plan/1"
    public static let feetPerMeter: Double = 3.280839895013123
    public static let metricTwinToleranceM: Double = 0.001

    public let schema: String
    public let contract: String
    public let planId: String
    public let dossierId: String
    public let createdAt: String
    public let addressLine: String
    public let walkerRole: String
    public let markerDictionary: String
    public let markerSizeMmPrinted: Int
    public let map: MapReference
    public let records: [RecordSnapshot]
    public let anchor: AnchorSpec
    public let markers: [MarkerPin]
    public let mustCapture: [MustCaptureItem]
    public let expectedReadings: [ExpectedReading]
    public let tapeChecks: [TapeCheck]
    public let walkSteps: [WalkStep]
    public let estimatedWalkMin: Double

    enum CodingKeys: String, CodingKey {
        case schema
        case contract
        case planId = "plan_id"
        case dossierId = "dossier_id"
        case createdAt = "created_at"
        case addressLine = "address_line"
        case walkerRole = "walker_role"
        case markerDictionary = "marker_dictionary"
        case markerSizeMmPrinted = "marker_size_mm_printed"
        case map
        case records
        case anchor
        case markers
        case mustCapture = "must_capture"
        case expectedReadings = "expected_readings"
        case tapeChecks = "tape_checks"
        case walkSteps = "walk_steps"
        case estimatedWalkMin = "estimated_walk_min"
    }

    public struct FrontChoice: Codable, Equatable, Sendable {
        public let edgeIndex: Int
        public let streetName: String?
        public let cardinalSide: String
        public let rule: String
        public let reason: String

        enum CodingKeys: String, CodingKey {
            case edgeIndex = "edge_index"
            case streetName = "street_name"
            case cardinalSide = "cardinal_side"
            case rule
            case reason
        }
    }

    public struct DossierFrame: Codable, Equatable, Sendable {
        public let originLonLat: [Double]
        public let plusXBearingDeg: Double
        public let northInLotCcwDeg: Double
        public let metersPerDegLon: Double
        public let metersPerDegLat: Double
        public let ftPerM: Double
        public let detXy: Int
        public let projectionResidualFt: Double
        public let frontChoice: FrontChoice

        enum CodingKeys: String, CodingKey {
            case originLonLat = "origin_lon_lat"
            case plusXBearingDeg = "plus_x_bearing_deg"
            case northInLotCcwDeg = "north_in_lot_ccw_deg"
            case metersPerDegLon = "meters_per_deg_lon"
            case metersPerDegLat = "meters_per_deg_lat"
            case ftPerM = "ft_per_m"
            case detXy = "det_xy"
            case projectionResidualFt = "projection_residual_ft"
            case frontChoice = "front_choice"
        }
    }

    public struct MapZone: Codable, Equatable, Sendable {
        public let zoneId: String
        public let label: String
        public let polygonLotFt: [[Double]]
        public let polygonLotM: [[Double]]

        enum CodingKeys: String, CodingKey {
            case zoneId = "zone_id"
            case label
            case polygonLotFt = "polygon_lot_ft"
            case polygonLotM = "polygon_lot_m"
        }
    }

    public struct MapPin: Codable, Equatable, Sendable {
        public let id: String
        public let kind: String
        public let label: String
        public let atLotFt: [Double]
        public let atLotM: [Double]

        enum CodingKeys: String, CodingKey {
            case id
            case kind
            case label
            case atLotFt = "at_lot_ft"
            case atLotM = "at_lot_m"
        }
    }

    public struct MapReference: Codable, Equatable, Sendable {
        public let imagePath: String?
        public let imageSha256: String?
        public let widthPx: Int?
        public let heightPx: Int?
        public let cornersLotFt: [[Double]]?
        public let cornersLotM: [[Double]]?
        public let cornersLonLat: [[Double]]?
        public let lotOutlineFt: [[Double]]
        public let lotOutlineM: [[Double]]
        public let houseOutlineFt: [[Double]]
        public let houseOutlineM: [[Double]]
        public let possibleZonesFt: [[[Double]]]
        public let possibleZonesM: [[[Double]]]
        public let zones: [MapZone]
        public let pins: [MapPin]
        public let frame: DossierFrame

        enum CodingKeys: String, CodingKey {
            case imagePath = "image_path"
            case imageSha256 = "image_sha256"
            case widthPx = "width_px"
            case heightPx = "height_px"
            case cornersLotFt = "corners_lot_ft"
            case cornersLotM = "corners_lot_m"
            case cornersLonLat = "corners_lon_lat"
            case lotOutlineFt = "lot_outline_ft"
            case lotOutlineM = "lot_outline_m"
            case houseOutlineFt = "house_outline_ft"
            case houseOutlineM = "house_outline_m"
            case possibleZonesFt = "possible_zones_ft"
            case possibleZonesM = "possible_zones_m"
            case zones
            case pins
            case frame
        }
    }

    public struct LayerSource: Codable, Equatable, Sendable {
        public let issuer: String
        public let serviceUrl: String
        public let query: String
        public let document: String?

        enum CodingKeys: String, CodingKey {
            case issuer
            case serviceUrl = "service_url"
            case query
            case document
        }
    }

    public struct StatedAccuracy: Codable, Equatable, Sendable {
        public let value: Double
        public let unit: String
    }

    public struct AssumedAccuracy: Codable, Equatable, Sendable {
        public let value: Double
        public let unit: String
        public let by: String
        public let why: String
    }

    public struct RecordSnapshot: Codable, Equatable, Sendable {
        public let layer: String
        public let status: String
        public let reason: String?
        public let source: LayerSource?
        public let recordDate: String?
        public let statedAccuracy: StatedAccuracy?
        public let assumedAccuracy: AssumedAccuracy?

        enum CodingKeys: String, CodingKey {
            case layer
            case status
            case reason
            case source
            case recordDate = "record_date"
            case statedAccuracy = "stated_accuracy"
            case assumedAccuracy = "assumed_accuracy"
        }
    }

    public struct AnchorPrimary: Codable, Equatable, Sendable {
        public let wallId: String
        public let cornerALotFt: [Double]
        public let cornerBLotFt: [Double]
        public let cornerALotM: [Double]
        public let cornerBLotM: [Double]
        public let lengthFt: Double
        public let lengthM: Double
        public let toleranceFt: Double
        public let toleranceM: Double
        public let why: String

        enum CodingKeys: String, CodingKey {
            case wallId = "wall_id"
            case cornerALotFt = "corner_a_lot_ft"
            case cornerBLotFt = "corner_b_lot_ft"
            case cornerALotM = "corner_a_lot_m"
            case cornerBLotM = "corner_b_lot_m"
            case lengthFt = "length_ft"
            case lengthM = "length_m"
            case toleranceFt = "tolerance_ft"
            case toleranceM = "tolerance_m"
            case why
        }
    }

    public struct AnchorFallback: Codable, Equatable, Sendable {
        public let kind: String
        public let cornerALotFt: [Double]
        public let cornerBLotFt: [Double]
        public let cornerALotM: [Double]
        public let cornerBLotM: [Double]
        public let instruction: String

        enum CodingKeys: String, CodingKey {
            case kind
            case cornerALotFt = "corner_a_lot_ft"
            case cornerBLotFt = "corner_b_lot_ft"
            case cornerALotM = "corner_a_lot_m"
            case cornerBLotM = "corner_b_lot_m"
            case instruction
        }
    }

    public struct AnchorSpec: Codable, Equatable, Sendable {
        public let primary: AnchorPrimary
        public let fallback: AnchorFallback?
    }

    public enum MarkerRole: String, Codable, Equatable, Sendable {
        case originAnchorA = "origin_anchor_a"
        case anchorB = "anchor_b"
        case zone = "zone"
        case frontFallback = "front_fallback"
    }

    public enum MarkerPlacement: String, Codable, Equatable, Sendable {
        case flatGround = "flat_ground"
        case upright = "upright"
    }

    public struct LotCoordsFt: Codable, Equatable, Sendable {
        public let xFt: Double
        public let yFt: Double

        enum CodingKeys: String, CodingKey {
            case xFt = "x_ft"
            case yFt = "y_ft"
        }
    }

    public struct LotCoordsM: Codable, Equatable, Sendable {
        public let xM: Double
        public let yM: Double

        enum CodingKeys: String, CodingKey {
            case xM = "x_m"
            case yM = "y_m"
        }
    }

    public struct MarkerPin: Codable, Equatable, Sendable {
        public let id: Int
        public let role: MarkerRole
        public let dictionary: String
        public let sizeMmPrinted: Int
        public let atLotFt: LotCoordsFt
        public let atLotM: LotCoordsM
        public let placement: MarkerPlacement
        public let plusXBearingInLotDeg: Double?
        public let wallOffsetM: Double?
        public let instruction: String

        enum CodingKeys: String, CodingKey {
            case id
            case role
            case dictionary
            case sizeMmPrinted = "size_mm_printed"
            case atLotFt = "at_lot_ft"
            case atLotM = "at_lot_m"
            case placement
            case plusXBearingInLotDeg = "plus_x_bearing_in_lot_deg"
            case wallOffsetM = "wall_offset_m"
            case instruction
        }
    }

    public enum CaptureGeometry: Codable, Equatable, Sendable {
        case point([Double])
        case path([[Double]])

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let pt = try? container.decode([Double].self) {
                self = .point(pt)
            } else if let pts = try? container.decode([[Double]].self) {
                self = .path(pts)
            } else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Expected [Double] or [[Double]] for must_capture.where geometry"
                )
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .point(let pt):
                try container.encode(pt)
            case .path(let pts):
                try container.encode(pts)
            }
        }
    }

    public struct MustCaptureWhere: Codable, Equatable, Sendable {
        public let kind: String
        public let lotFt: CaptureGeometry
        public let lotM: CaptureGeometry

        enum CodingKeys: String, CodingKey {
            case kind
            case lotFt = "lot_ft"
            case lotM = "lot_m"
        }
    }

    public enum CaptureKind: String, Codable, Equatable, Sendable {
        case anchorCorner = "anchor_corner"
        case wall = "wall"
        case eave = "eave"
        case zoneCoverage = "zone_coverage"
        case passage = "passage"
        case tag = "tag"
    }

    public struct MustCaptureItem: Codable, Equatable, Sendable {
        public let id: String
        public let kind: CaptureKind
        public let tagKind: String?
        public let label: String
        public let location: MustCaptureWhere
        public let why: String
        public let required: Bool
        public let instruction: String

        enum CodingKeys: String, CodingKey {
            case id
            case kind
            case tagKind = "tag_kind"
            case label
            case location = "where"
            case why
            case required
            case instruction
        }
    }

    public enum ReadingReference: String, Codable, Equatable, Sendable {
        case recordOutline = "record_outline"
        case wallFoot = "wall_foot"
        case roofEdge = "roof_edge"
    }

    public struct ExpectedReading: Codable, Equatable, Sendable {
        public let id: String
        public let kind: String
        public let checkedWhen: String
        public let between: [String]
        public let reference: ReadingReference
        public let assumedEaveDepthM: Double?
        public let assumedEaveDepthFt: Double?
        public let expectedM: Double
        public let toleranceM: Double
        public let expectedFt: Double
        public let toleranceFt: Double
        public let source: String
        public let warningMessage: String

        enum CodingKeys: String, CodingKey {
            case id
            case kind
            case checkedWhen = "checked_when"
            case between
            case reference
            case assumedEaveDepthM = "assumed_eave_depth_m"
            case assumedEaveDepthFt = "assumed_eave_depth_ft"
            case expectedM = "expected_m"
            case toleranceM = "tolerance_m"
            case expectedFt = "expected_ft"
            case toleranceFt = "tolerance_ft"
            case source
            case warningMessage = "warning_message"
        }
    }

    public struct TapeCheck: Codable, Equatable, Sendable {
        public let id: Int
        public let role: String
        public let label: String
        public let fromId: String
        public let toId: String
        public let expectedM: Double?
        public let expectedFt: Double?
        public let minLengthM: Double
        public let minLengthFt: Double?
        public let tapeUncertaintyM: Double
        public let tapeUncertaintyFt: Double
        public let instruction: String

        enum CodingKeys: String, CodingKey {
            case id
            case role
            case label
            case fromId = "from_id"
            case toId = "to_id"
            case expectedM = "expected_m"
            case expectedFt = "expected_ft"
            case minLengthM = "min_length_m"
            case minLengthFt = "min_length_ft"
            case tapeUncertaintyM = "tape_uncertainty_m"
            case tapeUncertaintyFt = "tape_uncertainty_ft"
            case instruction
        }
    }

    public struct WalkStep: Codable, Equatable, Sendable {
        public let stepIndex: Int
        public let phase: String
        public let yardstickEnginePhase: String
        public let title: String
        public let instruction: String
        public let targetIds: [String]

        enum CodingKeys: String, CodingKey {
            case stepIndex = "step_index"
            case phase
            case yardstickEnginePhase = "yardstick_engine_phase"
            case title
            case instruction
            case targetIds = "target_ids"
        }
    }

    // MARK: - Decode + Validate

    public static func decodeAndValidate(from data: Data) throws -> VisitPlan {
        let decoder = JSONDecoder()
        let plan = try decoder.decode(VisitPlan.self, from: data)
        try plan.validateContractRules()
        return plan
    }

    public func validateContractRules() throws {
        guard schema == Self.expectedSchema else {
            throw VisitPlanValidationError.invalidSchemaVersion(schema)
        }
        guard contract == Self.expectedContract else {
            throw VisitPlanValidationError.invalidContractVersion(contract)
        }

        // Rule 1: marker_0_flat_ground
        guard let marker0 = markers.first(where: { $0.id == 0 }) else {
            throw VisitPlanValidationError.ruleViolation(
                rule: "marker_0_flat_ground",
                detail: "markers must include Marker 0 (id == 0)"
            )
        }
        guard marker0.role == .originAnchorA && marker0.placement == .flatGround else {
            throw VisitPlanValidationError.ruleViolation(
                rule: "marker_0_flat_ground",
                detail: "Marker 0 must have role == \"origin_anchor_a\" and placement == \"flat_ground\""
            )
        }
        for m in markers where m.role == .originAnchorA {
            if m.placement != .flatGround {
                throw VisitPlanValidationError.ruleViolation(
                    rule: "marker_0_flat_ground",
                    detail: "Marker \(m.id) (origin_anchor_a) must have placement == \"flat_ground\""
                )
            }
        }

        // Rule 2: map_image_corners_required
        if map.imagePath != nil {
            guard map.imageSha256 != nil,
                  map.widthPx != nil,
                  map.heightPx != nil,
                  map.cornersLotFt != nil,
                  map.cornersLotM != nil,
                  map.cornersLonLat != nil else {
                throw VisitPlanValidationError.ruleViolation(
                    rule: "map_image_corners_required",
                    detail: "When map.image_path is non-null, image_sha256, width_px, height_px, corners_lot_ft, corners_lot_m, and corners_lon_lat must all be non-null"
                )
            }
        }

        // Rule 3: metric_twin_consistency_1mm
        try Self.checkPoint2DTwin(
            ft: anchor.primary.cornerALotFt,
            m: anchor.primary.cornerALotM,
            path: "anchor.primary.corner_a"
        )
        try Self.checkPoint2DTwin(
            ft: anchor.primary.cornerBLotFt,
            m: anchor.primary.cornerBLotM,
            path: "anchor.primary.corner_b"
        )
        try Self.checkScalarTwin(
            ft: anchor.primary.lengthFt,
            m: anchor.primary.lengthM,
            path: "anchor.primary.length"
        )
        try Self.checkScalarTwin(
            ft: anchor.primary.toleranceFt,
            m: anchor.primary.toleranceM,
            path: "anchor.primary.tolerance"
        )

        if let fallback = anchor.fallback {
            try Self.checkPoint2DTwin(
                ft: fallback.cornerALotFt,
                m: fallback.cornerALotM,
                path: "anchor.fallback.corner_a"
            )
            try Self.checkPoint2DTwin(
                ft: fallback.cornerBLotFt,
                m: fallback.cornerBLotM,
                path: "anchor.fallback.corner_b"
            )
        }

        for marker in markers {
            try Self.checkScalarTwin(
                ft: marker.atLotFt.xFt,
                m: marker.atLotM.xM,
                path: "marker[\(marker.id)].x"
            )
            try Self.checkScalarTwin(
                ft: marker.atLotFt.yFt,
                m: marker.atLotM.yM,
                path: "marker[\(marker.id)].y"
            )
        }

        for item in mustCapture {
            switch (item.location.lotFt, item.location.lotM) {
            case (.point(let ft), .point(let m)):
                try Self.checkPoint2DTwin(ft: ft, m: m, path: "must_capture(\(item.id)).where")
            case (.path(let ptsFt), .path(let ptsM)):
                try Self.checkPathTwin(ft: ptsFt, m: ptsM, path: "must_capture(\(item.id)).where")
            default:
                throw VisitPlanValidationError.ruleViolation(
                    rule: "metric_twin_consistency_1mm",
                    detail: "must_capture(\(item.id)).where geometry mismatch between lot_ft and lot_m"
                )
            }
        }

        for reading in expectedReadings {
            try Self.checkScalarTwin(
                ft: reading.expectedFt,
                m: reading.expectedM,
                path: "expected_reading(\(reading.id)).expected"
            )
            try Self.checkScalarTwin(
                ft: reading.toleranceFt,
                m: reading.toleranceM,
                path: "expected_reading(\(reading.id)).tolerance"
            )
            if (reading.assumedEaveDepthFt == nil) != (reading.assumedEaveDepthM == nil) {
                throw VisitPlanValidationError.ruleViolation(
                    rule: "metric_twin_consistency_1mm",
                    detail: "expected_reading(\(reading.id)) must provide both assumed_eave_depth_ft and assumed_eave_depth_m or neither"
                )
            }
            if let eaveFt = reading.assumedEaveDepthFt, let eaveM = reading.assumedEaveDepthM {
                try Self.checkScalarTwin(
                    ft: eaveFt,
                    m: eaveM,
                    path: "expected_reading(\(reading.id)).assumed_eave_depth"
                )
            }
        }

        for tape in tapeChecks {
            if (tape.expectedFt == nil) != (tape.expectedM == nil) {
                throw VisitPlanValidationError.ruleViolation(
                    rule: "metric_twin_consistency_1mm",
                    detail: "tape_check[\(tape.id)] must provide both expected_ft and expected_m or neither"
                )
            }
            if let ft = tape.expectedFt, let m = tape.expectedM {
                try Self.checkScalarTwin(ft: ft, m: m, path: "tape_check[\(tape.id)].expected")
            }
            if let minFt = tape.minLengthFt {
                try Self.checkScalarTwin(ft: minFt, m: tape.minLengthM, path: "tape_check[\(tape.id)].min_length")
            }
            try Self.checkScalarTwin(
                ft: tape.tapeUncertaintyFt,
                m: tape.tapeUncertaintyM,
                path: "tape_check[\(tape.id)].tape_uncertainty"
            )
        }

        if let cFt = map.cornersLotFt, let cM = map.cornersLotM {
            try Self.checkPathTwin(ft: cFt, m: cM, path: "map.corners_lot")
        }
        try Self.checkPathTwin(ft: map.lotOutlineFt, m: map.lotOutlineM, path: "map.lot_outline")
        try Self.checkPathTwin(ft: map.houseOutlineFt, m: map.houseOutlineM, path: "map.house_outline")
        for zone in map.zones {
            try Self.checkPathTwin(ft: zone.polygonLotFt, m: zone.polygonLotM, path: "map.zones(\(zone.zoneId))")
        }
        for pin in map.pins {
            try Self.checkPoint2DTwin(ft: pin.atLotFt, m: pin.atLotM, path: "map.pins(\(pin.id))")
        }
    }

    private static func checkScalarTwin(ft: Double, m: Double, path: String) throws {
        let mFromFt = ft / feetPerMeter
        let diff = abs(mFromFt - m)
        if diff > metricTwinToleranceM {
            throw VisitPlanValidationError.ruleViolation(
                rule: "metric_twin_consistency_1mm",
                detail: "\(path): |\(ft) ft / \(feetPerMeter) - \(m) m| = \(diff) m > \(metricTwinToleranceM) m"
            )
        }
    }

    private static func checkPoint2DTwin(ft: [Double], m: [Double], path: String) throws {
        guard ft.count == 2, m.count == 2 else {
            throw VisitPlanValidationError.ruleViolation(
                rule: "metric_twin_consistency_1mm",
                detail: "\(path): 2D coordinates must have exactly 2 elements"
            )
        }
        try checkScalarTwin(ft: ft[0], m: m[0], path: "\(path)[0]")
        try checkScalarTwin(ft: ft[1], m: m[1], path: "\(path)[1]")
    }

    private static func checkPathTwin(ft: [[Double]], m: [[Double]], path: String) throws {
        guard ft.count == m.count else {
            throw VisitPlanValidationError.ruleViolation(
                rule: "metric_twin_consistency_1mm",
                detail: "\(path): point count mismatch (\(ft.count) vs \(m.count))"
            )
        }
        for (idx, pair) in zip(ft, m).enumerated() {
            try checkPoint2DTwin(ft: pair.0, m: pair.1, path: "\(path)[\(idx)]")
        }
    }
}
