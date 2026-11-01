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

// MARK: - YS-85: Thermal Adaptation Policy

/// Thermal state representation decoupled from `ProcessInfo` for deterministic replay and testing (`YS-85`).
public enum CaptureThermalLevel: String, Sendable, Equatable, CaseIterable {
    case nominal = "nominal"
    case fair = "fair"
    case serious = "serious"
    case critical = "critical"

    public static func fromSystem(_ state: ProcessInfo.ThermalState) -> CaptureThermalLevel {
        switch state {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
    }

    public var isSeriousOrWorse: Bool {
        self == .serious || self == .critical
    }
}

/// Schedule computed by `ThermalAdaptationPolicy` (`YS-85`).
///
/// At `.serious` (or `.critical`) thermal state:
/// - ArUco detection drops from `10 Hz` (every 3rd frame at 30 fps) to `5 Hz` (every 6th frame)
///   when no marker is being held, and stays at `10 Hz` during an active marker hold.
/// - Ground coverage updates drop from `2.0 Hz` (`0.50 s` interval) to `0.5 Hz` (`2.00 s` interval).
/// - UI animations become static (`animationsStatic == true`).
/// - Recording itself (RGB and LiDAR depth at `30 Hz`) is NEVER reduced (`recordingFPS == 30`).
public struct ThermalAdaptationSchedule: Sendable, Equatable {
    public let thermalLevel: CaptureThermalLevel
    public let isMarkerHoldActive: Bool
    public let detectionHz: Double
    public let detectionFrameStrideAt30FPS: Int
    public let coverageHz: Double
    public let coverageMinIntervalS: Double
    public let animationsStatic: Bool
    public let recordingFPS: Int

    public static func evaluate(
        thermalLevel: CaptureThermalLevel,
        isMarkerHoldActive: Bool
    ) -> ThermalAdaptationSchedule {
        if thermalLevel.isSeriousOrWorse {
            let detHz = isMarkerHoldActive ? 10.0 : 5.0
            let stride = isMarkerHoldActive ? 3 : 6
            return ThermalAdaptationSchedule(
                thermalLevel: thermalLevel,
                isMarkerHoldActive: isMarkerHoldActive,
                detectionHz: detHz,
                detectionFrameStrideAt30FPS: stride,
                coverageHz: 0.5,
                coverageMinIntervalS: 2.0,
                animationsStatic: true,
                recordingFPS: 30
            )
        } else {
            return ThermalAdaptationSchedule(
                thermalLevel: thermalLevel,
                isMarkerHoldActive: isMarkerHoldActive,
                detectionHz: 10.0,
                detectionFrameStrideAt30FPS: 3,
                coverageHz: 2.0,
                coverageMinIntervalS: 0.5,
                animationsStatic: false,
                recordingFPS: 30
            )
        }
    }
}

// MARK: - YS-78: Live 0.25 m Ground Coverage Grid

/// Discrete 2D ground cell `(ix, iz)` on the horizontal `0.25 m` yard grid (`YS-78`).
public struct GroundGridCell: Hashable, Sendable, Codable {
    public let ix: Int
    public let iz: Int

    public init(ix: Int, iz: Int) {
        self.ix = ix
        self.iz = iz
    }
}

/// Live `0.25 m` ground coverage grid built from high-confidence LiDAR depth (`YS-78`).
///
/// Filters depth samples to:
/// - `range 0.5–4.0 m`,
/// - within `±0.25 m` of the marker/ground plane (`abs(heightAboveGroundM) <= 0.25`),
/// - local surface normal within `35°` of vertical (`cos(angleToUp) >= cos(35°)`),
/// - confidence `>= 1` (and tracks high-confidence `== 2` cells).
///
/// Updated at `2 Hz` (`0.5 Hz` under serious thermal state, `YS-85`) on the background detection queue.
public struct GroundCoverageGrid: Sendable, Equatable {
    public let cellSizeM: Double
    public let minRangeM: Double
    public let maxRangeM: Double
    public let maxAbsGroundHeightM: Double
    public let maxSurfaceTiltFromVerticalDeg: Double

    public private(set) var groundPlaneYWorld: Double? = nil
    public private(set) var isAnchoredToMarker: Bool = false
    public private(set) var originXZ: SIMD2<Double> = .zero
    public private(set) var uAxisXZ: SIMD2<Double> = SIMD2<Double>(1.0, 0.0)
    public private(set) var vAxisXZ: SIMD2<Double> = SIMD2<Double>(0.0, 1.0)
    public private(set) var coveredCells: Set<GroundGridCell> = []
    public private(set) var highConfidenceCells: Set<GroundGridCell> = []
    public private(set) var lastUpdatedTimestampS: Double = -100.0
    public private(set) var updateCount: Int = 0

    public init(
        cellSizeM: Double = 0.25,
        minRangeM: Double = 0.50,
        maxRangeM: Double = 4.00,
        maxAbsGroundHeightM: Double = 0.25,
        maxSurfaceTiltFromVerticalDeg: Double = 35.0
    ) {
        self.cellSizeM = cellSizeM
        self.minRangeM = minRangeM
        self.maxRangeM = maxRangeM
        self.maxAbsGroundHeightM = maxAbsGroundHeightM
        self.maxSurfaceTiltFromVerticalDeg = maxSurfaceTiltFromVerticalDeg
    }

    public var coveredCellCount: Int {
        coveredCells.count
    }

    public var coveredAreaM2: Double {
        Double(coveredCells.count) * cellSizeM * cellSizeM
    }

    /// Anchor the ground plane height to a measured floor marker (`YS-78`).
    /// On the first marker sighting, snaps directly to `yWorld` (replacing any startup depth heuristic)
    /// and purges any unanchored cells if the heuristic height differed by `> 0.10 m`.
    public mutating func anchorGroundPlaneY(_ yWorld: Double) {
        if !isAnchoredToMarker {
            if let unanchoredY = groundPlaneYWorld, abs(unanchoredY - yWorld) > 0.10 {
                coveredCells.removeAll()
                highConfidenceCells.removeAll()
            }
            groundPlaneYWorld = yWorld
            isAnchoredToMarker = true
        } else if let existing = groundPlaneYWorld {
            groundPlaneYWorld = 0.8 * existing + 0.2 * yWorld
        } else {
            groundPlaneYWorld = yWorld
            isAnchoredToMarker = true
        }
    }

    /// Anchor the ground coverage grid to Marker 0's horizontal origin and yaw (`YS-78`) so cell indices `(ix, iz)`
    /// align with the Marker-0 Yard Frame used by `coverage.json`.
    public mutating func anchorMarker0Frame(
        originWorld: SIMD3<Double>,
        xAxisWorld: SIMD3<Double> = SIMD3<Double>(1.0, 0.0, 0.0)
    ) {
        anchorGroundPlaneY(originWorld.y)
        let newOriginXZ = SIMD2<Double>(originWorld.x, originWorld.z)
        let rawU = SIMD2<Double>(xAxisWorld.x, xAxisWorld.z)
        let lenU = simd_length(rawU)
        let newU = lenU > 1e-6 ? (rawU / lenU) : SIMD2<Double>(1.0, 0.0)
        // Right-handed horizontal frame where world +Y is up -> second horizontal axis is (-u.z, u.x) or (u.z, -u.x)
        // To match default identity (u=(1,0) -> v=(0,1)), v = (-u.y, u.x)
        let newV = SIMD2<Double>(-newU.y, newU.x)
        if simd_length(newOriginXZ - originXZ) > 0.05 || simd_dot(newU, uAxisXZ) < 0.995 {
            coveredCells.removeAll()
            highConfidenceCells.removeAll()
        }
        originXZ = newOriginXZ
        uAxisXZ = newU
        vAxisXZ = newV
    }

    /// Map a 3D world point onto the Marker-0 anchored `0.25 m` horizontal grid (`YS-78`).
    public func cell(forWorldPoint pt: SIMD3<Double>) -> GroundGridCell {
        let relXZ = SIMD2<Double>(pt.x - originXZ.x, pt.z - originXZ.y)
        let localX = simd_dot(relXZ, uAxisXZ)
        let localZ = simd_dot(relXZ, vAxisXZ)
        let ix = Int(floor(localX / cellSizeM))
        let iz = Int(floor(localZ / cellSizeM))
        return GroundGridCell(ix: ix, iz: iz)
    }

    /// Check whether enough time has elapsed for the next coverage update (`2 Hz` normal, `0.5 Hz` serious thermal).
    public func shouldUpdate(at timestampS: Double, minIntervalS: Double) -> Bool {
        (timestampS - lastUpdatedTimestampS) >= (minIntervalS - 1e-6)
    }

    /// Integrate a LiDAR depth & confidence map into the `0.25 m` ground grid (`YS-78`) and also return
    /// look-up telemetry (`validDepthSampleCount`, `pointsAbove2mCount`) for `YS-79`.
    @discardableResult
    public mutating func ingestDepthFrame(
        timestampS: Double,
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
        sampleStep: Int = 4
    ) -> (addedCells: Int, validDepthSamples: Int, elevatedPointsAbove2m: Int) {
        guard depthMapMM.count == depthWidth * depthHeight,
              confidenceMap.count == depthWidth * depthHeight,
              depthWidth > sampleStep,
              depthHeight > sampleStep,
              rgbWidth > 0,
              rgbHeight > 0,
              fx > 0,
              fy > 0
        else {
            return (0, 0, 0)
        }

        lastUpdatedTimestampS = timestampS
        updateCount += 1

        let scaleX = Double(depthWidth) / Double(rgbWidth)
        let scaleY = Double(depthHeight) / Double(rgbHeight)
        let fxD = fx * scaleX
        let fyD = fy * scaleY
        let cxD = cx * scaleX
        let cyD = cy * scaleY
        let cosMaxTilt = cos(maxSurfaceTiltFromVerticalDeg * .pi / 180.0)

        let camPosW = SIMD3<Double>(
            cameraToWorld.columns.3.x,
            cameraToWorld.columns.3.y,
            cameraToWorld.columns.3.z
        )

        // Helper to unproject a depth pixel (u, v) to ARKit world coordinates
        func worldPoint(u: Int, v: Int) -> (pt: SIMD3<Double>, rangeM: Double, conf: UInt8)? {
            let idx = v * depthWidth + u
            let mm = depthMapMM[idx]
            let conf = confidenceMap[idx]
            guard mm > 0, conf >= 1 else { return nil }
            let z = Double(mm) * 0.001
            let xCam = (Double(u) + 0.5 - cxD) / fxD * z
            let yCam = -(Double(v) + 0.5 - cyD) / fyD * z
            let zCam = -z
            let rng = sqrt(xCam * xCam + yCam * yCam + zCam * zCam)
            let pW4 = cameraToWorld * SIMD4<Double>(xCam, yCam, zCam, 1.0)
            return (SIMD3<Double>(pW4.x, pW4.y, pW4.z), rng, conf)
        }

        // If groundPlaneYWorld is not yet anchored by a marker, estimate it from the lowest 15th percentile
        // of near-horizontal points below the camera when looking downward.
        if groundPlaneYWorld == nil {
            var candidateYs: [Double] = []
            var v = sampleStep
            while v < depthHeight - sampleStep {
                var u = sampleStep
                while u < depthWidth - sampleStep {
                    if let c = worldPoint(u: u, v: v),
                       c.rangeM >= minRangeM,
                       c.rangeM <= maxRangeM,
                       c.pt.y < camPosW.y - 0.50 {
                        candidateYs.append(c.pt.y)
                    }
                    u += sampleStep * 2
                }
                v += sampleStep * 2
            }
            if candidateYs.count >= 12 {
                candidateYs.sort()
                groundPlaneYWorld = candidateYs[candidateYs.count / 6]
            }
        }

        let refGroundY = groundPlaneYWorld ?? (camPosW.y - 1.30)
        let beforeCount = coveredCells.count
        var validDepthSamples = 0
        var elevatedPointsAbove2m = 0

        var v = sampleStep
        while v < depthHeight - sampleStep {
            var u = sampleStep
            while u < depthWidth - sampleStep {
                if let center = worldPoint(u: u, v: v) {
                    if center.rangeM >= minRangeM && center.rangeM <= 4.50 {
                        validDepthSamples += 1
                        if (center.pt.y - refGroundY) > 2.00 {
                            elevatedPointsAbove2m += 1
                        }
                    }

                    if center.rangeM >= minRangeM,
                       center.rangeM <= maxRangeM,
                       abs(center.pt.y - refGroundY) <= maxAbsGroundHeightM,
                       let right = worldPoint(u: u + sampleStep, v: v),
                       let down = worldPoint(u: u, v: v + sampleStep) {
                        let dU = right.pt - center.pt
                        let dV = down.pt - center.pt
                        let crossVec = simd_cross(dU, dV)
                        let len = simd_length(crossVec)
                        if len > 1e-6 {
                            let normalUpAbs = abs(crossVec.y / len)
                            if normalUpAbs >= cosMaxTilt {
                                let cell = cell(forWorldPoint: center.pt)
                                coveredCells.insert(cell)
                                if center.conf >= 2 {
                                    highConfidenceCells.insert(cell)
                                }
                            }
                        }
                    }
                }
                u += sampleStep
            }
            v += sampleStep
        }

        return (coveredCells.count - beforeCount, validDepthSamples, elevatedPointsAbove2m)
    }

    /// Transform world-frame ground points into Marker-0 Yard Frame `0.25 m` cells and compute
    /// the cell agreement ratio against a reference set of yard-frame ground points (`YS-78`).
    public static func computeCellAgreementWithToolkitGroundPoints(
        worldGroundPoints: [(positionWorld: SIMD3<Double>, confidence: UInt8, rangeM: Double)],
        groundYWorld: Double,
        originWorld: SIMD3<Double> = .zero,
        xAxisWorld: SIMD3<Double> = SIMD3<Double>(1.0, 0.0, 0.0),
        ingestedGrid: GroundCoverageGrid? = nil,
        cellSizeM: Double = 0.25,
        maxAbsZ: Double = 0.25,
        minRangeM: Double = 0.50,
        maxRangeM: Double = 4.00
    ) -> (phoneCells: Set<GroundGridCell>, toolkitCells: Set<GroundGridCell>, agreementRatio: Double) {
        var refGrid = GroundCoverageGrid(cellSizeM: cellSizeM)
        refGrid.anchorMarker0Frame(originWorld: SIMD3<Double>(originWorld.x, groundYWorld, originWorld.z), xAxisWorld: xAxisWorld)

        var phoneSet: Set<GroundGridCell> = ingestedGrid?.coveredCells ?? []
        let useIngested = (ingestedGrid != nil)
        var toolkitSet: Set<GroundGridCell> = []

        for sample in worldGroundPoints {
            guard sample.confidence >= 1 else { continue }
            let dh = abs(sample.positionWorld.y - groundYWorld)
            let cell = refGrid.cell(forWorldPoint: sample.positionWorld)

            // Phone rule (YS-78): 0.5–4.0 m, within ±0.25 m of marker plane
            if !useIngested && sample.rangeM >= minRangeM && sample.rangeM <= maxRangeM && dh <= maxAbsZ {
                phoneSet.insert(cell)
            }
            // Toolkit rule (geometry.py:876): range <= 4.0 m, within ±0.25 m of marker plane in Marker-0 Yard Frame
            if sample.rangeM <= maxRangeM && dh <= maxAbsZ {
                toolkitSet.insert(cell)
            }
        }

        guard !toolkitSet.isEmpty else {
            return (phoneSet, toolkitSet, 1.0)
        }
        let intersectionCount = phoneSet.intersection(toolkitSet).count
        let unionCount = max(1, phoneSet.union(toolkitSet).count)
        return (phoneSet, toolkitSet, Double(intersectionCount) / Double(unionCount))
    }
}

// MARK: - YS-79: Look-Up & House-Wall Step

/// Tracks the post-markers-1–3 house-wall and roof-edge look-up step (`YS-79`).
///
/// Coaches `"Look up at the roof edge and the trees"` for at least `3.0 s` cumulative at a tilt of
/// `+15°` to `+35°` with valid depth. Above `+45°` with no depth, coaches `"Point at the roof edge, not the sky"`.
/// Provides a one-tap `Skip` recorded in `guidance.csv`.
public struct HouseWallLookUpTracker: Sendable, Equatable {
    public static let minTargetPitchDeg: Double = 15.0
    public static let maxTargetPitchDeg: Double = 35.0
    public static let skyWarningPitchDeg: Double = 45.0
    public static let requiredDurationS: Double = 3.0

    public private(set) var accumulatedValidDurationS: Double = 0.0
    public private(set) var isCompleted: Bool = false
    public private(set) var isSkipped: Bool = false
    public private(set) var elevatedPointsAbove2mCount: Int = 0
    public private(set) var insufficientFeaturesLossCount: Int = 0
    public private(set) var skyWarningActive: Bool = false

    public init() {}

    public var isDone: Bool {
        isCompleted || isSkipped
    }

    public var progressFraction: Double {
        if isDone { return 1.0 }
        return min(1.0, max(0.0, accumulatedValidDurationS / Self.requiredDurationS))
    }

    /// Update the look-up step for a frame with `dt`, `pitchDeg`, `hasValidDepth`, `elevatedPointsAbove2m`, and `trackingReason`.
    public mutating func update(
        dt: Double,
        pitchDeg: Double,
        hasValidDepth: Bool,
        elevatedPointsAbove2m: Int,
        trackingState: String,
        trackingReason: String
    ) {
        elevatedPointsAbove2mCount += max(0, elevatedPointsAbove2m)
        if trackingState == "limited" && trackingReason == "insufficient_features" {
            insufficientFeaturesLossCount += 1
        }
        guard !isDone else {
            skyWarningActive = false
            return
        }

        if pitchDeg > Self.skyWarningPitchDeg && !hasValidDepth {
            skyWarningActive = true
            return
        }
        skyWarningActive = false

        if trackingState == "normal",
           hasValidDepth,
           pitchDeg >= Self.minTargetPitchDeg,
           pitchDeg <= Self.maxTargetPitchDeg,
           dt > 0.0 {
            accumulatedValidDurationS += min(dt, 0.25)
            if accumulatedValidDurationS >= Self.requiredDurationS && elevatedPointsAbove2mCount > 0 {
                isCompleted = true
            }
        }
    }

    public mutating func skip() {
        guard !isCompleted else { return }
        isSkipped = true
        skyWarningActive = false
    }
}

// MARK: - YS-74: Marker Lock Ring & Deterministic Guidance Engine

/// System haptic pattern emitted by `GuidedCaptureEngine` (`YS-74`, `DESIGN-SPEC-NATIVE.md` §5).
///
/// Corrections NEVER emit a haptic, and all haptics are suppressed while a voice note is recording.
public enum GuidedHapticEvent: Sendable, Equatable {
    case sessionStart
    case sessionStop
    case markerFirstSeen(markerID: Int)
    /// Confirmed photo during a hold (`segment` in `1...3`, intensity `0.5`, `0.7`, `0.9`).
    case holdSegmentConfirmed(markerID: Int, segment: Int, intensity: Double)
    case markerLocked(markerID: Int)
    case loopClosedPathComplete
    case warning
    case error
}

/// Visual state of the 3-segment lock ring drawn on a marker in the camera view (`YS-74`, `DESIGN-SPEC-NATIVE.md` §3a).
public struct MarkerRingVisualState: Sendable, Equatable {
    public let markerID: Int
    public let isClosingHold: Bool
    public let confirmedSegments: Int
    public let requiredSegments: Int
    public let isPaused: Bool
    public let pauseReason: String?
    public let showStepCloserChevron: Bool
    public let isLocked: Bool
    public let positionWorld: SIMD3<Double>
    public let cornersUV: [SIMD2<Double>]
    public let rangeMeters: Double

    public init(
        markerID: Int,
        isClosingHold: Bool,
        confirmedSegments: Int,
        requiredSegments: Int = 3,
        isPaused: Bool,
        pauseReason: String?,
        showStepCloserChevron: Bool,
        isLocked: Bool,
        positionWorld: SIMD3<Double>,
        cornersUV: [SIMD2<Double>],
        rangeMeters: Double
    ) {
        self.markerID = markerID
        self.isClosingHold = isClosingHold
        self.confirmedSegments = confirmedSegments
        self.requiredSegments = requiredSegments
        self.isPaused = isPaused
        self.pauseReason = pauseReason
        self.showStepCloserChevron = showStepCloserChevron
        self.isLocked = isLocked
        self.positionWorld = positionWorld
        self.cornersUV = cornersUV
        self.rangeMeters = rangeMeters
    }

    /// Computes normalized portrait HUD coordinates `(x, y)` in `[0.1, 0.9]` from sensor pixel coordinates `(u, v)`.
    /// ARKit `capturedImage` sensor buffers are landscape-right (`1920x1440`), while the HUD runs in portrait:
    /// `portraitX = 1.0 - (v / rgbHeight)`, `portraitY = u / rgbWidth`.
    public func normalizedPortraitCenter(rgbWidth: Double = 1920.0, rgbHeight: Double = 1440.0) -> SIMD2<Double> {
        guard !cornersUV.isEmpty else {
            return SIMD2<Double>(0.5, 0.5)
        }
        let sumU = cornersUV.reduce(0.0) { $0 + $1.x }
        let sumV = cornersUV.reduce(0.0) { $0 + $1.y }
        let avgU = sumU / Double(cornersUV.count)
        let avgV = sumV / Double(cornersUV.count)
        let safeW = max(1.0, rgbWidth)
        let safeH = max(1.0, rgbHeight)
        if avgU > 1.5 || avgV > 1.5 {
            let portraitX = min(0.9, max(0.1, 1.0 - (avgV / safeH)))
            let portraitY = min(0.9, max(0.1, avgU / safeW))
            return SIMD2<Double>(portraitX, portraitY)
        }
        return SIMD2<Double>(min(0.9, max(0.1, avgU)), min(0.9, max(0.1, avgV)))
    }
}

/// High-level phase of the guided yard walk (`YS-74`, `YS-79`).
public enum GuidedWalkPhase: String, Sendable, Equatable {
    case openingMarker0 = "opening_marker_0"
    case markers1To3 = "markers_1_to_3"
    case houseWallLookUp = "house_wall_lookup"
    case closingMarker0 = "closing_marker_0"
    case readyToFinish = "ready_to_finish"
}

/// Confirmed marker still sample used by `GuidedCaptureEngine` to evaluate the 3-still lock rule (`YS-74`).
public struct ConfirmedMarkerStillSample: Sendable, Equatable {
    public let timestampS: Double
    public let savedIndex: Int
    public let markerID: Int
    public let positionWorld: SIMD3<Double>
    public let sizeMMPrinted: Double
    public let sizeMMMeasured: Double
    public let viewingAngleDeg: Double
    public let rangeMeters: Double
    public let cornersUV: [SIMD2<Double>]

    public init(
        timestampS: Double,
        savedIndex: Int,
        sighting: DetectedMarkerSighting
    ) {
        self.timestampS = timestampS
        self.savedIndex = savedIndex
        self.markerID = sighting.markerID
        self.positionWorld = sighting.positionWorld
        self.sizeMMPrinted = sighting.sizeMMPrinted
        self.sizeMMMeasured = sighting.sizeMMMeasured
        self.viewingAngleDeg = sighting.viewingAngleDeg
        self.rangeMeters = sighting.rangeMeters
        self.cornersUV = sighting.cornersUV
    }
}

/// Per-marker hold accumulator that pauses rather than resets when a condition breaks (`YS-74`).
public struct MarkerHoldProgress: Sendable, Equatable {
    public let markerID: Int
    public let isClosingHold: Bool
    public var firstSeenTimestampS: Double? = nil
    public var lastSeenTimestampS: Double? = nil
    public var confirmedSamples: [ConfirmedMarkerStillSample] = []
    public var isLocked: Bool = false
    public var lockedAtTimestampS: Double? = nil
    public var isPaused: Bool = false
    public var pauseReason: String? = nil
    public var latestPositionWorld: SIMD3<Double> = .zero
    public var latestCornersUV: [SIMD2<Double>] = []
    public var latestRangeMeters: Double = 1.0

    public init(markerID: Int, isClosingHold: Bool = false) {
        self.markerID = markerID
        self.isClosingHold = isClosingHold
    }

    public var confirmedSegments: Int {
        if isLocked { return 3 }
        return min(3, confirmedSamples.count)
    }

    /// Compute the maximum 3D position spread across `samples` (maximum pairwise 3D distance and distance from median, `YS-74`).
    public static func maxSpreadFromMedianM(_ samples: [ConfirmedMarkerStillSample]) -> Double {
        guard samples.count >= 2 else { return 0.0 }
        let med = PhoneLoopRuleState.medianPosition3D(samples.map(\.positionWorld))
        var maxDist = 0.0
        for i in 0..<samples.count {
            let dMed = simd_length(samples[i].positionWorld - med)
            if dMed > maxDist { maxDist = dMed }
            for j in (i + 1)..<samples.count {
                let dPair = simd_length(samples[i].positionWorld - samples[j].positionWorld)
                if dPair > maxDist { maxDist = dPair }
            }
        }
        return maxDist
    }
}

/// Pure, deterministic state machine for guided capture coaching, 3-still marker locks,
/// house-wall look-up (`YS-79`), and thermal rate adaptation (`YS-74`, `YS-85`).
public struct GuidedCaptureEngine: Sendable {
    public static let debounceDurationS: Double = 0.70
    public static let minInstructionDwellS: Double = 1.50
    public static let minHoldSpanS: Double = 1.00
    public static let maxViewingAngleDeg: Double = 40.0
    public static let maxPositionSpreadM: Double = 0.025
    public static let maxSizeDeviationRatio: Double = 0.03
    public static let pitchOutOfBandTriggerS: Double = 3.00

    public var requireLookUpStep: Bool
    public var isVoiceCoachEnabled: Bool = true
    public var preWalkTapeMeters0To1: Double? = nil {
        didSet {
            recomputeMarker0To1TapeCheckIfPossible()
        }
    }
    public private(set) var generatedMarker0To1TapeCheck: PreWalkTapeCheckEntry? = nil
    public private(set) var phase: GuidedWalkPhase = .openingMarker0
    public private(set) var activeInstruction: String = "Stand over marker 0 and point at it"
    public private(set) var activeInstructionKey: String = "origin_find_marker_0"
    public private(set) var activeInstructionSinceS: Double = -100.0
    public private(set) var instructionHistory: [(timestampS: Double, key: String, text: String)] = []
    public private(set) var spokenHistory: [(timestampS: Double, key: String, glassText: String, spokenPhrase: String)] = []
    public private(set) var lastSpokenPhraseThisStep: String? = nil

    public private(set) var openingMarker0Hold = MarkerHoldProgress(markerID: 0, isClosingHold: false)
    public private(set) var marker1To3Holds: [Int: MarkerHoldProgress] = [
        1: MarkerHoldProgress(markerID: 1),
        2: MarkerHoldProgress(markerID: 2),
        3: MarkerHoldProgress(markerID: 3),
    ]
    public private(set) var closingMarker0Hold = MarkerHoldProgress(markerID: 0, isClosingHold: true)

    public private(set) var lockedMarkerIDs: Set<Int> = []
    public private(set) var lockedPositionsByMarkerID: [Int: SIMD3<Double>] = [:]
    public private(set) var isClosingMarker0Locked: Bool = false
    public private(set) var endEarlyChosen: Bool = false
    public private(set) var fixNowRequiresReturnToMarker0: Bool = false
    public private(set) var isRecordingVoiceNote: Bool = false

    public private(set) var lookUpTracker = HouseWallLookUpTracker()
    public private(set) var activeRingState: MarkerRingVisualState? = nil
    public private(set) var latestThermalSchedule = ThermalAdaptationSchedule.evaluate(
        thermalLevel: .nominal,
        isMarkerHoldActive: false
    )

    // Debounce, dwell & voice coach state
    private var activeInstructionIsCorrection: Bool = false
    private var pendingCandidateKey: String? = nil
    private var pendingCandidateText: String? = nil
    private var pendingCandidateSinceS: Double = 0.0
    private var pitchTooHighDurationS: Double = 0.0
    private var pitchTooLowDurationS: Double = 0.0
    private var warnedThermalSerious: Bool = false
    private var warnedLowStorage: Bool = false
    private var justLockedBannerUntilS: Double = -100.0
    private var justLockedKey: String = ""
    private var justLockedText: String = ""
    private var lastActiveHoldMarkerID: Int? = nil
    private var lastRingSightingTimestampS: Double = -100.0
    private var holdPausedAccumulatedS: Double = 0.0
    private var didSpeakHoldPausedMovingThisPause: Bool = false
    private var spokenOnceKeys: Set<String> = []

    public init(requireLookUpStep: Bool = true, preWalkTapeMeters0To1: Double? = nil) {
        self.requireLookUpStep = requireLookUpStep
        self.preWalkTapeMeters0To1 = preWalkTapeMeters0To1
    }

    /// Recompute the automatic centre-to-centre `tape_checks` entry between Marker 0 and Marker 1 (`YS-77`)
    /// once both Marker 0 and Marker 1 are locked and a pre-walk tape distance `> 0` was entered.
    public mutating func recomputeMarker0To1TapeCheckIfPossible() {
        guard let tapeM = preWalkTapeMeters0To1, tapeM > 0,
              openingMarker0Hold.isLocked,
              !openingMarker0Hold.confirmedSamples.isEmpty,
              let hold1 = marker1To3Holds[1],
              hold1.isLocked,
              !hold1.confirmedSamples.isEmpty
        else {
            return
        }
        let samples0 = openingMarker0Hold.confirmedSamples
        let samples1 = hold1.confirmedSamples
        let ptA = PhoneLoopRuleState.medianPosition3D(samples0.map(\.positionWorld))
        let ptB = PhoneLoopRuleState.medianPosition3D(samples1.map(\.positionWorld))
        let last0 = samples0[samples0.count - 1]
        let last1 = samples1[samples1.count - 1]
        generatedMarker0To1TapeCheck = PreWalkTapeCheckEntry(
            id: 0,
            role: "primary",
            tAS: last0.timestampS,
            tBS: last1.timestampS,
            savedIndexA: last0.savedIndex,
            savedIndexB: last1.savedIndex,
            pointAWorldM: ptA,
            pointBWorldM: ptB,
            tapeM: tapeM
        )
    }

    /// True only when `YS-71` confirms the loop is closed AND Marker 0's closing hold is locked
    /// after Markers 1–3 (or End Early) and after any `"Fix now"` (`YS-72`, `YS-74`).
    public func isFinishAvailable(unifiedLoopClosed: Bool) -> Bool {
        let markersReady = endEarlyChosen || (lockedMarkerIDs.contains(1) && lockedMarkerIDs.contains(2) && lockedMarkerIDs.contains(3))
        return unifiedLoopClosed
            && lockedMarkerIDs.contains(0)
            && markersReady
            && isClosingMarker0Locked
            && !fixNowRequiresReturnToMarker0
    }

    public mutating func setRecordingVoiceNote(_ active: Bool, at timestampS: Double) {
        isRecordingVoiceNote = active
        if active {
            commitInstruction(
                key: "note_recording",
                text: "Listening… tap to stop",
                isCorrection: false,
                at: timestampS,
                forceImmediate: true
            )
        }
    }

    public mutating func markEndEarlyChosen(at timestampS: Double) {
        endEarlyChosen = true
        if phase == .markers1To3 || phase == .houseWallLookUp {
            phase = .closingMarker0
            commitInstruction(
                key: "return_marker_0",
                text: "Head back to marker 0",
                isCorrection: false,
                at: timestampS,
                forceImmediate: true
            )
        }
    }

    /// Require a fresh closing hold at Marker 0 after any `"Fix now"` action from Walk Review (`YS-72`, `YS-74`).
    /// If `targetMarkerID` is in `1...3`, also unlocks that marker first so the operator re-locks it before returning to Marker 0.
    public mutating func beginFixNowRequiringReturnToMarker0(targetMarkerID: Int? = nil, at timestampS: Double) {
        fixNowRequiresReturnToMarker0 = true
        isClosingMarker0Locked = false
        closingMarker0Hold = MarkerHoldProgress(markerID: 0, isClosingHold: true)
        if let mid = targetMarkerID, (1...3).contains(mid) {
            lockedMarkerIDs.remove(mid)
            marker1To3Holds[mid] = MarkerHoldProgress(markerID: mid)
            phase = .markers1To3
            commitInstruction(
                key: "walk_to_marker_\(mid)",
                text: "Walk to marker \(mid)",
                isCorrection: false,
                at: timestampS,
                forceImmediate: true
            )
        } else {
            phase = .closingMarker0
            commitInstruction(
                key: "return_marker_0",
                text: "Head back to marker 0",
                isCorrection: false,
                at: timestampS,
                forceImmediate: true
            )
        }
    }

    public mutating func skipLookUpStep(at timestampS: Double) {
        lookUpTracker.skip()
        if phase == .houseWallLookUp {
            phase = .closingMarker0
            commitInstruction(
                key: "return_marker_0",
                text: "Head back to marker 0",
                isCorrection: false,
                at: timestampS,
                forceImmediate: true
            )
        }
    }

    /// Process a single frame and return the active instruction, optional ring visual state, optional haptic event,
    /// and thermal adaptation schedule (`YS-74`, `YS-79`, `YS-85`).
    @discardableResult
    public mutating func step(
        timestampS: Double,
        savedIndex: Int,
        dt: Double,
        trackingState: String,
        trackingReason: String = "",
        pitchDeg: Double,
        turnRateRadS: Double,
        predictedBlurPx: Double,
        cameraPositionWorld: SIMD3<Double>,
        confirmedSightings: [DetectedMarkerSighting],
        rawSightings: [DetectedMarkerSighting] = [],
        unifiedLoopClosed: Bool,
        thermalLevel: CaptureThermalLevel = .nominal,
        lowDiskSpaceWarning: Bool = false,
        hasValidDepthForLookUp: Bool = true,
        elevatedPointsAbove2m: Int = 0,
        didRunMarkerDetection: Bool = true
    ) -> (
        instruction: String,
        instructionKey: String,
        ringState: MarkerRingVisualState?,
        haptic: GuidedHapticEvent?,
        thermalSchedule: ThermalAdaptationSchedule
    ) {
        let wasInstructionHistoryEmpty = instructionHistory.isEmpty
        if wasInstructionHistoryEmpty {
            activeInstructionSinceS = timestampS
            instructionHistory.append((timestampS, activeInstructionKey, activeInstruction))
        }
        lastSpokenPhraseThisStep = nil

        let phaseBeforeStep = phase
        let keyBeforeStep = activeInstructionKey
        var didLockMarkerThisStep = false
        var didStartHoldThisStep = false
        var emittedHaptic: GuidedHapticEvent? = nil
        var activeHoldMarkerID: Int? = nil
        let holdGraceActive = !didRunMarkerDetection && ((timestampS - lastRingSightingTimestampS) <= 0.45)
        if !holdGraceActive {
            activeRingState = nil
            lastActiveHoldMarkerID = nil
        } else {
            activeHoldMarkerID = lastActiveHoldMarkerID
        }

        // 1. Determine which marker hold (if any) is relevant in the current phase
        let allVisible = !confirmedSightings.isEmpty ? confirmedSightings : rawSightings
        let targetSighting: (sighting: DetectedMarkerSighting, isConfirmedStill: Bool, isClosing: Bool)? = {
            switch phase {
            case .openingMarker0:
                if let c0 = confirmedSightings.first(where: { $0.markerID == 0 }) {
                    return (c0, true, false)
                }
                if let r0 = allVisible.first(where: { $0.markerID == 0 }) {
                    return (r0, false, false)
                }
                return nil

            case .markers1To3:
                // Any unlocked marker in 1...3
                for s in confirmedSightings where (1...3).contains(s.markerID) && !lockedMarkerIDs.contains(s.markerID) {
                    return (s, true, false)
                }
                for s in allVisible where (1...3).contains(s.markerID) && !lockedMarkerIDs.contains(s.markerID) {
                    return (s, false, false)
                }
                return nil

            case .houseWallLookUp:
                // If the user walks straight back to Marker 0 during look-up, allow transitioning to closing hold
                if let c0 = confirmedSightings.first(where: { $0.markerID == 0 }) {
                    return (c0, true, true)
                }
                return nil

            case .closingMarker0, .readyToFinish:
                if !isClosingMarker0Locked {
                    if let c0 = confirmedSightings.first(where: { $0.markerID == 0 }) {
                        return (c0, true, true)
                    }
                    if let r0 = allVisible.first(where: { $0.markerID == 0 }) {
                        return (r0, false, true)
                    }
                }
                return nil
            }
        }()

        // 2. Update active marker hold (if a target marker is in view)
        if let target = targetSighting {
            let mid = target.sighting.markerID
            activeHoldMarkerID = mid
            if phase == .houseWallLookUp && target.isClosing {
                phase = .closingMarker0
            }

            var hold: MarkerHoldProgress = {
                if target.isClosing { return closingMarker0Hold }
                if mid == 0 { return openingMarker0Hold }
                return marker1To3Holds[mid] ?? MarkerHoldProgress(markerID: mid)
            }()

            let wasFirstSeen = (hold.firstSeenTimestampS == nil)
            if wasFirstSeen {
                hold.firstSeenTimestampS = timestampS
                didStartHoldThisStep = true
                emittedHaptic = .markerFirstSeen(markerID: mid)
            }
            hold.lastSeenTimestampS = timestampS
            hold.latestPositionWorld = target.sighting.positionWorld
            hold.latestCornersUV = target.sighting.cornersUV
            hold.latestRangeMeters = target.sighting.rangeMeters

            let rngXZ = hypot(
                cameraPositionWorld.x - target.sighting.positionWorld.x,
                cameraPositionWorld.z - target.sighting.positionWorld.z
            )
            let gateVerdict = ExposureAndStillnessGate.evaluateMarkerSighting(
                predictedBlurPx: predictedBlurPx,
                turnRateRadS: turnRateRadS,
                rangeMeters3D: target.sighting.rangeMeters,
                horizontalRangeMeters: rngXZ
            )
            let maxAngle = Self.maxViewingAngleDeg
            let maxSizeDev = Self.maxSizeDeviationRatio
            let maxSpread = Self.maxPositionSpreadM

            let angleOK = target.sighting.viewingAngleDeg <= maxAngle
            let sizeOK = target.sighting.sizeMMPrinted > 0
                ? (abs(target.sighting.sizeMMMeasured - target.sighting.sizeMMPrinted) / target.sighting.sizeMMPrinted) <= maxSizeDev
                : true
            let conditionsOK = (trackingState == "normal") && gateVerdict.passes && angleOK && sizeOK

            if conditionsOK && target.isConfirmedStill {
                let sample = ConfirmedMarkerStillSample(
                    timestampS: timestampS,
                    savedIndex: savedIndex,
                    sighting: target.sighting
                )

                var candidateSamples = hold.confirmedSamples
                let canAppend: Bool = {
                    guard let last = candidateSamples.last, let first = candidateSamples.first else {
                        return true
                    }
                    let dtLast = timestampS - last.timestampS
                    if candidateSamples.count == 1 {
                        return dtLast >= 0.30
                    } else if candidateSamples.count == 2 {
                        return dtLast >= 0.30 && (timestampS - first.timestampS) >= (Self.minHoldSpanS - 1e-6)
                    }
                    return false
                }()

                if canAppend {
                    candidateSamples.append(sample)
                    if MarkerHoldProgress.maxSpreadFromMedianM(candidateSamples) <= maxSpread {
                        hold.confirmedSamples = candidateSamples
                        hold.isPaused = false
                        hold.pauseReason = nil
                        let seg = hold.confirmedSegments
                        let intensities = [1: 0.5, 2: 0.7, 3: 0.9]
                        emittedHaptic = .holdSegmentConfirmed(
                            markerID: mid,
                            segment: seg,
                            intensity: intensities[seg] ?? 0.7
                        )

                        if hold.confirmedSamples.count >= 3,
                           let firstS = hold.confirmedSamples.first,
                           let lastS = hold.confirmedSamples.last,
                           (lastS.timestampS - firstS.timestampS) >= (Self.minHoldSpanS - 1e-6) {
                            hold.isLocked = true
                            hold.lockedAtTimestampS = timestampS
                            didLockMarkerThisStep = true
                            let lockedPos = PhoneLoopRuleState.medianPosition3D(hold.confirmedSamples.map(\.positionWorld))
                            lockedPositionsByMarkerID[mid] = lockedPos

                            if target.isClosing {
                                isClosingMarker0Locked = true
                                fixNowRequiresReturnToMarker0 = false
                                emittedHaptic = .loopClosedPathComplete
                            } else {
                                lockedMarkerIDs.insert(mid)
                                emittedHaptic = .markerLocked(markerID: mid)
                                if mid == 0 {
                                    justLockedKey = "locked_marker_0_opening"
                                    justLockedText = "Got marker 0. Every marker works like that."
                                } else {
                                    justLockedKey = "locked_marker_\(mid)"
                                    justLockedText = "Got marker \(mid)"
                                }
                                justLockedBannerUntilS = timestampS + Self.minInstructionDwellS
                            }
                        }
                    } else {
                        // Position spread > 2.5 cm: pause without resetting to 0 (slide window to keep latest consistent samples)
                        while candidateSamples.count > 1 && MarkerHoldProgress.maxSpreadFromMedianM(candidateSamples) > maxSpread {
                            candidateSamples.removeFirst()
                        }
                        hold.confirmedSamples = candidateSamples
                        hold.isPaused = true
                        hold.pauseReason = "moving"
                    }
                } else {
                    // If `candidateSamples.count == 2` and `dtLast >= 0.30` but `(timestampS - first.timestampS) < 1.00`,
                    // check if `first` was an approach outlier relative to `[last, sample]`. If so, slide out `first`.
                    if candidateSamples.count == 2,
                       let first = candidateSamples.first,
                       let last = candidateSamples.last,
                       (timestampS - last.timestampS) >= 0.30 {
                        if MarkerHoldProgress.maxSpreadFromMedianM([first, sample]) > maxSpread,
                           MarkerHoldProgress.maxSpreadFromMedianM([last, sample]) <= maxSpread {
                            hold.confirmedSamples = [last, sample]
                        }
                    }
                    hold.isPaused = false
                    hold.pauseReason = nil
                }
            } else if conditionsOK && !target.isConfirmedStill {
                // Intermediate 10 Hz detection frame between 0.35 s JPEG stills while all hold conditions remain valid:
                // keep the ring active and unpaused (`YS-74`).
                hold.isPaused = false
                hold.pauseReason = nil
            } else if !hold.isLocked {
                // Condition broke: pause rather than reset (`YS-74`)
                hold.isPaused = !hold.confirmedSamples.isEmpty
                if gateVerdict.reason == "marker_too_far" {
                    hold.pauseReason = "too_far"
                } else if gateVerdict.reason == "marker_too_close" {
                    hold.pauseReason = "too_close"
                } else if !gateVerdict.passes || !angleOK || trackingState != "normal" {
                    hold.pauseReason = "moving"
                } else {
                    hold.pauseReason = nil
                }
            }

            let showChevron = (hold.pauseReason == "too_far") || (target.sighting.rangeMeters > 1.50)
            activeRingState = MarkerRingVisualState(
                markerID: mid,
                isClosingHold: target.isClosing,
                confirmedSegments: hold.confirmedSegments,
                isPaused: hold.isPaused,
                pauseReason: hold.pauseReason,
                showStepCloserChevron: showChevron,
                isLocked: hold.isLocked,
                positionWorld: hold.latestPositionWorld,
                cornersUV: hold.latestCornersUV,
                rangeMeters: hold.latestRangeMeters
            )
            lastRingSightingTimestampS = timestampS
            lastActiveHoldMarkerID = hold.isLocked ? nil : mid
            activeHoldMarkerID = lastActiveHoldMarkerID

            if target.isClosing {
                closingMarker0Hold = hold
            } else if mid == 0 {
                openingMarker0Hold = hold
            } else {
                marker1To3Holds[mid] = hold
            }
            if didLockMarkerThisStep && (mid == 0 || mid == 1) {
                recomputeMarker0To1TapeCheckIfPossible()
            }
        }

        // 3. Advance phase state machine
        if phase == .openingMarker0 && lockedMarkerIDs.contains(0) {
            phase = .markers1To3
        }
        if phase == .markers1To3 {
            let all1To3Locked = lockedMarkerIDs.contains(1) && lockedMarkerIDs.contains(2) && lockedMarkerIDs.contains(3)
            if all1To3Locked || endEarlyChosen {
                if requireLookUpStep && !lookUpTracker.isDone && !endEarlyChosen {
                    phase = .houseWallLookUp
                } else {
                    phase = .closingMarker0
                }
            }
        }
        if phase == .houseWallLookUp {
            lookUpTracker.update(
                dt: dt,
                pitchDeg: pitchDeg,
                hasValidDepth: hasValidDepthForLookUp,
                elevatedPointsAbove2m: elevatedPointsAbove2m,
                trackingState: trackingState,
                trackingReason: trackingReason
            )
            if lookUpTracker.isDone {
                phase = .closingMarker0
            }
        }
        if phase == .closingMarker0 && isFinishAvailable(unifiedLoopClosed: unifiedLoopClosed) {
            phase = .readyToFinish
        } else if phase == .readyToFinish && !isFinishAvailable(unifiedLoopClosed: unifiedLoopClosed) {
            phase = .closingMarker0
        }

        // 4. Update pitch out-of-band timers (only outside look-up and active holds)
        let isHoldCounting = (activeRingState != nil && !(activeRingState?.isLocked ?? true))
        if phase != .houseWallLookUp && !isHoldCounting && dt > 0 {
            if pitchDeg > -18.0 {
                pitchTooHighDurationS += min(dt, 0.25)
                pitchTooLowDurationS = 0.0
            } else if pitchDeg < -42.0 {
                pitchTooLowDurationS += min(dt, 0.25)
                pitchTooHighDurationS = 0.0
            } else {
                pitchTooHighDurationS = max(0.0, pitchTooHighDurationS - dt)
                pitchTooLowDurationS = max(0.0, pitchTooLowDurationS - dt)
            }
        } else {
            pitchTooHighDurationS = 0.0
            pitchTooLowDurationS = 0.0
        }

        // 5. Select highest-priority candidate instruction (DESIGN-SPEC-NATIVE.md §4c)
        let (candKey, candText, isImmediatePhaseOrLockCue): (String, String, Bool) = {
            if isRecordingVoiceNote {
                return ("note_recording", "Listening… tap to stop", true)
            }
            if lowDiskSpaceWarning && !warnedLowStorage {
                warnedLowStorage = true
                if emittedHaptic == nil { emittedHaptic = .warning }
                return ("storage_low", "Almost out of space. Finish soon.", true)
            }
            if thermalLevel.isSeriousOrWorse && !warnedThermalSerious {
                warnedThermalSerious = true
                if emittedHaptic == nil { emittedHaptic = .warning }
                return ("thermal_serious", "iPhone is getting warm. Finish soon.", true)
            }
            if trackingState == "limited" {
                if trackingReason == "insufficient_features" {
                    return ("tracking_insufficient_features", "Point at something with more detail, like the fence", false)
                }
                if trackingReason == "excessive_motion" {
                    return ("tracking_excessive_motion", "Move the phone more slowly", false)
                }
            }
            if timestampS < justLockedBannerUntilS {
                return (justLockedKey, justLockedText, true)
            }
            if let ring = activeRingState, !ring.isLocked {
                if ring.pauseReason == "too_far" {
                    return ("hold_too_far_\(ring.markerID)", "Step closer to marker \(ring.markerID)", false)
                }
                if ring.pauseReason == "too_close" {
                    return ("hold_too_close_\(ring.markerID)", "Step back from marker \(ring.markerID)", false)
                }
                if ring.isPaused && ring.confirmedSegments > 0 {
                    return ("hold_paused_moving", "Hold steady… almost there", false)
                }
                return ("hold_steady", "Hold steady", true)
            }

            // Corrections are silent while a hold is counting; outside a hold, check pace & tilt (`YS-73`, `YS-74`)
            if turnRateRadS > ExposureAndStillnessGate.maxTurnRateRadS {
                return ("pace_turn_slower", "Turn a little slower", false)
            }
            if predictedBlurPx > ExposureAndStillnessGate.maxPredictedBlurPx {
                return ("pace_hold_steady", "Hold steady", false)
            }
            if phase == .houseWallLookUp {
                if lookUpTracker.skyWarningActive {
                    return ("lookup_sky_warning", "Point at the roof edge, not the sky", false)
                }
                return ("lookup_step", "Look up at the roof edge and the trees", true)
            }
            if pitchTooHighDurationS >= Self.pitchOutOfBandTriggerS {
                return ("tilt_point_ground", "Point at the ground ahead", false)
            }
            if pitchTooLowDurationS >= Self.pitchOutOfBandTriggerS {
                return ("tilt_lift_phone", "Lift the phone a little", false)
            }

            switch phase {
            case .openingMarker0:
                return ("origin_find_marker_0", "Stand over marker 0 and point at it", true)
            case .markers1To3:
                let remaining = [1, 2, 3].filter { !lockedMarkerIDs.contains($0) }
                if remaining.count == 1, let nextID = remaining.first {
                    return ("walk_to_marker_\(nextID)", "Walk to marker \(nextID)", true)
                }
                return ("look_for_markers", "Look around for your markers", true)
            case .houseWallLookUp:
                return ("lookup_step", "Look up at the roof edge and the trees", true)
            case .closingMarker0:
                return ("return_marker_0", "Head back to marker 0", true)
            case .readyToFinish:
                return ("all_done_finish", "All done. Tap Finish.", true)
            }
        }()

        // Apply 0.7 s debounce for transient corrections (both on entry and when clearing back to baseline)
        // and 1.5 s minimum dwell (`YS-74`, `DESIGN-SPEC-NATIVE.md` §4c).
        if candKey != activeInstructionKey {
            let phaseChangedThisStep = (phase != phaseBeforeStep)
            let isUrgentOverride = candKey == "note_recording"
                || candKey == "storage_low"
                || candKey == "thermal_serious"
                || candKey == "all_done_finish"
                || candKey.hasPrefix("locked_marker_")
                || phaseChangedThisStep
                || didLockMarkerThisStep
                || (candKey == "hold_steady" && didStartHoldThisStep && !activeInstructionIsCorrection)

            if isImmediatePhaseOrLockCue && !activeInstructionIsCorrection {
                let dwellElapsed = (timestampS - activeInstructionSinceS) >= (Self.minInstructionDwellS - 1e-6)
                if dwellElapsed || isUrgentOverride || candKey == "hold_steady" {
                    commitInstruction(
                        key: candKey,
                        text: candText,
                        isCorrection: !isImmediatePhaseOrLockCue,
                        at: timestampS,
                        forceImmediate: true
                    )
                }
            } else if isUrgentOverride {
                commitInstruction(
                    key: candKey,
                    text: candText,
                    isCorrection: !isImmediatePhaseOrLockCue,
                    at: timestampS,
                    forceImmediate: true
                )
            } else {
                // Either entering a correction OR clearing an active correction back to the phase baseline:
                // require the condition/fix to hold for `0.7 s` (`debounceDurationS`) AND the active instruction
                // to have dwelled for `>= 1.5 s` (`minInstructionDwellS`).
                if pendingCandidateKey != candKey {
                    pendingCandidateKey = candKey
                    pendingCandidateText = candText
                    pendingCandidateSinceS = timestampS
                } else if (timestampS - pendingCandidateSinceS) >= (Self.debounceDurationS - 1e-6),
                          (timestampS - activeInstructionSinceS) >= (Self.minInstructionDwellS - 1e-6) {
                    commitInstruction(
                        key: candKey,
                        text: candText,
                        isCorrection: !isImmediatePhaseOrLockCue,
                        at: timestampS,
                        forceImmediate: true
                    )
                }
            }
        } else {
            pendingCandidateKey = nil
            pendingCandidateText = nil
        }

        // Strict haptic rules (YS-74):
        // 1. No haptics while a voice note is recording.
        // 2. Corrections never trigger a haptic.
        if isRecordingVoiceNote {
            emittedHaptic = nil
        }

        // Voice coach rules (`YS-75`, `DESIGN-SPEC-NATIVE.md` §4c):
        // - Spoken only for "Locked", phase changes, and non-counting corrections; never for counts (`hold_steady`).
        // - Silent while a voice note is recording or when `isVoiceCoachEnabled == false`.
        // - Hold paused due to motion (`hold_paused_moving`) speaks `"Hold steady."` after 2.0 s of pause.
        if activeInstructionKey == "hold_paused_moving" {
            holdPausedAccumulatedS += max(0.0, dt)
        } else {
            holdPausedAccumulatedS = 0.0
            didSpeakHoldPausedMovingThisPause = false
        }

        if isVoiceCoachEnabled && !isRecordingVoiceNote {
            if wasInstructionHistoryEmpty && activeInstructionKey == keyBeforeStep {
                recordSpokenPhraseForActiveInstructionIfNeeded(at: timestampS)
            } else if activeInstructionKey == "hold_paused_moving" && !didSpeakHoldPausedMovingThisPause && holdPausedAccumulatedS >= 2.0 {
                recordSpokenPhraseForActiveInstructionIfNeeded(at: timestampS)
            }
        } else {
            lastSpokenPhraseThisStep = nil
        }

        latestThermalSchedule = ThermalAdaptationSchedule.evaluate(
            thermalLevel: thermalLevel,
            isMarkerHoldActive: activeHoldMarkerID != nil
        )

        return (
            activeInstruction,
            activeInstructionKey,
            activeRingState,
            emittedHaptic,
            latestThermalSchedule
        )
    }

    private mutating func commitInstruction(
        key: String,
        text: String,
        isCorrection: Bool,
        at timestampS: Double,
        forceImmediate: Bool
    ) {
        guard key != activeInstructionKey || text != activeInstruction else { return }
        activeInstructionKey = key
        activeInstruction = text
        activeInstructionIsCorrection = isCorrection
        activeInstructionSinceS = timestampS
        pendingCandidateKey = nil
        pendingCandidateText = nil
        instructionHistory.append((timestampS, key, text))
        if isVoiceCoachEnabled && !isRecordingVoiceNote {
            recordSpokenPhraseForActiveInstructionIfNeeded(at: timestampS)
        } else {
            lastSpokenPhraseThisStep = nil
        }
    }

    private mutating func recordSpokenPhraseForActiveInstructionIfNeeded(at timestampS: Double) {
        guard isVoiceCoachEnabled && !isRecordingVoiceNote else {
            lastSpokenPhraseThisStep = nil
            return
        }
        guard let phrase = GuidedVoiceCatalog.spokenPhrase(
            forInstructionKey: activeInstructionKey,
            holdPausedDurationS: holdPausedAccumulatedS,
            alreadySpokenOnceKeys: spokenOnceKeys
        ) else {
            return
        }
        if GuidedVoiceCatalog.isSpokenAtMostOncePerWalk(key: activeInstructionKey) {
            spokenOnceKeys.insert(activeInstructionKey)
        }
        if activeInstructionKey == "hold_paused_moving" {
            didSpeakHoldPausedMovingThisPause = true
        }
        lastSpokenPhraseThisStep = phrase
        spokenHistory.append((timestampS, activeInstructionKey, activeInstruction, phrase))
    }
}

// MARK: - YS-75: Fixed Voice Coach Phrase Catalog

/// Maps deterministic `GuidedCaptureEngine` instruction keys to the exact fixed spoken phrases
/// from `DESIGN-SPEC-NATIVE.md` §4c (`YS-75`).
///
/// Rules (`YS-75`, `DESIGN-SPEC-NATIVE.md` §4a, §4c):
/// - Fixed phrases, spoken only for `"Locked"`, phase changes, and non-counting corrections; **never** for counts (`1, 2, 3` or `"Hold steady"` while counting).
/// - The voice says `"marker zero"`, `"marker one"`, `"marker two"`, `"marker three"`, while the glass shows `"marker 0"`.
/// - Silent while a voice note is recording (`note_recording`, `note_added`).
public enum GuidedVoiceCatalog: Sendable {
    public static func markerNumberWord(_ id: Int) -> String {
        switch id {
        case 0: return "zero"
        case 1: return "one"
        case 2: return "two"
        case 3: return "three"
        case 4: return "four"
        case 5: return "five"
        default: return "\(id)"
        }
    }

    public static func isSpokenAtMostOncePerWalk(key: String) -> Bool {
        switch key {
        case "storage_low", "thermal_serious", "battery_low", "dark_exposure":
            return true
        default:
            return false
        }
    }

    /// Return the spoken phrase for `key`, or `nil` if the instruction should be silent on the voice channel.
    public static func spokenPhrase(
        forInstructionKey key: String,
        holdPausedDurationS: Double = 0.0,
        alreadySpokenOnceKeys: Set<String> = []
    ) -> String? {
        if isSpokenAtMostOncePerWalk(key: key) && alreadySpokenOnceKeys.contains(key) {
            return nil
        }
        switch key {
        case "origin_find_marker_0":
            return "Stand over marker zero and point at it."
        case "hold_steady":
            // YS-75: Never speak during hold counting!
            return nil
        case "hold_paused_moving":
            // DESIGN-SPEC-NATIVE.md §4c: (after 2 s) "Hold steady."
            return holdPausedDurationS >= 2.0 ? "Hold steady." : nil
        case "locked_marker_0_opening":
            return "Got it. Every marker works like that."
        case "look_for_markers":
            return "Turn slowly and look for your markers."
        case "pace_turn_slower":
            return "Turn a little slower."
        case "pace_walk_slower":
            return "Slow down a little."
        case "pace_hold_steady":
            return "Hold steady."
        case "tilt_point_ground":
            return "Point at the ground ahead."
        case "tilt_lift_phone":
            return "Lift the phone a little."
        case "tracking_insufficient_features":
            return "Point at something with more detail, like the fence."
        case "tracking_excessive_motion":
            return "Move the phone more slowly."
        case "dark_exposure":
            return "It's getting dark. Hold each marker longer."
        case "lookup_step":
            return "Now look up at the roof edge, then the trees."
        case "lookup_sky_warning":
            return "Point at the roof edge, not the sky."
        case "return_marker_0":
            return "That's all of them. Head back to marker zero."
        case "all_done_finish":
            return "You're back at the start. All done."
        case "storage_low":
            return "Almost out of space. Finish soon."
        case "thermal_serious":
            return "iPhone is getting warm. Finish soon."
        case "thermal_critical":
            return "Your walk was saved. iPhone needs to cool down."
        case "battery_low":
            return "Battery low. Finish soon."
        case "note_recording", "note_added", "walk_interrupted":
            return nil
        default:
            if key.hasPrefix("locked_marker_"),
               let mid = Int(key.replacingOccurrences(of: "locked_marker_", with: "")) {
                return "Got marker \(markerNumberWord(mid))."
            }
            if key.hasPrefix("walk_to_marker_"),
               let mid = Int(key.replacingOccurrences(of: "walk_to_marker_", with: "")) {
                return "Now walk to marker \(markerNumberWord(mid))."
            }
            if key.hasPrefix("hold_too_far_") {
                return "Step closer."
            }
            if key.hasPrefix("hold_too_close_") {
                return "Step back."
            }
            return nil
        }
    }
}

// MARK: - YS-76: Pre-Flight Checklist & Stable-Frame Lens Smudge Detector

/// Result of evaluating a camera luma frame for a smudged or obstructed lens (`YS-76`).
///
/// Only runs on a **stable frame** (`turnRateRadS <= 0.12`, `predictedBlurPx <= 1.5`, `cameraVelocityMps <= 0.25`)
/// so a clean lens at dusk while moving never false-triggers the smudge card.
public struct LensSmudgeProbeResult: Sendable, Equatable {
    public let isStableFrame: Bool
    public let isSmudged: Bool
    public let maxTileLaplacianVariance: Double
    public let medianTileLaplacianVariance: Double
    public let maxTileContrast: Double

    public init(
        isStableFrame: Bool,
        isSmudged: Bool,
        maxTileLaplacianVariance: Double,
        medianTileLaplacianVariance: Double,
        maxTileContrast: Double
    ) {
        self.isStableFrame = isStableFrame
        self.isSmudged = isSmudged
        self.maxTileLaplacianVariance = maxTileLaplacianVariance
        self.medianTileLaplacianVariance = medianTileLaplacianVariance
        self.maxTileContrast = maxTileContrast
    }
}

/// Evaluates whether a camera frame shows a smudged/covered lens (`YS-76`).
public enum LensSmudgeDetector: Sendable {
    public static let maxStableTurnRateRadS: Double = 0.12
    public static let maxStablePredictedBlurPx: Double = 1.50
    public static let maxStableCameraVelocityMps: Double = 0.25
    public static let minCleanMaxTileLaplacianVariance: Double = 18.0

    public static func evaluate(
        lumaBytes: [UInt8],
        width: Int,
        height: Int,
        turnRateRadS: Double,
        predictedBlurPx: Double,
        cameraVelocityMps: Double = 0.0
    ) -> LensSmudgeProbeResult {
        let isStable = turnRateRadS <= maxStableTurnRateRadS
            && predictedBlurPx <= maxStablePredictedBlurPx
            && cameraVelocityMps <= maxStableCameraVelocityMps
        guard isStable, width >= 32, height >= 32, lumaBytes.count >= width * height else {
            return LensSmudgeProbeResult(
                isStableFrame: false,
                isSmudged: false,
                maxTileLaplacianVariance: 0.0,
                medianTileLaplacianVariance: 0.0,
                maxTileContrast: 0.0
            )
        }

        // Evaluate a 4x4 grid over the central 80% of the image
        let xStart = width / 10
        let yStart = height / 10
        let regionW = (width * 8) / 10
        let regionH = (height * 8) / 10
        let tileW = max(8, regionW / 4)
        let tileH = max(8, regionH / 4)

        var tileVariances: [Double] = []
        tileVariances.reserveCapacity(16)
        var maxContrast = 0.0

        for ty in 0..<4 {
            for tx in 0..<4 {
                let u0 = xStart + tx * tileW
                let v0 = yStart + ty * tileH
                let u1 = min(width - 2, u0 + tileW)
                let v1 = min(height - 2, v0 + tileH)
                guard u1 > u0 + 2, v1 > v0 + 2 else { continue }

                var sumLap = 0.0
                var sumLapSq = 0.0
                var count = 0
                var minLuma: UInt8 = 255
                var maxLuma: UInt8 = 0

                for v in stride(from: v0 + 1, to: v1, by: 2) {
                    let row = v * width
                    let rowUp = (v - 1) * width
                    let rowDn = (v + 1) * width
                    for u in stride(from: u0 + 1, to: u1, by: 2) {
                        let c = lumaBytes[row + u]
                        if c < minLuma { minLuma = c }
                        if c > maxLuma { maxLuma = c }
                        let lap = Double(lumaBytes[rowUp + u])
                            + Double(lumaBytes[rowDn + u])
                            + Double(lumaBytes[row + u - 1])
                            + Double(lumaBytes[row + u + 1])
                            - 4.0 * Double(c)
                        sumLap += lap
                        sumLapSq += lap * lap
                        count += 1
                    }
                }

                if count > 0 {
                    let mean = sumLap / Double(count)
                    let variance = max(0.0, (sumLapSq / Double(count)) - (mean * mean))
                    tileVariances.append(variance)
                    let contrast = Double(maxLuma) - Double(minLuma)
                    if contrast > maxContrast {
                        maxContrast = contrast
                    }
                }
            }
        }

        guard !tileVariances.isEmpty else {
            return LensSmudgeProbeResult(
                isStableFrame: true,
                isSmudged: false,
                maxTileLaplacianVariance: 0.0,
                medianTileLaplacianVariance: 0.0,
                maxTileContrast: 0.0
            )
        }

        let sorted = tileVariances.sorted()
        let maxVar = sorted.last ?? 0.0
        let medVar = sorted[sorted.count / 2]
        // A thumb or heavy smudge across the lens blurs high-frequency detail in every tile (`maxVar < 18.0`),
        // whereas a clean lens on a stable frame (even at dusk) preserves sharp edges in at least one tile (`maxVar >= 18.0`).
        let smudged = maxVar < minCleanMaxTileLaplacianVariance
        return LensSmudgeProbeResult(
            isStableFrame: true,
            isSmudged: smudged,
            maxTileLaplacianVariance: maxVar,
            medianTileLaplacianVariance: medVar,
            maxTileContrast: maxContrast
        )
    }
}

public enum PreFlightCheckState: String, Sendable, Equatable {
    case pass
    case warn
    case block
    case hidden
}

public struct PreFlightCheckRow: Sendable, Equatable {
    public let id: String
    public let state: PreFlightCheckState
    public let title: String
    public let detail: String
    public let actionTitle: String?

    public init(
        id: String,
        state: PreFlightCheckState,
        title: String,
        detail: String,
        actionTitle: String? = nil
    ) {
        self.id = id
        self.state = state
        self.title = title
        self.detail = detail
        self.actionTitle = actionTitle
    }
}

public struct PreFlightChecklistReport: Sendable, Equatable {
    public let daylightRow: PreFlightCheckRow
    public let storageRow: PreFlightCheckRow
    public let batteryRow: PreFlightCheckRow
    public let thermalRow: PreFlightCheckRow
    public let lensRow: PreFlightCheckRow
    public let printCheckRow: PreFlightCheckRow
    public let tapeMeasurementRow: PreFlightCheckRow
    public let canStartWalk: Bool
    public let blockingReasons: [String]

    public var visibleRows: [PreFlightCheckRow] {
        [
            daylightRow,
            storageRow,
            batteryRow,
            thermalRow,
            lensRow,
            printCheckRow,
            tapeMeasurementRow,
        ].filter { $0.state != .hidden }
    }
}

/// Evaluates all pre-flight readiness checks (`YS-76`, `YS-77`, `YS-80` corrections #3 & #4).
public enum PreFlightChecklistEvaluator: Sendable {
    /// Minimum free disk space to start a walk: `10 GB` (`10_000_000_000` bytes, `YS-76`, `YS-80` #4).
    public static let minRequiredStorageBytes: Int64 = 10_000_000_000
    /// Warning band for free disk space: `10–12 GB` (`12_000_000_000` bytes, `YS-76`).
    public static let warnStorageBytes: Int64 = 12_000_000_000
    /// Battery block threshold: `<= 10%` (`0.10`, `YS-76`).
    public static let blockBatteryFraction: Double = 0.10
    /// Battery warn threshold: `<= 20%` (`0.20`, `YS-76`).
    public static let warnBatteryFraction: Double = 0.20

    public static func evaluate(
        availableDiskBytes: Int64,
        batteryLevelFraction: Double,
        thermalLevel: CaptureThermalLevel,
        exposureDurationS: Double,
        iso: Double,
        isLocationAuthorized: Bool,
        minutesUntilSunset: Double? = nil,
        sunsetTimeText: String? = nil,
        lensProbe: LensSmudgeProbeResult? = nil,
        printedMarkerSizeMM: Double = 180.0,
        isPrintSizeConfirmed: Bool = false,
        preWalkTapeMeters0To1: Double? = nil
    ) -> PreFlightChecklistReport {
        // 1. Storage check (`>= 10 GB` required, warns at `10–12 GB`, `YS-76`, `YS-80` #4)
        let gbFree = Double(max(0, availableDiskBytes)) / 1_000_000_000.0
        let storageRow: PreFlightCheckRow = {
            if availableDiskBytes < minRequiredStorageBytes {
                return PreFlightCheckRow(
                    id: "storage",
                    state: .block,
                    title: "Space on iPhone",
                    detail: "Free up 10 GB to start.",
                    actionTitle: "Open Settings"
                )
            } else if availableDiskBytes < warnStorageBytes {
                return PreFlightCheckRow(
                    id: "storage",
                    state: .warn,
                    title: "Space on iPhone",
                    detail: String(format: "%.1f GB free", gbFree)
                )
            } else {
                return PreFlightCheckRow(
                    id: "storage",
                    state: .pass,
                    title: "Space on iPhone",
                    detail: String(format: "%.0f GB free", gbFree)
                )
            }
        }()

        // 2. Battery check (warns <= 20%, blocks <= 10%, `YS-76`)
        let clampedBatt = batteryLevelFraction < 0 ? 1.0 : min(1.0, max(0.0, batteryLevelFraction))
        let battPct = Int((clampedBatt * 100.0).rounded())
        let batteryRow: PreFlightCheckRow = {
            if clampedBatt <= blockBatteryFraction + 1e-6 {
                return PreFlightCheckRow(
                    id: "battery",
                    state: .block,
                    title: "Battery",
                    detail: "\(battPct)%. Charge above 10% to start."
                )
            } else if clampedBatt <= warnBatteryFraction + 1e-6 {
                return PreFlightCheckRow(
                    id: "battery",
                    state: .warn,
                    title: "Battery",
                    detail: "\(battPct)%. Plug in if you can."
                )
            } else {
                return PreFlightCheckRow(
                    id: "battery",
                    state: .pass,
                    title: "Battery",
                    detail: "\(battPct)%"
                )
            }
        }()

        // 3. Heat / thermal check (warns at fair, blocks at serious or worse, `YS-76`)
        let thermalRow: PreFlightCheckRow = {
            switch thermalLevel {
            case .nominal:
                return PreFlightCheckRow(
                    id: "thermal",
                    state: .hidden,
                    title: "Temperature",
                    detail: "Normal"
                )
            case .fair:
                return PreFlightCheckRow(
                    id: "thermal",
                    state: .warn,
                    title: "Temperature",
                    detail: "iPhone is warm. Rest it in the shade first."
                )
            case .serious, .critical:
                return PreFlightCheckRow(
                    id: "thermal",
                    state: .block,
                    title: "Temperature",
                    detail: "iPhone is too hot to start."
                )
            }
        }()

        // 4. Daylight & Live Exposure Probe check (`YS-76`):
        // - "light from a live exposure probe ('Too dark' is said here, never mid-walk)"
        // - "Location is optional: if it's refused, the daylight row falls back to the exposure probe and never blocks."
        let exposureIndex = exposureDurationS * max(50.0, iso)
        let isProbeTooDark = (exposureDurationS >= 0.045 && iso >= 1000.0) || exposureIndex >= 50.0
        let isProbeDim = (exposureDurationS > 0.020) || exposureIndex >= 14.0
        let daylightRow: PreFlightCheckRow = {
            if !isLocationAuthorized {
                // Location refused: fall back to exposure probe and NEVER block (`YS-76`)
                if isProbeTooDark || isProbeDim {
                    return PreFlightCheckRow(
                        id: "daylight",
                        state: .warn,
                        title: "Daylight",
                        detail: "Getting dim. Walks take about 10 min."
                    )
                }
                return PreFlightCheckRow(
                    id: "daylight",
                    state: .pass,
                    title: "Daylight",
                    detail: "Enough light"
                )
            }

            if isProbeTooDark || (minutesUntilSunset != nil && minutesUntilSunset! <= -15.0) {
                return PreFlightCheckRow(
                    id: "daylight",
                    state: .block,
                    title: "Daylight",
                    detail: "Too dark for sharp photos."
                )
            }
            if let mins = minutesUntilSunset, mins <= 30.0 {
                return PreFlightCheckRow(
                    id: "daylight",
                    state: .warn,
                    title: "Daylight",
                    detail: "Dark in \( max(1, Int(mins.rounded())) ) min. Walks take about 10."
                )
            }
            if isProbeDim {
                return PreFlightCheckRow(
                    id: "daylight",
                    state: .warn,
                    title: "Daylight",
                    detail: "Getting dim. Walks take about 10 min."
                )
            }
            let detailText = sunsetTimeText.map { "Until \($0)" } ?? "Enough light"
            return PreFlightCheckRow(
                id: "daylight",
                state: .pass,
                title: "Daylight",
                detail: detailText
            )
        }()

        // 5. Lens smudge check (evaluated on a stable frame only, `YS-76`)
        let lensRow: PreFlightCheckRow = {
            if let probe = lensProbe, probe.isStableFrame, probe.isSmudged {
                return PreFlightCheckRow(
                    id: "lens",
                    state: .warn,
                    title: "Lens looks smudged",
                    detail: "Wipe it, then Check Again",
                    actionTitle: "Check Again"
                )
            }
            return PreFlightCheckRow(
                id: "lens",
                state: .hidden,
                title: "Lens",
                detail: "Clean"
            )
        }()

        // 6. Print check card (`YS-76`, `DESIGN-SPEC-NATIVE.md` §4d)
        let cmVal = printedMarkerSizeMM / 10.0
        let cmText = abs(cmVal.rounded() - cmVal) < 0.05
            ? String(format: "%.0f cm", cmVal)
            : String(format: "%.1f cm", cmVal)
        let printCheckRow = PreFlightCheckRow(
            id: "print_check",
            state: isPrintSizeConfirmed ? .pass : .warn,
            title: "Check your print",
            detail: isPrintSizeConfirmed
                ? "Checked (\(cmText)) ✓"
                : "Lay a tape on the black square of marker 0. Does it measure \(cmText)?",
            actionTitle: isPrintSizeConfirmed ? nil : "Check"
        )

        // 7. Pre-walk centre-to-centre Marker 0 -> Marker 1 tape measurement (`YS-77`)
        let tapeMeasurementRow: PreFlightCheckRow = {
            if let tapeM = preWalkTapeMeters0To1, tapeM > 0 {
                return PreFlightCheckRow(
                    id: "tape_measurement",
                    state: .pass,
                    title: "Tape measurement",
                    detail: String(format: "%.2f m ✓", tapeM),
                    actionTitle: "Edit"
                )
            }
            return PreFlightCheckRow(
                id: "tape_measurement",
                state: .warn,
                title: "Tape measurement",
                detail: "Measure from the centre of marker 0 to the centre of marker 1.",
                actionTitle: "Add"
            )
        }()

        var blockers: [String] = []
        for row in [daylightRow, storageRow, batteryRow, thermalRow] where row.state == .block {
            blockers.append(row.detail)
        }

        return PreFlightChecklistReport(
            daylightRow: daylightRow,
            storageRow: storageRow,
            batteryRow: batteryRow,
            thermalRow: thermalRow,
            lensRow: lensRow,
            printCheckRow: printCheckRow,
            tapeMeasurementRow: tapeMeasurementRow,
            canStartWalk: blockers.isEmpty,
            blockingReasons: blockers
        )
    }
}

// MARK: - YS-77: Centre-to-Centre Pre-Walk Tape Check (Marker 0 -> Marker 1)

/// Represents a standard `session.json["tape_checks"]` entry (`contract/session.schema.json:406-480`)
/// automatically generated from the locked 3D positions of Marker 0 and Marker 1 (`YS-77`).
public struct PreWalkTapeCheckEntry: Sendable, Equatable {
    public let id: Int
    public let role: String
    public let tAS: Double
    public let tBS: Double
    public let savedIndexA: Int
    public let savedIndexB: Int
    public let pointAWorldM: SIMD3<Double>
    public let pointBWorldM: SIMD3<Double>
    public let tapeM: Double
    public let measuredM: Double
    public let residualM: Double

    public init(
        id: Int = 0,
        role: String = "primary",
        tAS: Double,
        tBS: Double,
        savedIndexA: Int,
        savedIndexB: Int,
        pointAWorldM: SIMD3<Double>,
        pointBWorldM: SIMD3<Double>,
        tapeM: Double
    ) {
        self.id = id
        self.role = role
        self.tAS = tAS
        self.tBS = tBS
        self.savedIndexA = savedIndexA
        self.savedIndexB = savedIndexB
        self.pointAWorldM = pointAWorldM
        self.pointBWorldM = pointBWorldM
        self.tapeM = tapeM
        let dist = simd_length(pointBWorldM - pointAWorldM)
        self.measuredM = dist
        self.residualM = abs(dist - tapeM)
    }

    /// Produces the exact dictionary validated by `contract/session.schema.json:406-480` and `recompute_tape_check` (`grading.py:37-57`).
    public func toJSONDictionary() -> [String: Any] {
        [
            "id": id,
            "role": role,
            "t_a_s": tAS,
            "t_b_s": tBS,
            "saved_index_a": savedIndexA,
            "saved_index_b": savedIndexB,
            "point_a_world_m": [pointAWorldM.x, pointAWorldM.y, pointAWorldM.z],
            "point_b_world_m": [pointBWorldM.x, pointBWorldM.y, pointBWorldM.z],
            "tape_m": tapeM,
            "measured_m": measuredM,
            "residual_m": residualM,
        ]
    }
}
