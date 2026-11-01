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

    public mutating func anchorGroundPlaneY(_ yWorld: Double) {
        if groundPlaneYWorld == nil {
            groundPlaneYWorld = yWorld
        } else if let existing = groundPlaneYWorld {
            groundPlaneYWorld = 0.8 * existing + 0.2 * yWorld
        }
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
                                let ix = Int(floor(center.pt.x / cellSizeM))
                                let iz = Int(floor(center.pt.z / cellSizeM))
                                let cell = GroundGridCell(ix: ix, iz: iz)
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

    /// Transform world-frame `(ix, iz)` ground cells into Marker-0 Yard Frame `0.25 m` cells and compute
    /// the cell agreement ratio against a reference set of yard-frame ground points (`YS-78`).
    public static func computeCellAgreementWithToolkitGroundPoints(
        worldGroundPoints: [(positionWorld: SIMD3<Double>, confidence: UInt8, rangeM: Double)],
        groundYWorld: Double,
        cellSizeM: Double = 0.25,
        maxAbsZ: Double = 0.25,
        minRangeM: Double = 0.50,
        maxRangeM: Double = 4.00
    ) -> (phoneCells: Set<GroundGridCell>, toolkitCells: Set<GroundGridCell>, agreementRatio: Double) {
        var phoneSet: Set<GroundGridCell> = []
        var toolkitSet: Set<GroundGridCell> = []

        for sample in worldGroundPoints {
            guard sample.confidence >= 1 else { continue }
            let dh = abs(sample.positionWorld.y - groundYWorld)
            let ix = Int(floor(sample.positionWorld.x / cellSizeM))
            let iz = Int(floor(sample.positionWorld.z / cellSizeM))
            let cell = GroundGridCell(ix: ix, iz: iz)

            // Phone rule (YS-78): 0.5–4.0 m, within ±0.25 m of marker plane
            if sample.rangeM >= minRangeM && sample.rangeM <= maxRangeM && dh <= maxAbsZ {
                phoneSet.insert(cell)
            }
            // Toolkit rule (geometry.py:876): range <= 4.0 m, within ±0.25 m of marker plane
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
            if accumulatedValidDurationS >= Self.requiredDurationS {
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

    /// Compute the maximum 3D distance from the component-wise median of `samples`.
    public static func maxSpreadFromMedianM(_ samples: [ConfirmedMarkerStillSample]) -> Double {
        guard samples.count >= 2 else { return 0.0 }
        let med = PhoneLoopRuleState.medianPosition3D(samples.map(\.positionWorld))
        var maxDist = 0.0
        for s in samples {
            let d = simd_length(s.positionWorld - med)
            if d > maxDist { maxDist = d }
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
    public private(set) var phase: GuidedWalkPhase = .openingMarker0
    public private(set) var activeInstruction: String = "Stand over marker 0 and point at it"
    public private(set) var activeInstructionKey: String = "origin_find_marker_0"
    public private(set) var activeInstructionSinceS: Double = -100.0
    public private(set) var instructionHistory: [(timestampS: Double, key: String, text: String)] = []

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

    // Debounce & dwell state
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

    public init(requireLookUpStep: Bool = true) {
        self.requireLookUpStep = requireLookUpStep
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
                at: timestampS,
                forceImmediate: true
            )
        }
    }

    public mutating func markEndEarlyChosen(at timestampS: Double) {
        endEarlyChosen = true
        if phase == .markers1To3 {
            phase = .closingMarker0
            commitInstruction(
                key: "return_marker_0",
                text: "Head back to marker 0",
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
                at: timestampS,
                forceImmediate: true
            )
        } else {
            phase = .closingMarker0
            commitInstruction(
                key: "return_marker_0",
                text: "Head back to marker 0",
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
        relaxViewingAngleAndSizeForLegacyReplay: Bool = false,
        didRunMarkerDetection: Bool = true
    ) -> (
        instruction: String,
        instructionKey: String,
        ringState: MarkerRingVisualState?,
        haptic: GuidedHapticEvent?,
        thermalSchedule: ThermalAdaptationSchedule
    ) {
        if instructionHistory.isEmpty {
            activeInstructionSinceS = timestampS
            instructionHistory.append((timestampS, activeInstructionKey, activeInstruction))
        }

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
            let maxAngle = relaxViewingAngleAndSizeForLegacyReplay ? 65.0 : Self.maxViewingAngleDeg
            let maxSizeDev = Self.maxSizeDeviationRatio
            let maxSpread = relaxViewingAngleAndSizeForLegacyReplay ? 0.08 : Self.maxPositionSpreadM

            let angleOK = target.sighting.viewingAngleDeg <= maxAngle
            let sizeOK = target.sighting.sizeMMPrinted > 0
                ? (abs(target.sighting.sizeMMMeasured - target.sighting.sizeMMPrinted) / target.sighting.sizeMMPrinted) <= maxSizeDev
                : true

            if trackingState == "normal" && target.isConfirmedStill && gateVerdict.passes && angleOK && sizeOK {
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
                    hold.isPaused = false
                    hold.pauseReason = nil
                }
            } else if !hold.isLocked {
                // Condition broke: pause rather than reset (YS-74)
                hold.isPaused = !hold.confirmedSamples.isEmpty
                if gateVerdict.reason == "marker_too_far" {
                    hold.pauseReason = "too_far"
                } else if !gateVerdict.passes || !angleOK {
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
                if ring.isPaused && ring.confirmedSegments > 0 {
                    return ("hold_paused_moving", "Hold steady… almost there", false)
                }
                return ("hold_steady", "Hold steady", true)
            }

            // Corrections are silent while a hold is counting; outside a hold, check pace & tilt
            if turnRateRadS > ExposureAndStillnessGate.maxTurnRateRadS && predictedBlurPx > ExposureAndStillnessGate.maxPredictedBlurPx {
                return ("pace_turn_slower", "Turn a little slower", false)
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

        // Apply 0.7 s debounce for transient corrections and 1.5 s minimum dwell
        if candKey != activeInstructionKey {
            if isImmediatePhaseOrLockCue {
                let dwellElapsed = (timestampS - activeInstructionSinceS) >= (Self.minInstructionDwellS - 1e-6)
                // Final completion ("All done. Tap Finish.") or "Head back to marker 0" when all 1..3 lock transitions immediately
                let isTerminalOrPhaseShift = candKey == "all_done_finish"
                    || candKey == "return_marker_0"
                    || candKey == "lookup_step"
                    || candKey.hasPrefix("locked_marker_")
                    || candKey == "hold_steady"
                if dwellElapsed || isTerminalOrPhaseShift {
                    commitInstruction(key: candKey, text: candText, at: timestampS, forceImmediate: true)
                }
            } else {
                if pendingCandidateKey != candKey {
                    pendingCandidateKey = candKey
                    pendingCandidateText = candText
                    pendingCandidateSinceS = timestampS
                } else if (timestampS - pendingCandidateSinceS) >= (Self.debounceDurationS - 1e-6),
                          (timestampS - activeInstructionSinceS) >= (Self.minInstructionDwellS - 1e-6) {
                    commitInstruction(key: candKey, text: candText, at: timestampS, forceImmediate: true)
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
        at timestampS: Double,
        forceImmediate: Bool
    ) {
        guard key != activeInstructionKey || text != activeInstruction else { return }
        activeInstructionKey = key
        activeInstruction = text
        activeInstructionSinceS = timestampS
        pendingCandidateKey = nil
        pendingCandidateText = nil
        instructionHistory.append((timestampS, key, text))
    }
}
