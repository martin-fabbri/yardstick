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

import AVFoundation
import CoreImage
import CoreLocation
import CryptoKit
import Foundation
import ImageIO
import UIKit
import UniformTypeIdentifiers
import simd

#if canImport(ARKit)
import ARKit
#endif

/// Crash-safe, zero-`ARFrame`-retention LiDAR capture session recorder (`YS-00`–`YS-14`, `YS-21`, `YS-22`, `YS-30`–`YS-33`, `YS-40`–`YS-44`).
///
/// Key engineering invariants (`docs/DESIGN.md`):
/// 1. **Zero `ARFrame` retention**: inside `session(_:didUpdate:)`, pixel buffers and transforms are copied out
///    synchronously in < 2 ms and dispatched to `writerQueue`; the `ARFrame` reference is never stored.
/// 2. **Gravity-only world alignment (`YS-03`, `Q020`)**: `.gravity` prevents magnetometer yaw spikes mid-walk;
///    True North is calibrated at Marker 0 at start using >= 10 heading pairs (`YS-14`).
/// 3. **Single-clock audio (`YS-01`, `YS-06`, `Q018`)**: `providesAudioData = true` delivers audio on the exact
///    same monotonic host clock as `ARFrame.timestamp`, resampled to 16 kHz mono 16-bit PCM `audio.wav`.
/// 4. **Crash safety (`YS-08`, `YS-09`, `YS-10`)**: `isIdleTimerDisabled = true`, `movieFragmentInterval = 5s`
///    on `AVAssetWriter`, explicit stream write error handling, and interruption/background observers.
@MainActor
public final class CaptureSessionRecorder: NSObject, ObservableObject {
    public enum SessionPhase: String, Sendable {
        case setup
        case recording
        case sealing
        case completed
        case failed
    }

    @Published public private(set) var phase: SessionPhase = .setup
    @Published public private(set) var statusBanner: String = "Ready — place Marker 0 on flat ground"
    @Published public private(set) var captureID: String = ""
    @Published public private(set) var savedFrameCount: Int = 0
    @Published public private(set) var droppedFrameCount: Int = 0
    @Published public private(set) var detectedMarkerIDs: Set<Int> = []
    @Published public private(set) var northSampleCount: Int = 0
    @Published public private(set) var trackingStateText: String = "normal"
    @Published public private(set) var pitchDeg: Double = -30.0
    @Published public private(set) var rollDeg: Double = 0.0
    @Published public private(set) var crosshairDepthMeters: Double? = nil
    @Published public private(set) var pendingTapeEndpointAWorld: SIMD3<Double>? = nil
    @Published public private(set) var pendingTapeEndpointBWorld: SIMD3<Double>? = nil
    @Published public private(set) var tapeCheckCount: Int = 0
    @Published public private(set) var loopClosureGapMeters: Double? = nil
    @Published public private(set) var provisionalGrade: String = "—"
    @Published public private(set) var provisionalReasons: [String] = []
    @Published public private(set) var archiveZipURL: URL? = nil

    /// Per-marker printed size configuration (`250.0` mm Letter/A4 or `180.0` mm clipboard).
    @Published public var markerSizesMM: [Int: Double] = [
        0: 250.0, 1: 250.0, 2: 250.0, 3: 250.0, 4: 250.0, 5: 250.0,
    ]

    #if canImport(ARKit)
    public let arSession = ARSession()
    #endif

    private let writerQueue = DispatchQueue(label: "io.yardstick.capture.writer", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let locationManager = CLLocationManager()
    private var notificationObservers: [NSObjectProtocol] = []

    private var captureDirURL: URL?
    private var t0MonotonicS: Double?
    private var audioFirstSampleTS: Double?
    private var startedAtUTC: String = ""
    private var lastHeartbeatS: Double = 0.0
    private var lastStillSavedTimeS: Double = -10.0
    private var lastNorthSampleTimeS: Double = -10.0
    private var lastDeliveredFrameTimestampS: Double?
    private var pendingForceMarkerDetect: Bool = false

    private let requestedWidth: Int = 1920
    private let requestedHeight: Int = 1440
    private let requestedFPS: Int = 30
    private var actualWidth: Int = 1920
    private var actualHeight: Int = 1440
    private var actualFPS: Int = 30

    private var assetWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoSessionStarted: Bool = false
    private var pendingWriterJobs: Int = 0

    private var latestCameraTransform: simd_double4x4 = matrix_identity_double4x4
    private var latestCrosshairWorld: SIMD3<Double>?
    private var latestSavedIndex: Int = 0
    private var latestTimestampS: Double = 0.0
    private var latestTrueHeadingDeg: Double?
    private var latestMagneticHeadingDeg: Double?
    private var latestHeadingAccuracyDeg: Double?
    private var latestLocation: CLLocation?
    private var locationAuthStatus: String = "unavailable"
    private var northSamples: [[String: Double]] = []
    private var marker0FirstPassPos: SIMD3<Double>?
    private var marker0LastPassPos: SIMD3<Double>?
    private var marker0FirstPassTimeS: Double?
    private var marker0LastPassTimeS: Double?
    private var marker0MaxExcursionM: Double = 0.0
    private var tapeChecks: [[String: Any]] = []
    private var pendingTapeASavedIndex: Int = 0
    private var pendingTapeATimeS: Double = 0.0
    private var pendingTapeBSavedIndex: Int = 0
    private var pendingTapeBTimeS: Double = 0.0
    private var audioPCMByteCount: UInt32 = 0
    private var savedStillsIndices: Set<Int> = []

    public override init() {
        super.init()
        locationManager.delegate = self
        locationManager.headingFilter = kCLHeadingFilterNone
        #if canImport(ARKit)
        arSession.delegate = self
        #endif
        registerLifecycleObservers()
    }

    /// Exact hardware model identifier (e.g. `"iPhone18,1"`) via `uname` (`YS-00`).
    public static var hardwareModelIdentifier: String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machineMirror = Mirror(reflecting: sysinfo.machine)
        let identifier = machineMirror.children.reduce("") { id, element in
            guard let value = element.value as? Int8, value != 0 else { return id }
            return id + String(UnicodeScalar(UInt8(value)))
        }
        return identifier.isEmpty ? UIDevice.current.model : identifier
    }

    /// Verify hardware supports LiDAR scene depth (`YS-00`).
    public static var isLiDARSupported: Bool {
        #if canImport(ARKit) && !targetEnvironment(simulator)
        return ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
            || ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        #else
        return false
        #endif
    }

    private func registerLifecycleObservers() {
        let nc = NotificationCenter.default
        let resignObs = nc.addObserver(
            forName: UIApplication.willResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleInterruption(stopReason: "backgrounded")
            }
        }
        let bgObs = nc.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.handleInterruption(stopReason: "backgrounded")
            }
        }
        let audioObs = nc.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notif in
            let typeVal = notif.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            if typeVal == AVAudioSession.InterruptionType.began.rawValue {
                Task { @MainActor in
                    self?.handleInterruption(stopReason: "interrupted")
                }
            }
        }
        notificationObservers = [resignObs, bgObs, audioObs]
    }

    public func toggleMarkerSize(for markerID: Int) {
        let current = markerSizesMM[markerID] ?? 250.0
        markerSizesMM[markerID] = (current == 250.0) ? 180.0 : 250.0
    }

    public func requestPermissionsAndStart() {
        guard Self.isLiDARSupported else {
            phase = .failed
            statusBanner = "Unsupported device: LiDAR sensor is required (YS-00)"
            return
        }

        UIApplication.shared.isIdleTimerDisabled = true
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            locationManager.startUpdatingHeading()
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let nowUTC = formatter.string(from: Date())
        startedAtUTC = nowUTC
        let compactTS = nowUTC.replacingOccurrences(of: ":", with: "").replacingOccurrences(of: "-", with: "")
        captureID = "\(compactTS)_yardstick"

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent(captureID, isDirectory: true)
        captureDirURL = dir
        savedFrameCount = 0
        droppedFrameCount = 0
        detectedMarkerIDs.removeAll()
        northSamples.removeAll()
        northSampleCount = 0
        tapeChecks.removeAll()
        tapeCheckCount = 0
        pendingTapeEndpointAWorld = nil
        pendingTapeEndpointBWorld = nil
        loopClosureGapMeters = nil
        marker0FirstPassPos = nil
        marker0LastPassPos = nil
        marker0FirstPassTimeS = nil
        marker0LastPassTimeS = nil
        marker0MaxExcursionM = 0.0
        audioPCMByteCount = 0
        t0MonotonicS = nil
        audioFirstSampleTS = nil
        lastStillSavedTimeS = -10.0
        lastNorthSampleTimeS = -10.0
        lastDeliveredFrameTimestampS = nil
        pendingWriterJobs = 0
        savedStillsIndices.removeAll()

        do {
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("depth"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("confidence"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: dir.appendingPathComponent("stills"), withIntermediateDirectories: true)
            try initializeCSVFiles(in: dir)
            try initializeAudioWavHeader(at: dir.appendingPathComponent("audio.wav"))

            #if canImport(ARKit) && !targetEnvironment(simulator)
            let config = ARWorldTrackingConfiguration()
            config.worldAlignment = .gravity
            config.providesAudioData = true
            config.environmentTexturing = .none
            config.isAutoFocusEnabled = true
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                config.frameSemantics = [.sceneDepth]
            }
            if let fmt = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
                Int($0.imageResolution.width) == requestedWidth
                    && Int($0.imageResolution.height) == requestedHeight
                    && $0.framesPerSecond == requestedFPS
            }) ?? ARWorldTrackingConfiguration.supportedVideoFormats.first {
                config.videoFormat = fmt
                actualWidth = Int(fmt.imageResolution.width)
                actualHeight = Int(fmt.imageResolution.height)
                actualFPS = fmt.framesPerSecond
            }
            try initializeVideoWriter(at: dir.appendingPathComponent("rgb.mp4"), width: actualWidth, height: actualHeight)
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            #else
            try initializeVideoWriter(at: dir.appendingPathComponent("rgb.mp4"), width: actualWidth, height: actualHeight)
            #endif

            writeSessionJSON(in: dir, status: "recording", stopReason: nil, failedStream: nil)
            phase = .recording
            statusBanner = "Recording — hold steady on Marker 0 for North & Origin"
        } catch {
            UIApplication.shared.isIdleTimerDisabled = false
            phase = .failed
            statusBanner = "Failed to initialize capture directory: \(error.localizedDescription)"
        }
    }

    private func initializeVideoWriter(at url: URL, width: Int, height: Int) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        writer.movieFragmentInterval = CMTime(seconds: 5.0, preferredTimescale: 600)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        if writer.canAdd(input) {
            writer.add(input)
        }
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "Yardstick", code: -1, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter startWriting failed"])
        }
        assetWriter = writer
        videoWriterInput = input
        pixelBufferAdaptor = adaptor
        videoSessionStarted = false
    }

    /// Record a timed user event (`marker`, `tape_a`, `tape_b`, `note`, `done`, `YS-07`).
    public func logEvent(kind: String, detail: String) {
        guard let dir = captureDirURL, phase == .recording else { return }
        if kind == "marker" {
            pendingForceMarkerDetect = true
        }
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        let safeDetail = detail.replacingOccurrences(of: ",", with: ";").replacingOccurrences(of: "\n", with: " ")
        let line = String(format: "%.6f,%d,%@,%@\n", t, sIdx, kind, safeDetail)
        appendString(line, to: dir.appendingPathComponent("events.csv"), stream: "events")
    }

    /// Capture Endpoint A of a tape-measure check using the live 5x5 center crosshair LiDAR patch (`YS-40`).
    public func captureTapeEndpointA() {
        guard let worldPt = latestCrosshairWorld else {
            statusBanner = "Aim center crosshair at surface (0.2–4.5 m) first"
            return
        }
        pendingTapeEndpointAWorld = worldPt
        pendingTapeASavedIndex = latestSavedIndex
        pendingTapeATimeS = latestTimestampS
        logEvent(kind: "tape_a", detail: String(format: "%.3f;%.3f;%.3f", worldPt.x, worldPt.y, worldPt.z))
        statusBanner = "Endpoint A locked — aim at Endpoint B and tap Tape B"
    }

    /// Immediately lock Endpoint B 3D world position when the user taps `Tape B` BEFORE presenting the keypad (`YS-40`).
    public func captureTapeEndpointB() -> Bool {
        guard pendingTapeEndpointAWorld != nil else {
            statusBanner = "Tap Tape A before locking Tape B"
            return false
        }
        guard let worldPt = latestCrosshairWorld else {
            statusBanner = "Aim center crosshair at Endpoint B surface (0.2–4.5 m) first"
            return false
        }
        pendingTapeEndpointBWorld = worldPt
        pendingTapeBSavedIndex = latestSavedIndex
        pendingTapeBTimeS = latestTimestampS
        statusBanner = "Endpoint B locked — enter tape-measure reading (m)"
        return true
    }

    /// Commit a primary or held-out tape check using the already-locked Endpoint A and Endpoint B (`YS-40`, `YS-41`).
    public func commitTapeEndpointB(tapeMeters: Double, role: String = "primary") {
        guard let ptA = pendingTapeEndpointAWorld,
              let ptB = pendingTapeEndpointBWorld ?? latestCrosshairWorld,
              tapeMeters > 0
        else {
            statusBanner = "Both endpoints A & B and a positive tape distance are required"
            return
        }
        let tBS = pendingTapeEndpointBWorld != nil ? pendingTapeBTimeS : latestTimestampS
        let sIdxB = pendingTapeEndpointBWorld != nil ? pendingTapeBSavedIndex : latestSavedIndex
        let measuredM = simd_length(ptB - ptA)
        let residualM = abs(measuredM - tapeMeters)
        let checkID = tapeChecks.count
        let entry: [String: Any] = [
            "id": checkID,
            "role": role,
            "t_a_s": pendingTapeATimeS,
            "t_b_s": tBS,
            "saved_index_a": pendingTapeASavedIndex,
            "saved_index_b": sIdxB,
            "point_a_world_m": [ptA.x, ptA.y, ptA.z],
            "point_b_world_m": [ptB.x, ptB.y, ptB.z],
            "tape_m": tapeMeters,
            "measured_m": measuredM,
            "residual_m": residualM,
        ]
        tapeChecks.append(entry)
        tapeCheckCount = tapeChecks.count
        pendingTapeEndpointAWorld = nil
        pendingTapeEndpointBWorld = nil
        logEvent(kind: "tape_b", detail: String(format: "tape=%.3fm measured=%.3fm res=%.3fm", tapeMeters, measuredM, residualM))
        updateProvisionalGrade()
        statusBanner = String(format: "Tape #%d logged: |%.2fm - %.2fm| = %.1f cm", checkID + 1, measuredM, tapeMeters, residualM * 100.0)
    }

    /// Handle an unexpected stream write error explicitly (`YS-10`).
    public func handleStreamFailure(stream: String, message: String) {
        guard phase == .recording, let dir = captureDirURL else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()
        finalizeAudioWavHeader(at: dir.appendingPathComponent("audio.wav"), pcmByteCount: audioPCMByteCount)
        provisionalGrade = "fail"
        provisionalReasons = ["stream_write_failed: \(stream) (\(message))"]
        writeSessionJSON(in: dir, status: "failed", stopReason: "write_error", failedStream: stream)
        phase = .failed
        statusBanner = "Capture halted — write failed on \(stream): \(message)"
    }

    /// Handle system interruption or backgrounding without silently claiming a complete recording (`YS-08`).
    public func handleInterruption(stopReason: String) {
        guard phase == .recording, let dir = captureDirURL else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()
        finalizeAudioWavHeader(at: dir.appendingPathComponent("audio.wav"), pcmByteCount: audioPCMByteCount)
        provisionalGrade = "fail"
        provisionalReasons = ["recording_interrupted: \(stopReason)"]
        writeSessionJSON(in: dir, status: "stopped", stopReason: stopReason, failedStream: nil)

        let writer = assetWriter
        let vInput = videoWriterInput
        writerQueue.async {
            if let writer, writer.status == .writing {
                vInput?.markAsFinished()
                let sema = DispatchSemaphore(value: 0)
                writer.finishWriting { sema.signal() }
                _ = sema.wait(timeout: .now() + 5.0)
            }
        }
        phase = .completed
        statusBanner = "Capture interrupted (\(stopReason)) — saved partial recording"
    }

    /// Cleanly stop the recording, finish `rgb.mp4`, hash all streams (`Q022`), write `manifest.json`, and build the store-mode `.zip` (`YS-21`, `YS-22`).
    public func stopAndSealCapture() {
        guard let dir = captureDirURL, phase == .recording else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        logEvent(kind: "done", detail: "User tapped Done")
        phase = .sealing
        statusBanner = "Sealing manifest.json and building capture archive…"

        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()

        finalizeAudioWavHeader(at: dir.appendingPathComponent("audio.wav"), pcmByteCount: audioPCMByteCount)
        updateProvisionalGrade()
        writeSessionJSON(in: dir, status: "stopped", stopReason: "user_done", failedStream: nil)

        let writer = assetWriter
        let vInput = videoWriterInput
        writerQueue.async { [weak self] in
            guard let self else { return }
            let sema = DispatchSemaphore(value: 0)
            if let writer, writer.status == .writing {
                vInput?.markAsFinished()
                writer.finishWriting {
                    sema.signal()
                }
                _ = sema.wait(timeout: .now() + 10.0)
            }
            do {
                let zipURL = try self.sealManifestAndPackZip(in: dir)
                Task { @MainActor in
                    self.archiveZipURL = zipURL
                    self.phase = .completed
                    self.statusBanner = "Capture sealed — Grade \(self.provisionalGrade)"
                }
            } catch {
                Task { @MainActor in
                    self.phase = .failed
                    self.statusBanner = "Seal error: \(error.localizedDescription)"
                }
            }
        }
    }

    private func updateProvisionalGrade() {
        let gapM = loopClosureGapMeters
        let primaryResiduals = tapeChecks.compactMap { item -> Double? in
            guard (item["role"] as? String) == "primary" else { return nil }
            return item["residual_m"] as? Double
        }
        let maxPrimaryRes = primaryResiduals.max() ?? 0.0
        let loopGapPerM = (gapM ?? 0.08) / 25.0
        let rss = sqrt(0.015 * 0.015 + (loopGapPerM * 10.0) * (loopGapPerM * 10.0) + 0.005 * 0.005)
        let distTol10m = max(maxPrimaryRes, 1.96 * rss)

        var band = 3
        let gVal = gapM ?? 0.0
        if distTol10m <= 0.05 && gVal <= 0.05 {
            band = 3
        } else if distTol10m <= 0.10 && gVal <= 0.12 {
            band = 2
        } else if distTol10m <= 0.20 && gVal <= 0.25 {
            band = 1
        } else {
            band = 0
        }

        var reasons: [String] = [String(format: "distance_tolerance_10m_m=%.3fm", distTol10m)]
        if !detectedMarkerIDs.contains(0) {
            band = min(band, 1)
            reasons.append("cap: no_marker_0 -> C")
        }
        if gapM == nil {
            band = min(band, 1)
            reasons.append("cap: no_loop_closure -> C")
        }
        if primaryResiduals.isEmpty {
            band = min(band, 1)
            reasons.append("cap: no_tape_check -> C")
        }
        let labels = [3: "A", 2: "B", 1: "C", 0: "fail"]
        provisionalGrade = labels[band] ?? "C"
        provisionalReasons = reasons
    }

    private func initializeCSVFiles(in dir: URL) throws {
        try "t,saved_index,dropped,drop_reason,tracking_state,tracking_reason\n"
            .write(to: dir.appendingPathComponent("frames.csv"), atomically: true, encoding: .utf8)
        try "t,saved_index,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height\n"
            .write(to: dir.appendingPathComponent("poses.csv"), atomically: true, encoding: .utf8)
        try "t,saved_index,file,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height\n"
            .write(to: dir.appendingPathComponent("stills.csv"), atomically: true, encoding: .utf8)
        try "t,saved_index,marker_id,size_mm_printed,size_mm_measured,x,y,z,qx,qy,qz,qw,c0_u,c0_v,c1_u,c1_v,c2_u,c2_v,c3_u,c3_v\n"
            .write(to: dir.appendingPathComponent("markers.csv"), atomically: true, encoding: .utf8)
        try "t,saved_index,kind,detail\n"
            .write(to: dir.appendingPathComponent("events.csv"), atomically: true, encoding: .utf8)
    }

    private func initializeAudioWavHeader(at url: URL) throws {
        var header = Data(count: 44)
        header.replaceSubrange(0..<4, with: "RIFF".data(using: .ascii)!)
        header.replaceSubrange(8..<12, with: "WAVE".data(using: .ascii)!)
        header.replaceSubrange(12..<16, with: "fmt ".data(using: .ascii)!)
        var fmtChunkSize: UInt32 = 16
        var audioFormat: UInt16 = 1  // PCM
        var numChannels: UInt16 = 1
        var sampleRate: UInt32 = 16000
        var byteRate: UInt32 = 32000
        var blockAlign: UInt16 = 2
        var bitsPerSample: UInt16 = 16
        withUnsafeBytes(of: &fmtChunkSize) { header.replaceSubrange(16..<20, with: $0) }
        withUnsafeBytes(of: &audioFormat) { header.replaceSubrange(20..<22, with: $0) }
        withUnsafeBytes(of: &numChannels) { header.replaceSubrange(22..<24, with: $0) }
        withUnsafeBytes(of: &sampleRate) { header.replaceSubrange(24..<28, with: $0) }
        withUnsafeBytes(of: &byteRate) { header.replaceSubrange(28..<32, with: $0) }
        withUnsafeBytes(of: &blockAlign) { header.replaceSubrange(32..<34, with: $0) }
        withUnsafeBytes(of: &bitsPerSample) { header.replaceSubrange(34..<36, with: $0) }
        header.replaceSubrange(36..<40, with: "data".data(using: .ascii)!)
        try header.write(to: url, options: .atomic)
    }

    private func finalizeAudioWavHeader(at url: URL, pcmByteCount: UInt32) {
        do {
            let fh = try FileHandle(forUpdating: url)
            var riffSize: UInt32 = 36 + pcmByteCount
            var dataSize: UInt32 = pcmByteCount
            try fh.seek(toOffset: 4)
            try fh.write(contentsOf: Data(bytes: &riffSize, count: 4))
            try fh.seek(toOffset: 40)
            try fh.write(contentsOf: Data(bytes: &dataSize, count: 4))
            try fh.close()
        } catch {
            handleStreamFailure(stream: "audio", message: error.localizedDescription)
        }
    }

    private func appendString(_ text: String, to fileURL: URL, stream: String) {
        guard let data = text.data(using: .utf8) else { return }
        do {
            let fh = try FileHandle(forWritingTo: fileURL)
            try fh.seekToEnd()
            try fh.write(contentsOf: data)
            try fh.close()
        } catch {
            handleStreamFailure(stream: stream, message: error.localizedDescription)
        }
    }

    private func writeSessionJSON(in dir: URL, status: String, stopReason: String?, failedStream: String?) {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let nowUTC = formatter.string(from: Date())
        let t0 = t0MonotonicS ?? latestTimestampS
        let tEnd = max(t0, latestTimestampS)
        let audioT0 = audioFirstSampleTS ?? t0

        var printedMap: [String: Double] = [:]
        for (k, v) in markerSizesMM {
            printedMap[String(k)] = v
        }

        var gapsList: [[String: Any]] = []
        if let gap = loopClosureGapMeters,
           let tFirst = marker0FirstPassTimeS,
           let tLast = marker0LastPassTimeS {
            gapsList.append([
                "marker_id": 0,
                "first_t_s": tFirst,
                "last_t_s": tLast,
                "gap_m": gap,
            ])
        }

        let stopPairAny: Any = (status == "stopped") ? ["t_s": tEnd, "utc": nowUTC] : NSNull()
        let stopReasonAny: Any = stopReason ?? NSNull()
        let failedStreamAny: Any = failedStream ?? NSNull()
        let latAny: Any = latestLocation?.coordinate.latitude ?? NSNull()
        let lonAny: Any = latestLocation?.coordinate.longitude ?? NSNull()
        let hAccAny: Any = latestLocation?.horizontalAccuracy ?? NSNull()
        let trueHAny: Any = latestTrueHeadingDeg ?? NSNull()
        let magHAny: Any = latestMagneticHeadingDeg ?? NSNull()
        let headAccAny: Any = latestHeadingAccuracyDeg ?? NSNull()
        let loopGapAny: Any = loopClosureGapMeters ?? NSNull()

        let sessionObj: [String: Any] = [
            "contract": "yardstick-capture/1",
            "capture_id": captureID,
            "device": [
                "model": Self.hardwareModelIdentifier,
                "name": UIDevice.current.name,
                "ios_version": UIDevice.current.systemVersion,
                "lidar_supported": Self.isLiDARSupported,
            ],
            "app_version": "0.1.0",
            "clock": [
                "started_at_utc": startedAtUTC,
                "t0_monotonic_s": t0,
                "start_clock_pair": ["t_s": t0, "utc": startedAtUTC],
                "stop_clock_pair": stopPairAny,
                "audio_first_sample_t_s": audioT0,
                "audio_sample_rate_hz": 16000,
            ],
            "camera_format": [
                "requested": ["width": requestedWidth, "height": requestedHeight, "fps": requestedFPS],
                "actual": ["width": actualWidth, "height": actualHeight, "fps": actualFPS],
            ],
            "pose_convention": "arkit_right_handed_y_up",
            "state": [
                "status": status,
                "last_heartbeat_t_s": tEnd,
                "stop_reason": stopReasonAny,
                "failed_stream": failedStreamAny,
            ],
            "location": [
                "status": locationAuthStatus,
                "latitude_deg": latAny,
                "longitude_deg": lonAny,
                "horizontal_accuracy_m": hAccAny,
                "true_heading_deg": trueHAny,
                "magnetic_heading_deg": magHAny,
                "heading_accuracy_deg": headAccAny,
                "north_samples": northSamples,
            ],
            "marker_config": [
                "dictionary": "DICT_5X5_250",
                "printed_sizes_mm": printedMap,
            ],
            "guide_thresholds": [
                "pitch_min_deg": -45.0,
                "pitch_max_deg": -15.0,
                "roll_max_abs_deg": 15.0,
            ],
            "loop_closure_gaps": gapsList,
            "tape_checks": tapeChecks,
            "provisional_grade": [
                "grade": provisionalGrade == "—" ? "C" : provisionalGrade,
                "distance_tolerance_10m_m": 0.05,
                "loop_closure_gap_m": loopGapAny,
                "reasons": provisionalReasons,
            ],
        ]

        do {
            let data = try JSONSerialization.data(withJSONObject: sessionObj, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: dir.appendingPathComponent("session.json"), options: .atomic)
        } catch {
            // Avoid infinite recursion if session.json itself fails
            phase = .failed
            statusBanner = "Failed to write session.json: \(error.localizedDescription)"
        }
    }

    nonisolated private func sealManifestAndPackZip(in dir: URL) throws -> URL {
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])
        var filesArray: [[String: Any]] = []

        while let fileURL = enumerator?.nextObject() as? URL {
            let rel = fileURL.path.replacingOccurrences(of: dir.path + "/", with: "")
            if rel == "manifest.json" || rel.hasPrefix("standard/") || rel.hasPrefix(".") {
                continue
            }
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard vals.isRegularFile == true else { continue }
            let data = try Data(contentsOf: fileURL)
            let digest = SHA256.hash(data: data)
            let hex = digest.map { String(format: "%02x", $0) }.joined()
            filesArray.append([
                "path": rel,
                "size_bytes": data.count,
                "sha256": hex,
            ])
        }
        filesArray.sort { ($0["path"] as? String ?? "") < ($1["path"] as? String ?? "") }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let manifestObj: [String: Any] = [
            "contract": "yardstick-capture/1",
            "capture_id": dir.lastPathComponent,
            "sealed_at_utc": formatter.string(from: Date()),
            "streams_declared": [
                "session", "frames", "rgb_video", "depth", "confidence",
                "poses", "audio", "events", "stills", "markers",
            ],
            "files": filesArray,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestObj, options: [.prettyPrinted, .sortedKeys])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"), options: .atomic)

        let zipURL = dir.deletingLastPathComponent().appendingPathComponent("\(dir.lastPathComponent).zip")
        try Self.writeStoreModeZipArchive(sourceDir: dir, destinationZipURL: zipURL)
        return zipURL
    }

    /// Encode 16-bit depth millimetres (`[UInt16]`) into a 16-bit grayscale PNG (`YS-04`).
    @discardableResult
    nonisolated public static func writeDepth16BitPNG(_ pixels: [UInt16], width: Int, height: Int, to url: URL) -> Bool {
        let bigEndian = pixels.map { $0.bigEndian }
        let data = bigEndian.withUnsafeBytes { Data($0) } as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2),
              let cgImg = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 16,
                  bitsPerPixel: 16,
                  bytesPerRow: width * 2,
                  space: cs,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Big.rawValue),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            return false
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        return CGImageDestinationFinalize(dest)
    }

    /// Encode 8-bit LiDAR confidence levels (`0, 1, 2`) into an 8-bit grayscale PNG (`YS-04`).
    @discardableResult
    nonisolated public static func writeConfidence8BitPNG(_ pixels: [UInt8], width: Int, height: Int, to url: URL) -> Bool {
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2),
              let cgImg = CGImage(
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
              ),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            return false
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        return CGImageDestinationFinalize(dest)
    }

    /// Copy an NV12 `CVPixelBuffer` synchronously into a newly allocated buffer so `ARFrame` is never retained (`YS-02`).
    nonisolated private static func clonePixelBuffer(_ src: CVPixelBuffer, pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)
        var dstOpt: CVPixelBuffer?
        if let pool {
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &dstOpt)
        }
        if dstOpt == nil {
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                format,
                [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
                &dstOpt
            )
        }
        guard let dst = dstOpt else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        let planeCount = CVPixelBufferGetPlaneCount(src)
        if planeCount > 0 {
            for plane in 0..<planeCount {
                let hPlane = CVPixelBufferGetHeightOfPlane(src, plane)
                let srcBPR = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dstBPR = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
                let copyBytes = min(srcBPR, dstBPR)
                guard let srcBase = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let dstBase = CVPixelBufferGetBaseAddressOfPlane(dst, plane)
                else { continue }
                for r in 0..<hPlane {
                    memcpy(dstBase.advanced(by: r * dstBPR), srcBase.advanced(by: r * srcBPR), copyBytes)
                }
            }
        } else if let srcBase = CVPixelBufferGetBaseAddress(src),
                  let dstBase = CVPixelBufferGetBaseAddress(dst) {
            let srcBPR = CVPixelBufferGetBytesPerRow(src)
            let dstBPR = CVPixelBufferGetBytesPerRow(dst)
            let copyBytes = min(srcBPR, dstBPR)
            for r in 0..<height {
                memcpy(dstBase.advanced(by: r * dstBPR), srcBase.advanced(by: r * srcBPR), copyBytes)
            }
        }
        return dst
    }

    /// Extract tightly-packed 8-bit luma plane (`width * height` bytes) from a bi-planar YpCbCr pixel buffer.
    nonisolated private static func extractLumaBytes(from pb: CVPixelBuffer) -> [UInt8]? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let width = CVPixelBufferGetWidthOfPlane(pb, 0)
        let height = CVPixelBufferGetHeightOfPlane(pb, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        guard width > 0, height > 0, let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else {
            return nil
        }
        var out = [UInt8](repeating: 0, count: width * height)
        out.withUnsafeMutableBytes { dstRaw in
            guard let dstBase = dstRaw.baseAddress else { return }
            for r in 0..<height {
                memcpy(dstBase.advanced(by: r * width), base.advanced(by: r * bpr), width)
            }
        }
        return out
    }

    /// Convert arbitrary `CMSampleBuffer` audio (e.g. 48 kHz or 44.1 kHz Float32/Int16) to 16,000 Hz mono Int16 PCM (`YS-06`).
    nonisolated public static func extractAndResampleTo16kHzMonoPCM(_ sampleBuffer: CMSampleBuffer) -> Data? {
        guard let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(fmtDesc),
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer)
        else {
            return nil
        }
        let asbd = asbdPtr.pointee
        let inRate = asbd.mSampleRate
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let bits = Int(asbd.mBitsPerChannel)
        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
        let numFrames = CMSampleBufferGetNumSamples(sampleBuffer)
        guard inRate > 0, numFrames > 0 else { return nil }

        var totalLength = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &totalLength,
            dataPointerOut: &dataPointer
        )
        guard status == kCMBlockBufferNoErr, let rawPtr = dataPointer, totalLength > 0 else {
            return nil
        }

        // 1. Decode to mono Float32 in [-1.0, 1.0]
        var monoFloats = [Float](repeating: 0.0, count: numFrames)
        let strideChannels = isNonInterleaved ? 1 : channels

        if isFloat && bits == 32 {
            let fPtr = UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Float.self)
            let maxSamples = totalLength / MemoryLayout<Float>.size
            for i in 0..<numFrames {
                let baseIdx = i * strideChannels
                if baseIdx < maxSamples {
                    monoFloats[i] = fPtr[baseIdx]
                }
            }
        } else if !isFloat && bits == 16 {
            let iPtr = UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Int16.self)
            let maxSamples = totalLength / MemoryLayout<Int16>.size
            for i in 0..<numFrames {
                let baseIdx = i * strideChannels
                if baseIdx < maxSamples {
                    monoFloats[i] = Float(Int16(littleEndian: iPtr[baseIdx])) / 32768.0
                }
            }
        } else if !isFloat && bits == 32 {
            let iPtr = UnsafeRawPointer(rawPtr).assumingMemoryBound(to: Int32.self)
            let maxSamples = totalLength / MemoryLayout<Int32>.size
            for i in 0..<numFrames {
                let baseIdx = i * strideChannels
                if baseIdx < maxSamples {
                    monoFloats[i] = Float(Int32(littleEndian: iPtr[baseIdx])) / 2147483648.0
                }
            }
        } else {
            return nil
        }

        // 2. Resample from `inRate` to 16,000 Hz mono Int16 with anti-aliasing box filter
        let outRate = 16000.0
        let ratio = inRate / outRate
        let outFrames = max(1, Int((Double(numFrames) / ratio).rounded()))
        var outPCM = [Int16](repeating: 0, count: outFrames)

        for oIdx in 0..<outFrames {
            let srcStart = Double(oIdx) * ratio
            let srcEnd = min(Double(numFrames), srcStart + max(1.0, ratio))
            let i0 = min(numFrames - 1, max(0, Int(srcStart.rounded(.down))))
            let i1 = min(numFrames - 1, max(i0, Int(srcEnd.rounded(.down))))
            var acc: Float = 0.0
            var cnt: Float = 0.0
            for k in i0...i1 {
                acc += monoFloats[k]
                cnt += 1.0
            }
            let sampleF = (cnt > 0) ? (acc / cnt) : monoFloats[i0]
            let scaled = max(-32768.0, min(32767.0, Double(sampleF) * 32767.0))
            outPCM[oIdx] = Int16(scaled.rounded()).littleEndian
        }

        return outPCM.withUnsafeBytes { Data($0) }
    }

    /// Pure-Swift ZIP_STORED (`compressionMethod = 0`) archiver so LiDAR PNGs & MP4 are packaged without re-compression (`YS-22`).
    nonisolated public static func writeStoreModeZipArchive(sourceDir: URL, destinationZipURL: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destinationZipURL.path) {
            try fm.removeItem(at: destinationZipURL)
        }
        fm.createFile(atPath: destinationZipURL.path, contents: nil)
        let fh = try FileHandle(forWritingTo: destinationZipURL)
        defer { try? fh.close() }

        let baseFolderName = sourceDir.lastPathComponent
        let enumerator = fm.enumerator(at: sourceDir, includingPropertiesForKeys: [.isRegularFileKey])
        var relPaths: [String] = []
        while let fileURL = enumerator?.nextObject() as? URL {
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            if vals.isRegularFile == true {
                let rel = fileURL.path.replacingOccurrences(of: sourceDir.path + "/", with: "")
                if !rel.hasPrefix(".") {
                    relPaths.append(rel)
                }
            }
        }
        relPaths.sort()

        struct CentralRecord {
            let nameData: Data
            let crc32: UInt32
            let size: UInt32
            let localHeaderOffset: UInt32
        }
        var centralRecords: [CentralRecord] = []
        var offset: UInt32 = 0

        for rel in relPaths {
            let fileURL = sourceDir.appendingPathComponent(rel)
            let fileData = try Data(contentsOf: fileURL)
            let arcName = "\(baseFolderName)/\(rel)"
            let nameData = arcName.data(using: .utf8) ?? Data()
            let crc = Self.crc32(fileData)
            let size = UInt32(fileData.count)

            var localHeader = Data()
            Self.appendUInt32(&localHeader, 0x04034b50)
            Self.appendUInt16(&localHeader, 20)
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt16(&localHeader, 0)  // Store mode
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt32(&localHeader, crc)
            Self.appendUInt32(&localHeader, size)
            Self.appendUInt32(&localHeader, size)
            Self.appendUInt16(&localHeader, UInt16(nameData.count))
            Self.appendUInt16(&localHeader, 0)
            localHeader.append(nameData)

            try fh.write(contentsOf: localHeader)
            try fh.write(contentsOf: fileData)

            centralRecords.append(
                CentralRecord(
                    nameData: nameData,
                    crc32: crc,
                    size: size,
                    localHeaderOffset: offset
                )
            )
            offset += UInt32(localHeader.count) + size
        }

        let centralStartOffset = offset
        var centralDirSize: UInt32 = 0

        for rec in centralRecords {
            var cd = Data()
            Self.appendUInt32(&cd, 0x02014b50)
            Self.appendUInt16(&cd, 20)
            Self.appendUInt16(&cd, 20)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)  // Store mode
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt32(&cd, rec.crc32)
            Self.appendUInt32(&cd, rec.size)
            Self.appendUInt32(&cd, rec.size)
            Self.appendUInt16(&cd, UInt16(rec.nameData.count))
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt32(&cd, 0)
            Self.appendUInt32(&cd, rec.localHeaderOffset)
            cd.append(rec.nameData)

            try fh.write(contentsOf: cd)
            centralDirSize += UInt32(cd.count)
        }

        var eocd = Data()
        Self.appendUInt32(&eocd, 0x06054b50)
        Self.appendUInt16(&eocd, 0)
        Self.appendUInt16(&eocd, 0)
        Self.appendUInt16(&eocd, UInt16(centralRecords.count))
        Self.appendUInt16(&eocd, UInt16(centralRecords.count))
        Self.appendUInt32(&eocd, centralDirSize)
        Self.appendUInt32(&eocd, centralStartOffset)
        Self.appendUInt16(&eocd, 0)
        try fh.write(contentsOf: eocd)
    }

    nonisolated private static func appendUInt16(_ data: inout Data, _ val: UInt16) {
        var le = val.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    nonisolated private static func appendUInt32(_ data: inout Data, _ val: UInt32) {
        var le = val.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    nonisolated private static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                if (crc & 1) != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320
                } else {
                    crc >>= 1
                }
            }
        }
        return ~crc
    }
}

#if canImport(ARKit)
extension CaptureSessionRecorder: ARSessionDelegate {
    nonisolated public func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor in
            self.handleInterruption(stopReason: "interrupted")
        }
    }

    nonisolated public func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in
            self.handleStreamFailure(stream: "poses", message: error.localizedDescription)
        }
    }

    nonisolated public func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        let pts = CMSampleBufferGetPresentationTimeStamp(audioSampleBuffer).seconds
        guard let pcm16kData = Self.extractAndResampleTo16kHzMonoPCM(audioSampleBuffer),
              !pcm16kData.isEmpty
        else {
            return
        }

        Task { @MainActor in
            guard self.phase == .recording, let dir = self.captureDirURL else { return }
            if self.audioFirstSampleTS == nil {
                self.audioFirstSampleTS = pts
            }
            let wavURL = dir.appendingPathComponent("audio.wav")
            do {
                let fh = try FileHandle(forWritingTo: wavURL)
                try fh.seekToEnd()
                try fh.write(contentsOf: pcm16kData)
                try fh.close()
                self.audioPCMByteCount += UInt32(pcm16kData.count)
            } catch {
                self.handleStreamFailure(stream: "audio", message: error.localizedDescription)
            }
        }
    }

    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let timestampS = frame.timestamp
        let cTransform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        let imgRes = frame.camera.imageResolution

        // Compute grip-invariant pitch from optical look ray (-Z_c dot +Y_w, YS-05)
        let lookRayY = Double(-cTransform.columns.2.y)
        let clampedY = min(1.0, max(-1.0, lookRayY))
        let pitchDegrees = asin(clampedY) * 180.0 / .pi

        // Portrait roll relative to world up +Y_w: in portrait orientation, screen-up is -X_c
        let upInCamX = Double(cTransform.columns.0.y)
        let upInCamY = Double(cTransform.columns.1.y)
        let rollDegrees = atan2(upInCamY, -upInCamX) * 180.0 / .pi

        // Camera horizontal yaw in world XZ plane for North calibration at Marker 0 (YS-14, Q020)
        let lookRayX = Double(-cTransform.columns.2.x)
        let lookRayZ = Double(-cTransform.columns.2.z)
        let yawWorldDeg = atan2(-lookRayZ, lookRayX) * 180.0 / .pi

        let trackingStr: String
        let reasonStr: String
        switch frame.camera.trackingState {
        case .normal:
            trackingStr = "normal"
            reasonStr = ""
        case .notAvailable:
            trackingStr = "not_available"
            reasonStr = "not_available"
        case .limited(let r):
            trackingStr = "limited"
            switch r {
            case .initializing: reasonStr = "initializing"
            case .excessiveMotion: reasonStr = "excessive_motion"
            case .insufficientFeatures: reasonStr = "insufficient_features"
            case .relocalizing: reasonStr = "relocalizing"
            @unknown default: reasonStr = "unknown"
            }
        }

        // Synchronously copy RGB pixel buffer and depth/confidence buffers in < 2 ms so `ARFrame` is NEVER retained!
        let copiedRGBBuffer = Self.clonePixelBuffer(frame.capturedImage, pool: nil)

        var centerDepthM: Double? = nil
        var centerWorldPt: SIMD3<Double>? = nil
        var depthCopyMM: [UInt16]? = nil
        var confCopyU8: [UInt8]? = nil
        var depthW = 0
        var depthH = 0

        let cMat = simd_double4x4(
            SIMD4<Double>(cTransform.columns.0),
            SIMD4<Double>(cTransform.columns.1),
            SIMD4<Double>(cTransform.columns.2),
            SIMD4<Double>(cTransform.columns.3)
        )

        if let sceneDepth = frame.sceneDepth, let confPB = sceneDepth.confidenceMap {
            let depthPB = sceneDepth.depthMap
            CVPixelBufferLockBaseAddress(depthPB, .readOnly)
            CVPixelBufferLockBaseAddress(confPB, .readOnly)
            depthW = CVPixelBufferGetWidth(depthPB)
            depthH = CVPixelBufferGetHeight(depthPB)
            let dRowBytes = CVPixelBufferGetBytesPerRow(depthPB)
            let cRowBytes = CVPixelBufferGetBytesPerRow(confPB)

            if let dBase = CVPixelBufferGetBaseAddress(depthPB),
               let cBase = CVPixelBufferGetBaseAddress(confPB) {
                var dArr = [UInt16](repeating: 0, count: depthW * depthH)
                var cArr = [UInt8](repeating: 0, count: depthW * depthH)
                for v in 0..<depthH {
                    let dRow = dBase.advanced(by: v * dRowBytes).assumingMemoryBound(to: Float32.self)
                    let cRow = cBase.advanced(by: v * cRowBytes).assumingMemoryBound(to: UInt8.self)
                    let baseIdx = v * depthW
                    for u in 0..<depthW {
                        let z = dRow[u]
                        if z.isFinite && z > 0.0 && z < 65.535 {
                            dArr[baseIdx + u] = UInt16(clamping: Int((z * 1000.0).rounded()))
                        }
                        cArr[baseIdx + u] = min(2, cRow[u])
                    }
                }
                depthCopyMM = dArr
                confCopyU8 = cArr

                // 5x5 center crosshair depth patch (YS-40)
                var validDepths: [UInt16] = []
                let uC = depthW / 2
                let vC = depthH / 2
                for dv in -2...2 {
                    let v = min(depthH - 1, max(0, vC + dv))
                    for du in -2...2 {
                        let u = min(depthW - 1, max(0, uC + du))
                        let mm = dArr[v * depthW + u]
                        if mm >= 200 && mm <= 4500 {
                            validDepths.append(mm)
                        }
                    }
                }
                if validDepths.count >= 13 {
                    validDepths.sort()
                    let medZ = Double(validDepths[validDepths.count / 2]) / 1000.0
                    centerDepthM = medZ
                    let pCam = SIMD4<Double>(0.0, 0.0, -medZ, 1.0)
                    let pWorld = cMat * pCam
                    centerWorldPt = SIMD3<Double>(pWorld.x, pWorld.y, pWorld.z)
                }
            }
            CVPixelBufferUnlockBaseAddress(confPB, .readOnly)
            CVPixelBufferUnlockBaseAddress(depthPB, .readOnly)
        }

        Task { @MainActor in
            guard self.phase == .recording, let dir = self.captureDirURL else { return }

            // Major 11: Record start_clock_pair at the FIRST delivered ARFrame
            if self.t0MonotonicS == nil {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime]
                self.startedAtUTC = formatter.string(from: Date())
                self.t0MonotonicS = timestampS
            }

            // Major 14: Detect upstream frame timestamp gaps (> 1.6x frame period) and log dropped rows (YS-02)
            let expectedDt = 1.0 / Double(max(1, self.actualFPS))
            if let prevT = self.lastDeliveredFrameTimestampS {
                let dt = timestampS - prevT
                if dt > 1.6 * expectedDt {
                    let missedCount = min(30, max(1, Int((dt / expectedDt).rounded()) - 1))
                    for k in 1...missedCount {
                        let interpT = prevT + Double(k) * (dt / Double(missedCount + 1))
                        self.droppedFrameCount += 1
                        let gapLine = String(
                            format: "%.6f,,true,upstream_frame_drop,%@,%@\n",
                            interpT, trackingStr, reasonStr
                        )
                        self.appendString(gapLine, to: dir.appendingPathComponent("frames.csv"), stream: "frames")
                    }
                }
            }
            self.lastDeliveredFrameTimestampS = timestampS

            self.latestTimestampS = timestampS
            self.latestCameraTransform = cMat
            self.pitchDeg = pitchDegrees
            self.rollDeg = rollDegrees
            self.trackingStateText = trackingStr
            self.crosshairDepthMeters = centerDepthM
            self.latestCrosshairWorld = centerWorldPt

            // Track camera excursion from Marker 0 Pass 1 for multi-pass loop closure separation
            let camPosW = SIMD3<Double>(cMat.columns.3.x, cMat.columns.3.y, cMat.columns.3.z)
            if let m0Pos = self.marker0FirstPassPos {
                let distFromM0 = simd_length(camPosW - m0Pos)
                if distFromM0 > self.marker0MaxExcursionM {
                    self.marker0MaxExcursionM = distFromM0
                }
            }

            // Sample North calibration pairs while at Marker 0 start (spaced >= 0.15 s) (YS-14, Q020)
            let nearM0Start: Bool = {
                guard let m0Pos = self.marker0FirstPassPos else { return false }
                return simd_length(camPosW - m0Pos) <= 2.5
            }()
            if nearM0Start,
               self.northSamples.count < 15,
               timestampS - self.lastNorthSampleTimeS >= 0.15,
               let trueH = self.latestTrueHeadingDeg,
               let acc = self.latestHeadingAccuracyDeg,
               acc >= 0.0 && acc <= 25.0 {
                self.lastNorthSampleTimeS = timestampS
                self.northSamples.append([
                    "t_s": timestampS,
                    "camera_yaw_world_deg": yawWorldDeg,
                    "true_heading_deg": trueH,
                    "heading_accuracy_deg": acc,
                ])
                self.northSampleCount = self.northSamples.count
            }

            guard let dPixels = depthCopyMM,
                  let cPixels = confCopyU8,
                  let rgbPB = copiedRGBBuffer
            else {
                self.droppedFrameCount += 1
                let dropLine = String(format: "%.6f,,true,missing_depth_buffer,%@,%@\n", timestampS, trackingStr, reasonStr)
                self.appendString(dropLine, to: dir.appendingPathComponent("frames.csv"), stream: "frames")
                return
            }

            // Backpressure guard so writerQueue never drops a saved_index frame (YS-02)
            if self.pendingWriterJobs > 8 || (self.videoWriterInput?.isReadyForMoreMediaData == false && self.videoSessionStarted) {
                self.droppedFrameCount += 1
                let dropLine = String(format: "%.6f,,true,writer_queue_full,%@,%@\n", timestampS, trackingStr, reasonStr)
                self.appendString(dropLine, to: dir.appendingPathComponent("frames.csv"), stream: "frames")
                return
            }

            let sIdx = self.savedFrameCount
            self.latestSavedIndex = sIdx
            self.savedFrameCount += 1
            self.pendingWriterJobs += 1

            let fLine = String(format: "%.6f,%d,false,,%@,%@\n", timestampS, sIdx, trackingStr, reasonStr)
            self.appendString(fLine, to: dir.appendingPathComponent("frames.csv"), stream: "frames")

            // Normalize quaternion before writing so poses.csv is guaranteed unit-norm within 1e-7 (YS-23)
            let rawQuat = simd_quatd(cMat)
            let normQuat = simd_normalize(rawQuat)
            let fx = Double(intrinsics.columns.0.x)
            let fy = Double(intrinsics.columns.1.y)
            let cx = Double(intrinsics.columns.2.x)
            let cy = Double(intrinsics.columns.2.y)
            let rgbW = Int(imgRes.width)
            let rgbH = Int(imgRes.height)

            let pLine = String(
                format: "%.6f,%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                timestampS, sIdx,
                camPosW.x, camPosW.y, camPosW.z,
                normQuat.imag.x, normQuat.imag.y, normQuat.imag.z, normQuat.real,
                fx, fy, cx, cy, rgbW, rgbH
            )
            self.appendString(pLine, to: dir.appendingPathComponent("poses.csv"), stream: "poses")

            let shouldRunArUco = (sIdx % 3 == 0) || self.pendingForceMarkerDetect
            self.pendingForceMarkerDetect = false
            let lastStillT = self.lastStillSavedTimeS
            let shouldConsiderPeriodicStill = (timestampS - lastStillT) >= 2.0
            let sizesSnapshot = self.markerSizesMM

            let depthURL = dir.appendingPathComponent(String(format: "depth/%06d.png", sIdx))
            let confURL = dir.appendingPathComponent(String(format: "confidence/%06d.png", sIdx))
            let wD = depthW
            let hD = depthH
            let writer = self.assetWriter
            let vInput = self.videoWriterInput
            let adaptor = self.pixelBufferAdaptor
            let needStartSession = !self.videoSessionStarted
            if needStartSession {
                self.videoSessionStarted = true
            }
            let ciCtx = self.ciContext

            self.writerQueue.async { [weak self] in
                guard let self else { return }

                // 1. Append frame to rgb.mp4 via AVAssetWriterInputPixelBufferAdaptor (Blocker 1)
                let pts = CMTime(seconds: timestampS, preferredTimescale: 600)
                var videoOK = true
                var writerErrMsg = "AVAssetWriter append failed"
                if let writer, let vInput, let adaptor, writer.status == .writing {
                    if needStartSession {
                        writer.startSession(atSourceTime: pts)
                    }
                    if vInput.isReadyForMoreMediaData {
                        videoOK = adaptor.append(rgbPB, withPresentationTime: pts)
                        if !videoOK, let err = writer.error {
                            writerErrMsg = err.localizedDescription
                        }
                    } else {
                        videoOK = false
                    }
                } else {
                    videoOK = false
                    if let err = writer?.error {
                        writerErrMsg = err.localizedDescription
                    }
                }

                // 2. Write 16-bit depth PNG and 8-bit confidence PNG (Blocker 2)
                let depthOK = Self.writeDepth16BitPNG(dPixels, width: wD, height: hD, to: depthURL)
                let confOK = Self.writeConfidence8BitPNG(cPixels, width: wD, height: hD, to: confURL)

                // 3. Run ArUco detector & periodic sharp still selector (Blocker 5 & Major 15)
                var detected3D: [DetectedMarkerSighting] = []
                var saveStillNow = false

                if shouldRunArUco || shouldConsiderPeriodicStill {
                    if let luma = Self.extractLumaBytes(from: rgbPB) {
                        if shouldRunArUco {
                            let dets2D = ArUcoDetector.detectMarkers2D(lumaBytes: luma, width: rgbW, height: rgbH)
                            for d2 in dets2D {
                                let pSize = sizesSnapshot[d2.markerID] ?? 250.0
                                if let s3D = ArUcoDetector.measureMarker3DOnLiDARPlane(
                                    markerID: d2.markerID,
                                    cornersUV: d2.cornersUV,
                                    printedSizeMM: pSize,
                                    depthMapMM: dPixels,
                                    confidenceMap: cPixels,
                                    depthWidth: wD,
                                    depthHeight: hD,
                                    rgbWidth: rgbW,
                                    rgbHeight: rgbH,
                                    fx: fx,
                                    fy: fy,
                                    cx: cx,
                                    cy: cy,
                                    cameraToWorld: cMat
                                ) {
                                    detected3D.append(s3D)
                                }
                            }
                        }
                        if !detected3D.isEmpty {
                            saveStillNow = true
                        } else if shouldConsiderPeriodicStill {
                            let sharpness = ArUcoDetector.computeLaplacianSharpness(lumaBytes: luma, width: rgbW, height: rgbH)
                            if sharpness >= 12.0 || (timestampS - lastStillT) >= 3.0 {
                                saveStillNow = true
                            }
                        }
                    }
                }

                var stillSavedOK = true
                let stillRelPath = String(format: "stills/%06d.jpg", sIdx)
                if saveStillNow {
                    let stillURL = dir.appendingPathComponent(stillRelPath)
                    let ciImg = CIImage(cvPixelBuffer: rgbPB)
                    let cs = CGColorSpaceCreateDeviceRGB()
                    do {
                        try ciCtx.writeJPEGRepresentation(
                            of: ciImg,
                            to: stillURL,
                            colorSpace: cs,
                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.92]
                        )
                    } catch {
                        stillSavedOK = false
                    }
                }

                let finalVideoOK = videoOK
                let finalWriterErr = writerErrMsg
                let finalSaveStill = saveStillNow
                let finalStillOK = stillSavedOK
                let finalDetected3D = detected3D

                Task { @MainActor in
                    self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                    guard self.phase == .recording else { return }

                    if !finalVideoOK {
                        self.handleStreamFailure(stream: "rgb_video", message: finalWriterErr)
                        return
                    }
                    if !depthOK {
                        self.handleStreamFailure(stream: "depth", message: "Failed to write \(depthURL.lastPathComponent)")
                        return
                    }
                    if !confOK {
                        self.handleStreamFailure(stream: "confidence", message: "Failed to write \(confURL.lastPathComponent)")
                        return
                    }
                    if !finalStillOK {
                        self.handleStreamFailure(stream: "stills", message: "Failed to write \(stillRelPath)")
                        return
                    }

                    if finalSaveStill && !self.savedStillsIndices.contains(sIdx) {
                        self.savedStillsIndices.insert(sIdx)
                        self.lastStillSavedTimeS = timestampS
                        let sLine = String(
                            format: "%.6f,%d,%@,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                            timestampS, sIdx, stillRelPath,
                            camPosW.x, camPosW.y, camPosW.z,
                            normQuat.imag.x, normQuat.imag.y, normQuat.imag.z, normQuat.real,
                            fx, fy, cx, cy, rgbW, rgbH
                        )
                        self.appendString(sLine, to: dir.appendingPathComponent("stills.csv"), stream: "stills")
                    }

                    for m in finalDetected3D {
                        self.detectedMarkerIDs.insert(m.markerID)
                        let mq = m.quaternionXYZW
                        let c = m.cornersUV
                        let mLine = String(
                            format: "%.6f,%d,%d,%.2f,%.3f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                            timestampS, sIdx, m.markerID,
                            m.sizeMMPrinted, m.sizeMMMeasured,
                            m.positionWorld.x, m.positionWorld.y, m.positionWorld.z,
                            mq.x, mq.y, mq.z, mq.w,
                            c[0].x, c[0].y, c[1].x, c[1].y, c[2].x, c[2].y, c[3].x, c[3].y
                        )
                        self.appendString(mLine, to: dir.appendingPathComponent("markers.csv"), stream: "markers")

                        if m.markerID == 0 {
                            if self.marker0FirstPassPos == nil {
                                self.marker0FirstPassPos = m.positionWorld
                                self.marker0FirstPassTimeS = timestampS
                                self.statusBanner = "Marker 0 Origin locked — walk yard & return to Marker 0 at end"
                            } else if let firstT = self.marker0FirstPassTimeS,
                                      let firstPos = self.marker0FirstPassPos,
                                      (timestampS - firstT) >= 3.0,
                                      (self.marker0MaxExcursionM >= 1.0 || (timestampS - firstT) >= 10.0) {
                                self.marker0LastPassPos = m.positionWorld
                                self.marker0LastPassTimeS = timestampS
                                let gap = simd_length(m.positionWorld - firstPos)
                                self.loopClosureGapMeters = gap
                                self.updateProvisionalGrade()
                                self.statusBanner = String(format: "Marker 0 Loop Closed (gap %.1f cm) — tap Done", gap * 100.0)
                            }
                        }
                    }
                }
            }

            if timestampS - self.lastHeartbeatS >= 5.0 {
                self.lastHeartbeatS = timestampS
                self.writeSessionJSON(in: dir, status: "recording", stopReason: nil, failedStream: nil)
            }
        }
    }
}
#endif

extension CaptureSessionRecorder: CLLocationManagerDelegate {
    nonisolated public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        Task { @MainActor in
            switch status {
            case .authorizedWhenInUse, .authorizedAlways:
                self.locationAuthStatus = "ok"
            case .denied, .restricted:
                self.locationAuthStatus = "denied"
            default:
                self.locationAuthStatus = "unavailable"
            }
        }
    }

    nonisolated public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        Task { @MainActor in
            self.latestLocation = loc
        }
    }

    nonisolated public func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let trueH = newHeading.trueHeading >= 0 ? newHeading.trueHeading : nil
        let magH = newHeading.magneticHeading
        let acc = newHeading.headingAccuracy
        Task { @MainActor in
            self.latestTrueHeadingDeg = trueH
            self.latestMagneticHeadingDeg = magH
            self.latestHeadingAccuracyDeg = acc
            if acc > 25.0 && self.northSamples.count < 10 {
                self.locationAuthStatus = "accuracy_exceeded"
            } else if trueH != nil && acc >= 0 && acc <= 25.0 {
                self.locationAuthStatus = "ok"
            }
        }
    }
}
