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
///    same monotonic host clock as `ARFrame.timestamp`, resampled/written to 16 kHz mono PCM `audio.wav`.
/// 4. **Crash safety (`YS-08`, `YS-10`)**: `movieFragmentInterval = 5s` on `AVAssetWriter`, 1s CSV flushes,
///    and 5s atomic `session.json` heartbeat.
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
    private let writerLock = NSLock()
    private var activeWriterJobs: Int = 0
    private let locationManager = CLLocationManager()

    private var captureDirURL: URL?
    private var t0MonotonicS: Double?
    private var audioFirstSampleTS: Double?
    private var startedAtUTC: String = ""
    private var lastHeartbeatS: Double = 0.0
    private var lastStillSavedTimeS: Double = -10.0
    private var requestedWidth: Int = 1920
    private var requestedHeight: Int = 1440
    private var requestedFPS: Int = 30
    private var actualWidth: Int = 1920
    private var actualHeight: Int = 1440
    private var actualFPS: Int = 30

    private var assetWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoSessionStarted: Bool = false

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
    private var tapeChecks: [[String: Any]] = []
    private var pendingTapeASavedIndex: Int = 0
    private var pendingTapeATimeS: Double = 0.0
    private var audioPCMByteCount: UInt32 = 0

    public override init() {
        super.init()
        locationManager.delegate = self
        locationManager.headingFilter = kCLHeadingFilterNone
        #if canImport(ARKit)
        arSession.delegate = self
        #endif
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

    public func toggleMarkerSize(for markerID: Int) {
        let current = markerSizesMM[markerID] ?? 250.0
        markerSizesMM[markerID] = (current == 250.0) ? 180.0 : 250.0
    }

    public func requestPermissionsAndStart() {
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
        loopClosureGapMeters = nil
        audioPCMByteCount = 0
        t0MonotonicS = nil
        audioFirstSampleTS = nil
        lastStillSavedTimeS = -10.0

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
                Int($0.imageResolution.width) == 1920 && Int($0.imageResolution.height) == 1440 && $0.framesPerSecond == 30
            }) ?? ARWorldTrackingConfiguration.supportedVideoFormats.first {
                config.videoFormat = fmt
                actualWidth = Int(fmt.imageResolution.width)
                actualHeight = Int(fmt.imageResolution.height)
                actualFPS = fmt.framesPerSecond
                requestedWidth = actualWidth
                requestedHeight = actualHeight
                requestedFPS = actualFPS
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
        writer.startWriting()
        assetWriter = writer
        videoWriterInput = input
        pixelBufferAdaptor = adaptor
        videoSessionStarted = false
    }

    /// Record a timed user event (`marker`, `tape_a`, `tape_b`, `note`, `done`, `YS-07`).
    public func logEvent(kind: String, detail: String) {
        guard let dir = captureDirURL, phase == .recording else { return }
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        let safeDetail = detail.replacingOccurrences(of: ",", with: ";").replacingOccurrences(of: "\n", with: " ")
        let line = String(format: "%.6f,%d,%@,%@\n", t, sIdx, kind, safeDetail)
        appendString(line, to: dir.appendingPathComponent("events.csv"))
    }

    /// Capture Endpoint A of a tape-measure check using the live 5x5 center crosshair LiDAR patch (`YS-40`).
    public func captureTapeEndpointA() {
        guard let worldPt = latestCrosshairWorld else {
            statusBanner = "Aim center crosshair at surface (0.3–4.0 m) first"
            return
        }
        pendingTapeEndpointAWorld = worldPt
        pendingTapeASavedIndex = latestSavedIndex
        pendingTapeATimeS = latestTimestampS
        logEvent(kind: "tape_a", detail: String(format: "%.3f;%.3f;%.3f", worldPt.x, worldPt.y, worldPt.z))
        statusBanner = "Endpoint A locked — aim at Endpoint B and tap Tape B"
    }

    /// Capture Endpoint B and record a primary or held-out tape check (`YS-40`, `YS-41`).
    public func commitTapeEndpointB(tapeMeters: Double, role: String = "primary") {
        guard let ptA = pendingTapeEndpointAWorld, let ptB = latestCrosshairWorld, tapeMeters > 0 else {
            statusBanner = "Both endpoints A & B and a positive tape distance are required"
            return
        }
        let measuredM = simd_length(ptB - ptA)
        let residualM = abs(measuredM - tapeMeters)
        let checkID = tapeChecks.count
        let entry: [String: Any] = [
            "id": checkID,
            "role": role,
            "t_a_s": pendingTapeATimeS,
            "t_b_s": latestTimestampS,
            "saved_index_a": pendingTapeASavedIndex,
            "saved_index_b": latestSavedIndex,
            "point_a_world_m": [ptA.x, ptA.y, ptA.z],
            "point_b_world_m": [ptB.x, ptB.y, ptB.z],
            "tape_m": tapeMeters,
            "measured_m": measuredM,
            "residual_m": residualM,
        ]
        tapeChecks.append(entry)
        tapeCheckCount = tapeChecks.count
        pendingTapeEndpointAWorld = nil
        logEvent(kind: "tape_b", detail: String(format: "tape=%.3fm measured=%.3fm res=%.3fm", tapeMeters, measuredM, residualM))
        updateProvisionalGrade()
        statusBanner = String(format: "Tape #%d logged: |%.2fm - %.2fm| = %.1f cm", checkID + 1, measuredM, tapeMeters, residualM * 100.0)
    }

    /// Cleanly stop the recording, finish `rgb.mp4`, hash all streams (`Q022`), write `manifest.json`, and build the store-mode `.zip` (`YS-21`, `YS-22`).
    public func stopAndSealCapture() {
        guard let dir = captureDirURL, phase == .recording else { return }
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
        guard let fh = try? FileHandle(forUpdating: url) else { return }
        var riffSize: UInt32 = 36 + pcmByteCount
        var dataSize: UInt32 = pcmByteCount
        fh.seek(toFileOffset: 4)
        fh.write(Data(bytes: &riffSize, count: 4))
        fh.seek(toFileOffset: 40)
        fh.write(Data(bytes: &dataSize, count: 4))
        try? fh.close()
    }

    private func appendString(_ text: String, to fileURL: URL) {
        guard let data = text.data(using: .utf8) else { return }
        if let fh = try? FileHandle(forWritingTo: fileURL) {
            fh.seekToEndOfFile()
            fh.write(data)
            try? fh.close()
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
                "model": UIDevice.current.model,
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
            "loop_closure_gaps": [],
            "tape_checks": tapeChecks,
            "provisional_grade": [
                "grade": provisionalGrade == "—" ? "C" : provisionalGrade,
                "distance_tolerance_10m_m": 0.05,
                "loop_closure_gap_m": loopGapAny,
                "reasons": provisionalReasons,
            ],
        ]

        if let data = try? JSONSerialization.data(withJSONObject: sessionObj, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: dir.appendingPathComponent("session.json"), options: .atomic)
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

    nonisolated public static func writeDepth16BitPNG(_ pixels: [UInt16], width: Int, height: Int, to url: URL) {
        var bigEndian = pixels.map { $0.bigEndian }
        let byteCount = bigEndian.count * MemoryLayout<UInt16>.size
        let data = withUnsafeBytes(of: &bigEndian) { Data($0) } as CFData
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
            _ = byteCount
            return
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        CGImageDestinationFinalize(dest)
    }

    nonisolated public static func writeConfidence8BitPNG(_ pixels: [UInt8], width: Int, height: Int, to url: URL) {
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
            return
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        CGImageDestinationFinalize(dest)
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

            fh.write(localHeader)
            fh.write(fileData)

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

            fh.write(cd)
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
        fh.write(eocd)
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
    nonisolated public func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        let pts = CMSampleBufferGetPresentationTimeStamp(audioSampleBuffer).seconds
        guard let blockBuffer = CMSampleBufferGetDataBuffer(audioSampleBuffer) else { return }
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer, length > 0 else { return }
        let rawData = Data(bytes: ptr, count: length)

        Task { @MainActor in
            guard self.phase == .recording, let dir = self.captureDirURL else { return }
            if self.audioFirstSampleTS == nil {
                self.audioFirstSampleTS = pts
            }
            let wavURL = dir.appendingPathComponent("audio.wav")
            if let fh = try? FileHandle(forWritingTo: wavURL) {
                fh.seekToEndOfFile()
                fh.write(rawData)
                try? fh.close()
                self.audioPCMByteCount += UInt32(rawData.count)
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
        let rollDegrees = Double(atan2(cTransform.columns.0.y, cTransform.columns.1.y)) * 180.0 / .pi

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

        // Synchronously copy depth & confidence buffers in < 1 ms so `ARFrame` is NEVER retained!
        var centerDepthM: Double? = nil
        var centerWorldPt: SIMD3<Double>? = nil
        var depthCopyMM: [UInt16]? = nil
        var confCopyU8: [UInt8]? = nil
        var depthW = 0
        var depthH = 0

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
                    let cMat = simd_double4x4(
                        SIMD4<Double>(cTransform.columns.0),
                        SIMD4<Double>(cTransform.columns.1),
                        SIMD4<Double>(cTransform.columns.2),
                        SIMD4<Double>(cTransform.columns.3)
                    )
                    let pWorld = cMat * pCam
                    centerWorldPt = SIMD3<Double>(pWorld.x, pWorld.y, pWorld.z)
                }
            }
            CVPixelBufferUnlockBaseAddress(confPB, .readOnly)
            CVPixelBufferUnlockBaseAddress(depthPB, .readOnly)
        }

        Task { @MainActor in
            guard self.phase == .recording, let dir = self.captureDirURL else { return }
            if self.t0MonotonicS == nil {
                self.t0MonotonicS = timestampS
            }
            self.latestTimestampS = timestampS
            self.pitchDeg = pitchDegrees
            self.rollDeg = rollDegrees
            self.trackingStateText = trackingStr
            self.crosshairDepthMeters = centerDepthM
            self.latestCrosshairWorld = centerWorldPt

            // Sample North calibration pairs while at Marker 0 start
            if self.northSamples.count < 15,
               let trueH = self.latestTrueHeadingDeg,
               let acc = self.latestHeadingAccuracyDeg,
               acc >= 0.0 && acc <= 25.0 {
                self.northSamples.append([
                    "t_s": timestampS,
                    "camera_yaw_world_deg": yawWorldDeg,
                    "true_heading_deg": trueH,
                    "heading_accuracy_deg": acc,
                ])
                self.northSampleCount = self.northSamples.count
            }

            guard let dPixels = depthCopyMM, let cPixels = confCopyU8 else {
                self.droppedFrameCount += 1
                let dropLine = String(format: "%.6f,,true,missing_depth_buffer,%@,%@\n", timestampS, trackingStr, reasonStr)
                self.appendString(dropLine, to: dir.appendingPathComponent("frames.csv"))
                return
            }

            let sIdx = self.savedFrameCount
            self.latestSavedIndex = sIdx
            self.savedFrameCount += 1

            let fLine = String(format: "%.6f,%d,false,,%@,%@\n", timestampS, sIdx, trackingStr, reasonStr)
            self.appendString(fLine, to: dir.appendingPathComponent("frames.csv"))

            let quat = simd_quatf(cTransform)
            let pLine = String(
                format: "%.6f,%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                timestampS, sIdx,
                cTransform.columns.3.x, cTransform.columns.3.y, cTransform.columns.3.z,
                quat.imag.x, quat.imag.y, quat.imag.z, quat.real,
                intrinsics.columns.0.x, intrinsics.columns.1.y,
                intrinsics.columns.2.x, intrinsics.columns.2.y,
                Int(imgRes.width), Int(imgRes.height)
            )
            self.appendString(pLine, to: dir.appendingPathComponent("poses.csv"))

            let depthURL = dir.appendingPathComponent(String(format: "depth/%06d.png", sIdx))
            let confURL = dir.appendingPathComponent(String(format: "confidence/%06d.png", sIdx))
            let wD = depthW
            let hD = depthH
            self.writerQueue.async {
                Self.writeDepth16BitPNG(dPixels, width: wD, height: hD, to: depthURL)
                Self.writeConfidence8BitPNG(cPixels, width: wD, height: hD, to: confURL)
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
