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

/// Sendable wrapper for a cloned `CVPixelBuffer` transferred strictly from the ARFrame callback to `writerQueue`.
private struct SendablePixelBuffer: @unchecked Sendable {
    let buffer: CVPixelBuffer
}

/// Sendable wrapper for an audio `CMSampleBuffer` transferred strictly from the ARSession audio callback to `writerQueue`.
private struct SendableAudioSampleBuffer: @unchecked Sendable {
    let buffer: CMSampleBuffer
}

/// Serial-queue disk writer engine confined strictly to `writerQueue` so no file I/O ever runs on `@MainActor`
/// and non-Sendable AVFoundation/FileHandle objects never cross concurrency domains (Blockers 3, 4, 5, 6).
private final class CaptureWriterEngine: @unchecked Sendable {
    let captureDirURL: URL
    let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    var assetWriter: AVAssetWriter?
    var videoWriterInput: AVAssetWriterInput?
    var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    var videoSessionStarted: Bool = false

    var framesFH: FileHandle?
    var posesFH: FileHandle?
    var stillsFH: FileHandle?
    var markersFH: FileHandle?
    var eventsFH: FileHandle?
    var audioFH: FileHandle?

    var audioConverter: AVAudioConverter?
    var audioInputFormat: AVAudioFormat?
    let audioOutputFormat: AVAudioFormat? = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16000.0,
        channels: 1,
        interleaved: true
    )

    var savedFrameCount: Int = 0
    var droppedFrameCount: Int = 0
    var audioPCMByteCount: UInt32 = 0
    var audioFirstSampleTS: Double? = nil
    var lastStillSavedTimeS: Double = -10.0
    var savedStillsIndices: Set<Int> = []
    var isClosed: Bool = false

    init(captureDirURL: URL, width: Int, height: Int) throws {
        self.captureDirURL = captureDirURL
        let fm = FileManager.default
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("depth"), withIntermediateDirectories: true)
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("confidence"), withIntermediateDirectories: true)
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("stills"), withIntermediateDirectories: true)

        try Self.createFileWithHeader(
            "t,saved_index,dropped,drop_reason,tracking_state,tracking_reason\n",
            at: captureDirURL.appendingPathComponent("frames.csv")
        )
        try Self.createFileWithHeader(
            "t,saved_index,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height\n",
            at: captureDirURL.appendingPathComponent("poses.csv")
        )
        try Self.createFileWithHeader(
            "t,saved_index,file,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height\n",
            at: captureDirURL.appendingPathComponent("stills.csv")
        )
        try Self.createFileWithHeader(
            "t,saved_index,marker_id,size_mm_printed,size_mm_measured,x,y,z,qx,qy,qz,qw,c0_u,c0_v,c1_u,c1_v,c2_u,c2_v,c3_u,c3_v\n",
            at: captureDirURL.appendingPathComponent("markers.csv")
        )
        try Self.createFileWithHeader(
            "t,saved_index,kind,detail\n",
            at: captureDirURL.appendingPathComponent("events.csv")
        )
        try Self.initializeAudioWavHeader(at: captureDirURL.appendingPathComponent("audio.wav"))

        self.framesFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("frames.csv"))
        try self.framesFH?.seekToEnd()
        self.posesFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("poses.csv"))
        try self.posesFH?.seekToEnd()
        self.stillsFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("stills.csv"))
        try self.stillsFH?.seekToEnd()
        self.markersFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("markers.csv"))
        try self.markersFH?.seekToEnd()
        self.eventsFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("events.csv"))
        try self.eventsFH?.seekToEnd()
        self.audioFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("audio.wav"))
        try self.audioFH?.seekToEnd()

        let videoURL = captureDirURL.appendingPathComponent("rgb.mp4")
        let writer = try AVAssetWriter(outputURL: videoURL, fileType: .mp4)
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
        guard writer.canAdd(input) else {
            throw NSError(domain: "Yardstick", code: -2, userInfo: [NSLocalizedDescriptionKey: "Cannot add video input to AVAssetWriter"])
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "Yardstick", code: -1, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter startWriting failed"])
        }
        self.assetWriter = writer
        self.videoWriterInput = input
        self.pixelBufferAdaptor = adaptor
    }

    private static func createFileWithHeader(_ header: String, at url: URL) throws {
        guard let data = header.data(using: .utf8) else {
            throw NSError(domain: "Yardstick", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to encode CSV header as UTF-8"])
        }
        try data.write(to: url, options: .atomic)
    }

    private static func initializeAudioWavHeader(at url: URL) throws {
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

    func appendLine(_ line: String, to fh: FileHandle?) throws {
        guard let data = line.data(using: .utf8) else {
            throw NSError(domain: "Yardstick", code: -4, userInfo: [NSLocalizedDescriptionKey: "UTF-8 string encoding returned nil"])
        }
        guard let fh else {
            throw NSError(domain: "Yardstick", code: -5, userInfo: [NSLocalizedDescriptionKey: "Stream FileHandle is nil"])
        }
        try fh.write(contentsOf: data)
    }

    /// Convert `CMSampleBuffer` audio to 16,000 Hz mono Int16 PCM using stateful `AVAudioConverter` (`YS-06`).
    func appendAudioSampleBuffer(_ sampleBuffer: CMSampleBuffer) throws {
        guard let outFmt = audioOutputFormat else {
            throw NSError(domain: "Yardstick", code: -10, userInfo: [NSLocalizedDescriptionKey: "16 kHz output AVAudioFormat unavailable"])
        }
        guard let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer) else {
            throw NSError(domain: "Yardstick", code: -11, userInfo: [NSLocalizedDescriptionKey: "CMSampleBuffer missing audio format description"])
        }
        let inFmt = AVAudioFormat(cmAudioFormatDescription: fmtDesc)
        let numFrames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard numFrames > 0, inFmt.sampleRate > 0 else {
            throw NSError(domain: "Yardstick", code: -12, userInfo: [NSLocalizedDescriptionKey: "Empty or zero-rate audio CMSampleBuffer"])
        }

        if audioConverter == nil || audioInputFormat != inFmt {
            guard let conv = AVAudioConverter(from: inFmt, to: outFmt) else {
                throw NSError(domain: "Yardstick", code: -13, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter from \(inFmt) to 16 kHz mono Int16"])
            }
            audioConverter = conv
            audioInputFormat = inFmt
        }
        guard let conv = audioConverter else {
            throw NSError(domain: "Yardstick", code: -14, userInfo: [NSLocalizedDescriptionKey: "AVAudioConverter unavailable"])
        }

        guard let inPCMBuffer = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: numFrames) else {
            throw NSError(domain: "Yardstick", code: -15, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate input AVAudioPCMBuffer"])
        }
        inPCMBuffer.frameLength = numFrames
        let copyStatus = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(numFrames),
            into: inPCMBuffer.mutableAudioBufferList
        )
        guard copyStatus == noErr else {
            throw NSError(domain: "Yardstick", code: Int(copyStatus), userInfo: [NSLocalizedDescriptionKey: "CMSampleBufferCopyPCMDataIntoAudioBufferList failed (\(copyStatus))"])
        }

        let ratio = outFmt.sampleRate / inFmt.sampleRate
        let outCapacity = AVAudioFrameCount(ceil(Double(numFrames) * ratio) + 32)
        guard let outPCMBuffer = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: outCapacity) else {
            throw NSError(domain: "Yardstick", code: -16, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate output AVAudioPCMBuffer"])
        }

        var providedInput = false
        var convError: NSError?
        let status = conv.convert(to: outPCMBuffer, error: &convError) { _, outStatus in
            if providedInput {
                outStatus.pointee = .noDataNow
                return nil
            }
            providedInput = true
            outStatus.pointee = .haveData
            return inPCMBuffer
        }
        if status == .error {
            throw convError ?? NSError(domain: "Yardstick", code: -17, userInfo: [NSLocalizedDescriptionKey: "AVAudioConverter conversion error"])
        }

        let outFrames = Int(outPCMBuffer.frameLength)
        if outFrames > 0, let int16Ptr = outPCMBuffer.int16ChannelData?[0] {
            let byteCount = outFrames * MemoryLayout<Int16>.size
            let pcmData = Data(bytes: int16Ptr, count: byteCount)
            guard let aFH = audioFH else {
                throw NSError(domain: "Yardstick", code: -18, userInfo: [NSLocalizedDescriptionKey: "audio.wav FileHandle is closed"])
            }
            try aFH.write(contentsOf: pcmData)
            audioPCMByteCount &+= UInt32(byteCount)
        }
    }

    /// Finalize `audio.wav` RIFF/data chunk sizes, finish `rgb.mp4`, and close all open file handles.
    /// Does NOT call `handleStreamFailure` so it can never cause recursive failure loops (Blocker 6).
    func finishStreamsAndCloseFiles() throws {
        guard !isClosed else { return }
        isClosed = true

        try? framesFH?.close()
        try? posesFH?.close()
        try? stillsFH?.close()
        try? markersFH?.close()
        try? eventsFH?.close()

        var audioFinalizeErr: Error?
        if let aFH = audioFH {
            do {
                var riffSize: UInt32 = (36 &+ audioPCMByteCount).littleEndian
                var dataSize: UInt32 = audioPCMByteCount.littleEndian
                try aFH.seek(toOffset: 4)
                try aFH.write(contentsOf: withUnsafeBytes(of: &riffSize) { Data($0) })
                try aFH.seek(toOffset: 40)
                try aFH.write(contentsOf: withUnsafeBytes(of: &dataSize) { Data($0) })
                try aFH.close()
            } catch {
                audioFinalizeErr = error
            }
        }
        audioFH = nil

        if let writer = assetWriter, writer.status == .writing {
            if videoSessionStarted {
                videoWriterInput?.markAsFinished()
                let sema = DispatchSemaphore(value: 0)
                writer.finishWriting {
                    sema.signal()
                }
                _ = sema.wait(timeout: .now() + 15.0)
                if writer.status == .failed {
                    throw writer.error ?? NSError(domain: "Yardstick", code: -20, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter finishWriting failed"])
                }
            } else {
                writer.cancelWriting()
            }
        }

        if let audioFinalizeErr {
            throw audioFinalizeErr
        }
    }
}

/// Crash-safe, zero-`ARFrame`-retention LiDAR capture session recorder (`YS-00`–`YS-14`, `YS-21`, `YS-22`, `YS-30`–`YS-33`, `YS-40`–`YS-44`).
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
    @Published public private(set) var tapeCheckSummaries: [String] = []
    @Published public private(set) var loopClosureGapMeters: Double? = nil
    @Published public private(set) var provisionalGrade: String = "—"
    @Published public private(set) var provisionalReasons: [String] = []
    @Published public private(set) var archiveZipURL: URL? = nil

    private static let markerSizesDefaultsKey = "io.buildnexus.yardstick.markerSizesMM"

    /// Per-marker printed size configuration (`180.0` mm for printed set IDs `0–3`, `250.0` mm for `4–5`), persisted in `UserDefaults`.
    @Published public var markerSizesMM: [Int: Double] = CaptureSessionRecorder.loadPersistedMarkerSizes()

    public static func loadPersistedMarkerSizes(defaults: UserDefaults = .standard) -> [Int: Double] {
        var sizes: [Int: Double] = [
            0: 180.0, 1: 180.0, 2: 180.0, 3: 180.0, 4: 250.0, 5: 250.0,
        ]
        if let saved = defaults.dictionary(forKey: markerSizesDefaultsKey) {
            for mid in 0..<6 {
                if let num = saved[String(mid)] as? NSNumber {
                    let val = num.doubleValue
                    if val == 180.0 || val == 250.0 {
                        sizes[mid] = val
                    }
                }
            }
        }
        return sizes
    }

    private func persistMarkerSizes(defaults: UserDefaults = .standard) {
        var dict: [String: Double] = [:]
        for (k, v) in markerSizesMM {
            dict[String(k)] = v
        }
        defaults.set(dict, forKey: Self.markerSizesDefaultsKey)
    }

    #if canImport(ARKit)
    public let arSession = ARSession()
    #endif

    private let writerQueue = DispatchQueue(label: "io.yardstick.capture.writer", qos: .userInitiated)
    private let locationManager = CLLocationManager()
    private var notificationObservers: [NSObjectProtocol] = []
    private var writerEngine: CaptureWriterEngine?

    private var captureDirURL: URL?
    private var t0MonotonicS: Double?
    private var audioFirstSampleTS: Double?
    private var startedAtUTC: String = ""
    private var lastHeartbeatS: Double = 0.0
    private var lastNorthSampleTimeS: Double = -10.0
    private var lastDeliveredFrameTimestampS: Double?
    private var pendingForceMarkerDetect: Bool = false
    private var pendingWriterJobs: Int = 0

    private var requestedWidth: Int = 1920
    private var requestedHeight: Int = 1440
    private var requestedFPS: Int = 30
    private var actualWidth: Int = 1920
    private var actualHeight: Int = 1440
    private var actualFPS: Int = 30

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
    private var provisionalDistanceTolerance10m: Double = 0.05
    private var activeBackgroundTaskID: UIBackgroundTaskIdentifier = .invalid

    private func beginSealingBackgroundTask(name: String) {
        endSealingBackgroundTask()
        activeBackgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in
            Task { @MainActor in
                self?.endSealingBackgroundTask()
            }
        }
    }

    private func endSealingBackgroundTask() {
        if activeBackgroundTaskID != .invalid {
            UIApplication.shared.endBackgroundTask(activeBackgroundTaskID)
            activeBackgroundTaskID = .invalid
        }
    }

    /// Precomputed 256-entry IEEE 802.3 CRC-32 lookup table (`0xEDB88320`) for fast streaming ZIP packaging (Blocker 4).
    nonisolated private static let crc32Table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 {
            c = (c & 1 != 0) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1)
        }
        return c
    }

    public override init() {
        super.init()
        locationManager.delegate = self
        locationManager.headingFilter = kCLHeadingFilterNone
        #if canImport(ARKit)
        arSession.delegate = self
        #endif
        registerLifecycleObservers()
    }

    /// Exact hardware model identifier (e.g. `"iPhone18,2"`) via `uname` (`YS-00`).
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

    /// Observe only actual backgrounding (`didEnterBackground`) and audio/AR session interruptions,
    /// NOT `willResignActiveNotification`, so permission alerts, Control Center, and notification banners
    /// never abort the walk (Blocker 1).
    private func registerLifecycleObservers() {
        let nc = NotificationCenter.default
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
        notificationObservers = [bgObs, audioObs]
    }

    public func toggleMarkerSize(for markerID: Int) {
        let current = markerSizesMM[markerID] ?? 250.0
        markerSizesMM[markerID] = (current == 250.0) ? 180.0 : 250.0
        persistMarkerSizes()
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
        tapeCheckSummaries.removeAll()
        pendingTapeEndpointAWorld = nil
        pendingTapeEndpointBWorld = nil
        loopClosureGapMeters = nil
        marker0FirstPassPos = nil
        marker0LastPassPos = nil
        marker0FirstPassTimeS = nil
        marker0LastPassTimeS = nil
        marker0MaxExcursionM = 0.0
        t0MonotonicS = nil
        audioFirstSampleTS = nil
        lastNorthSampleTimeS = -10.0
        lastDeliveredFrameTimestampS = nil
        pendingWriterJobs = 0
        provisionalGrade = "—"
        provisionalReasons = []
        provisionalDistanceTolerance10m = 0.05
        archiveZipURL = nil

        #if canImport(ARKit) && !targetEnvironment(simulator)
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.providesAudioData = true
        config.environmentTexturing = .none
        config.isAutoFocusEnabled = true
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = [.sceneDepth]
        }

        // Explicitly select highest-resolution 4:3 format at 30 fps (or highest-resolution 30 fps format),
        // never falling back to a 60 fps `.first` format when 30 fps is supported (`YS-03`, Blocker 5).
        let supported = ARWorldTrackingConfiguration.supportedVideoFormats
        let formats30 = supported.filter { $0.framesPerSecond == 30 }
        let candidatePool = formats30.isEmpty ? supported : formats30
        let fourByThreePool = candidatePool.filter { fmt in
            let w = Int(fmt.imageResolution.width)
            let h = Int(fmt.imageResolution.height)
            return abs(w * 3 - h * 4) <= 8
        }
        let bestPool = fourByThreePool.isEmpty ? candidatePool : fourByThreePool
        let chosenFormat = bestPool.max { a, b in
            let areaA = Int(a.imageResolution.width) * Int(a.imageResolution.height)
            let areaB = Int(b.imageResolution.width) * Int(b.imageResolution.height)
            return areaA < areaB
        }

        if let fmt = chosenFormat {
            config.videoFormat = fmt
            requestedWidth = Int(fmt.imageResolution.width)
            requestedHeight = Int(fmt.imageResolution.height)
            requestedFPS = fmt.framesPerSecond
            actualWidth = requestedWidth
            actualHeight = requestedHeight
            actualFPS = requestedFPS
        }
        #endif

        do {
            let engine = try CaptureWriterEngine(captureDirURL: dir, width: actualWidth, height: actualHeight)
            self.writerEngine = engine

            #if canImport(ARKit) && !targetEnvironment(simulator)
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            #endif

            let snapData = makeSessionJSONData(status: "recording", stopReason: nil, failedStream: nil)
            writerQueue.async {
                Self.writeSessionJSONDataOnQueue(snapData, in: dir)
            }
            phase = .recording
            statusBanner = "Recording (\(actualWidth)×\(actualHeight)@\(actualFPS)fps) — hold on Marker 0"
        } catch {
            UIApplication.shared.isIdleTimerDisabled = false
            phase = .failed
            statusBanner = "Failed to initialize capture: \(error.localizedDescription)"
        }
    }

    /// Record a timed user event (`marker`, `tape_a`, `tape_b`, `note`, `done`, `YS-07`) on `writerQueue`.
    public func logEvent(kind: String, detail: String) {
        guard phase == .recording, let engine = writerEngine else { return }
        if kind == "marker" {
            pendingForceMarkerDetect = true
        }
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        let safeDetail = detail.replacingOccurrences(of: ",", with: ";").replacingOccurrences(of: "\n", with: " ")
        let line = String(format: "%.6f,%d,%@,%@\n", t, sIdx, kind, safeDetail)
        writerQueue.async { [weak self] in
            do {
                try engine.appendLine(line, to: engine.eventsFH)
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "events", message: error.localizedDescription)
                }
            }
        }
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
        tapeCheckSummaries.append(
            String(
                format: "Tape #%d (%@): tape %.2f m · phone %.2f m (diff %.1f cm)",
                checkID + 1, role, tapeMeters, measuredM, residualM * 100.0
            )
        )
        pendingTapeEndpointAWorld = nil
        pendingTapeEndpointBWorld = nil
        logEvent(kind: "tape_b", detail: String(format: "tape=%.3fm measured=%.3fm res=%.3fm", tapeMeters, measuredM, residualM))
        updateProvisionalGrade()
        statusBanner = String(format: "Tape #%d logged: |%.2fm - %.2fm| = %.1f cm", checkID + 1, measuredM, tapeMeters, residualM * 100.0)
    }

    /// Handle an unexpected stream write error explicitly without recursive failure loops (`YS-10`, Blocker 6).
    public func handleStreamFailure(stream: String, message: String) {
        guard phase == .recording || phase == .sealing, let dir = captureDirURL else { return }
        // Transition to `.failed` FIRST so no subsequent error callback can re-enter `handleStreamFailure`
        phase = .failed
        UIApplication.shared.isIdleTimerDisabled = false
        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()
        provisionalGrade = "fail"
        provisionalReasons = ["stream_write_failed: \(stream) (\(message))"]
        statusBanner = "Capture halted — write failed on \(stream): \(message)"

        let engine = writerEngine
        let snapData = makeSessionJSONData(status: "failed", stopReason: "write_error", failedStream: stream)
        writerQueue.async {
            try? engine?.finishStreamsAndCloseFiles()
            Self.writeSessionJSONDataOnQueue(snapData, in: dir)
        }
    }

    /// Handle backgrounding or session interruption by finishing streams, writing `status: "interrupted"`,
    /// and sealing `manifest.json` + `.zip` inside a UIKit background task (`YS-08`, Blocker 1).
    public func handleInterruption(stopReason: String) {
        guard phase == .recording, let dir = captureDirURL, let engine = writerEngine else { return }
        phase = .sealing
        UIApplication.shared.isIdleTimerDisabled = false
        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()
        provisionalGrade = "fail"
        provisionalReasons = ["recording_interrupted: \(stopReason)"]
        statusBanner = "Capture interrupted (\(stopReason)) — sealing partial recording…"

        beginSealingBackgroundTask(name: "YardstickSealInterrupted")
        let snapData = makeSessionJSONData(status: "interrupted", stopReason: stopReason, failedStream: nil)
        writerQueue.async { [weak self] in
            try? engine.finishStreamsAndCloseFiles()
            Self.writeSessionJSONDataOnQueue(snapData, in: dir)
            let zipURL = try? Self.sealManifestAndPackZip(in: dir)
            Task { @MainActor in
                if let self {
                    self.archiveZipURL = zipURL
                    self.phase = .completed
                    self.statusBanner = "Capture interrupted (\(stopReason)) — partial archive sealed"
                    self.endSealingBackgroundTask()
                }
            }
        }
    }

    /// Cleanly stop the recording, finish `rgb.mp4`, hash all streams (`Q022`), write `manifest.json`, and build the store-mode `.zip` (`YS-21`, `YS-22`).
    public func stopAndSealCapture() {
        guard let dir = captureDirURL, phase == .recording, let engine = writerEngine else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        logEvent(kind: "done", detail: "User tapped Done")
        phase = .sealing
        statusBanner = "Sealing manifest.json and building capture archive…"

        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()

        beginSealingBackgroundTask(name: "YardstickSealDone")
        updateProvisionalGrade()
        let snapData = makeSessionJSONData(status: "stopped", stopReason: "user_done", failedStream: nil)

        writerQueue.async { [weak self] in
            guard let self else { return }
            do {
                try engine.finishStreamsAndCloseFiles()
                Self.writeSessionJSONDataOnQueue(snapData, in: dir)
                let zipURL = try Self.sealManifestAndPackZip(in: dir)
                Task { @MainActor in
                    self.archiveZipURL = zipURL
                    self.phase = .completed
                    self.statusBanner = "Capture sealed — Grade \(self.provisionalGrade)"
                    self.endSealingBackgroundTask()
                }
            } catch {
                let errMsg = error.localizedDescription
                Task { @MainActor in
                    self.handleStreamFailure(stream: "manifest", message: errMsg)
                    self.endSealingBackgroundTask()
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
        provisionalDistanceTolerance10m = distTol10m

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

    private func makeSessionJSONDictionary(status: String, stopReason: String?, failedStream: String?) -> [String: Any] {
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

        let stopPairAny: Any = (status == "stopped" || status == "interrupted") ? ["t_s": tEnd, "utc": nowUTC] : NSNull()
        let stopReasonAny: Any = stopReason ?? NSNull()
        let failedStreamAny: Any = failedStream ?? NSNull()
        let latAny: Any = latestLocation?.coordinate.latitude ?? NSNull()
        let lonAny: Any = latestLocation?.coordinate.longitude ?? NSNull()
        let hAccAny: Any = latestLocation?.horizontalAccuracy ?? NSNull()
        let trueHAny: Any = latestTrueHeadingDeg ?? NSNull()
        let magHAny: Any = latestMagneticHeadingDeg ?? NSNull()
        let headAccAny: Any = latestHeadingAccuracyDeg ?? NSNull()
        let loopGapAny: Any = loopClosureGapMeters ?? NSNull()

        return [
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
                "distance_tolerance_10m_m": provisionalDistanceTolerance10m,
                "loop_closure_gap_m": loopGapAny,
                "reasons": provisionalReasons,
            ],
        ]
    }

    private func makeSessionJSONData(status: String, stopReason: String?, failedStream: String?) -> Data? {
        let dict = makeSessionJSONDictionary(status: status, stopReason: stopReason, failedStream: failedStream)
        return try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
    }

    nonisolated private static func writeSessionJSONDataOnQueue(_ data: Data?, in dir: URL) {
        guard let data else { return }
        try? data.write(to: dir.appendingPathComponent("session.json"), options: .atomic)
    }

    /// Stream-compute SHA-256, table-driven CRC-32, and exact byte size in 64 KB chunks so large videos
    /// are never loaded into memory (Blocker 4).
    nonisolated public static func streamHashAndCRC32(of fileURL: URL) throws -> (sha256: String, crc32: UInt32, sizeBytes: UInt64) {
        let fh = try FileHandle(forReadingFrom: fileURL)
        defer { try? fh.close() }
        var hasher = SHA256()
        var crc: UInt32 = 0xFFFFFFFF
        var totalSize: UInt64 = 0
        let table = Self.crc32Table

        while true {
            var shouldBreak = false
            var readErr: Error?
            autoreleasepool {
                do {
                    guard let chunk = try fh.read(upToCount: 65536), !chunk.isEmpty else {
                        shouldBreak = true
                        return
                    }
                    hasher.update(data: chunk)
                    totalSize += UInt64(chunk.count)
                    chunk.withUnsafeBytes { rawBuf in
                        for byte in rawBuf {
                            let idx = Int((crc ^ UInt32(byte)) & 0xFF)
                            crc = (crc >> 8) ^ table[idx]
                        }
                    }
                } catch {
                    readErr = error
                    shouldBreak = true
                }
            }
            if let readErr { throw readErr }
            if shouldBreak { break }
        }

        let digest = hasher.finalize()
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return (hex, ~crc, totalSize)
    }

    nonisolated private static func sealManifestAndPackZip(in dir: URL) throws -> URL {
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey])
        var filesArray: [[String: Any]] = []
        var precomputedCRCByRelPath: [String: (crc32: UInt32, sizeBytes: UInt64)] = [:]

        while let fileURL = enumerator?.nextObject() as? URL {
            let rel = fileURL.path.replacingOccurrences(of: dir.path + "/", with: "")
            if rel == "manifest.json" || rel.hasPrefix("standard/") || rel.hasPrefix(".") {
                continue
            }
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard vals.isRegularFile == true else { continue }
            let (hex, crc, sizeBytes) = try Self.streamHashAndCRC32(of: fileURL)
            filesArray.append([
                "path": rel,
                "size_bytes": sizeBytes,
                "sha256": hex,
            ])
            precomputedCRCByRelPath[rel] = (crc, sizeBytes)
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
        let manifestURL = dir.appendingPathComponent("manifest.json")
        let manifestData = try JSONSerialization.data(withJSONObject: manifestObj, options: [.prettyPrinted, .sortedKeys])
        try manifestData.write(to: manifestURL, options: .atomic)
        let (_, manifestCRC, manifestSize) = try Self.streamHashAndCRC32(of: manifestURL)
        precomputedCRCByRelPath["manifest.json"] = (manifestCRC, manifestSize)

        let zipURL = dir.deletingLastPathComponent().appendingPathComponent("\(dir.lastPathComponent).zip")
        try Self.writeStoreModeZipArchive(
            sourceDir: dir,
            destinationZipURL: zipURL,
            precomputedMeta: precomputedCRCByRelPath
        )
        return zipURL
    }

    /// Encode 16-bit depth millimetres (`[UInt16]`) into a 16-bit linear grayscale PNG (`YS-04`).
    @discardableResult
    nonisolated public static func writeDepth16BitPNG(_ pixels: [UInt16], width: Int, height: Int, to url: URL) -> Bool {
        let bigEndian = pixels.map { $0.bigEndian }
        let data = bigEndian.withUnsafeBytes { Data($0) } as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.linearGray),
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

    /// Encode 8-bit LiDAR confidence levels (`0, 1, 2`) into an 8-bit linear grayscale PNG (`YS-04`).
    @discardableResult
    nonisolated public static func writeConfidence8BitPNG(_ pixels: [UInt8], width: Int, height: Int, to url: URL) -> Bool {
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data),
              let cs = CGColorSpace(name: CGColorSpace.linearGray),
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

    /// Streaming, table-CRC, ZIP64-capable `ZIP_STORED` (`compressionMethod = 0`) archiver (`YS-22`, Blocker 4).
    nonisolated public static func writeStoreModeZipArchive(
        sourceDir: URL,
        destinationZipURL: URL,
        precomputedMeta: [String: (crc32: UInt32, sizeBytes: UInt64)] = [:]
    ) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destinationZipURL.path) {
            try fm.removeItem(at: destinationZipURL)
        }
        fm.createFile(atPath: destinationZipURL.path, contents: nil)
        let dstFH = try FileHandle(forWritingTo: destinationZipURL)
        defer { try? dstFH.close() }

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
            let size: UInt64
            let localHeaderOffset: UInt64
            let usesZip64: Bool
        }
        var centralRecords: [CentralRecord] = []
        var offset: UInt64 = 0

        for rel in relPaths {
            let fileURL = sourceDir.appendingPathComponent(rel)
            let meta: (crc32: UInt32, sizeBytes: UInt64)
            if let cached = precomputedMeta[rel] {
                meta = cached
            } else {
                let (_, crc, sz) = try Self.streamHashAndCRC32(of: fileURL)
                meta = (crc, sz)
            }
            let arcName = "\(baseFolderName)/\(rel)"
            let nameData = arcName.data(using: .utf8) ?? Data()
            let usesZip64 = (meta.sizeBytes >= 0xFFFFFFFF) || (offset >= 0xFFFFFFFF)

            var extraField = Data()
            if usesZip64 {
                Self.appendUInt16(&extraField, 0x0001)  // ZIP64 extended info tag
                Self.appendUInt16(&extraField, 16)      // 2 x UInt64 sizes
                Self.appendUInt64(&extraField, meta.sizeBytes)
                Self.appendUInt64(&extraField, meta.sizeBytes)
            }

            var localHeader = Data()
            Self.appendUInt32(&localHeader, 0x04034b50)
            Self.appendUInt16(&localHeader, usesZip64 ? 45 : 20)
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt16(&localHeader, 0)  // Store mode
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt16(&localHeader, 0)
            Self.appendUInt32(&localHeader, meta.crc32)
            Self.appendUInt32(&localHeader, usesZip64 ? 0xFFFFFFFF : UInt32(meta.sizeBytes))
            Self.appendUInt32(&localHeader, usesZip64 ? 0xFFFFFFFF : UInt32(meta.sizeBytes))
            Self.appendUInt16(&localHeader, UInt16(nameData.count))
            Self.appendUInt16(&localHeader, UInt16(extraField.count))
            localHeader.append(nameData)
            localHeader.append(extraField)

            try dstFH.write(contentsOf: localHeader)

            // Stream file contents in 64 KB chunks
            let srcFH = try FileHandle(forReadingFrom: fileURL)
            defer { try? srcFH.close() }
            while true {
                var done = false
                var copyErr: Error?
                autoreleasepool {
                    do {
                        guard let chunk = try srcFH.read(upToCount: 65536), !chunk.isEmpty else {
                            done = true
                            return
                        }
                        try dstFH.write(contentsOf: chunk)
                    } catch {
                        copyErr = error
                        done = true
                    }
                }
                if let copyErr { throw copyErr }
                if done { break }
            }

            centralRecords.append(
                CentralRecord(
                    nameData: nameData,
                    crc32: meta.crc32,
                    size: meta.sizeBytes,
                    localHeaderOffset: offset,
                    usesZip64: usesZip64
                )
            )
            offset += UInt64(localHeader.count) + meta.sizeBytes
        }

        let centralStartOffset = offset
        var centralDirSize: UInt64 = 0
        var anyZip64 = (centralStartOffset >= 0xFFFFFFFF) || (centralRecords.count >= 0xFFFF)

        for rec in centralRecords {
            if rec.usesZip64 { anyZip64 = true }
            var cdExtra = Data()
            if rec.usesZip64 {
                Self.appendUInt16(&cdExtra, 0x0001)
                Self.appendUInt16(&cdExtra, 24)
                Self.appendUInt64(&cdExtra, rec.size)
                Self.appendUInt64(&cdExtra, rec.size)
                Self.appendUInt64(&cdExtra, rec.localHeaderOffset)
            }

            var cd = Data()
            Self.appendUInt32(&cd, 0x02014b50)
            Self.appendUInt16(&cd, rec.usesZip64 ? 45 : 20)
            Self.appendUInt16(&cd, rec.usesZip64 ? 45 : 20)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)  // Store mode
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt32(&cd, rec.crc32)
            Self.appendUInt32(&cd, rec.usesZip64 ? 0xFFFFFFFF : UInt32(rec.size))
            Self.appendUInt32(&cd, rec.usesZip64 ? 0xFFFFFFFF : UInt32(rec.size))
            Self.appendUInt16(&cd, UInt16(rec.nameData.count))
            Self.appendUInt16(&cd, UInt16(cdExtra.count))
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt16(&cd, 0)
            Self.appendUInt32(&cd, 0)
            Self.appendUInt32(&cd, rec.usesZip64 ? 0xFFFFFFFF : UInt32(rec.localHeaderOffset))
            cd.append(rec.nameData)
            cd.append(cdExtra)

            try dstFH.write(contentsOf: cd)
            centralDirSize += UInt64(cd.count)
        }

        if anyZip64 || centralDirSize >= 0xFFFFFFFF {
            let zip64EOCDOffset = centralStartOffset + centralDirSize
            var z64EOCD = Data()
            Self.appendUInt32(&z64EOCD, 0x06064b50)
            Self.appendUInt64(&z64EOCD, 44)
            Self.appendUInt16(&z64EOCD, 45)
            Self.appendUInt16(&z64EOCD, 45)
            Self.appendUInt32(&z64EOCD, 0)
            Self.appendUInt32(&z64EOCD, 0)
            Self.appendUInt64(&z64EOCD, UInt64(centralRecords.count))
            Self.appendUInt64(&z64EOCD, UInt64(centralRecords.count))
            Self.appendUInt64(&z64EOCD, centralDirSize)
            Self.appendUInt64(&z64EOCD, centralStartOffset)
            try dstFH.write(contentsOf: z64EOCD)

            var z64Locator = Data()
            Self.appendUInt32(&z64Locator, 0x07064b50)
            Self.appendUInt32(&z64Locator, 0)
            Self.appendUInt64(&z64Locator, zip64EOCDOffset)
            Self.appendUInt32(&z64Locator, 1)
            try dstFH.write(contentsOf: z64Locator)
        }

        var eocd = Data()
        Self.appendUInt32(&eocd, 0x06054b50)
        Self.appendUInt16(&eocd, 0)
        Self.appendUInt16(&eocd, 0)
        let count16 = UInt16(min(0xFFFF, centralRecords.count))
        Self.appendUInt16(&eocd, count16)
        Self.appendUInt16(&eocd, count16)
        Self.appendUInt32(&eocd, UInt32(min(0xFFFFFFFF, centralDirSize)))
        Self.appendUInt32(&eocd, UInt32(min(0xFFFFFFFF, centralStartOffset)))
        Self.appendUInt16(&eocd, 0)
        try dstFH.write(contentsOf: eocd)
    }

    nonisolated private static func appendUInt16(_ data: inout Data, _ val: UInt16) {
        var le = val.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    nonisolated private static func appendUInt32(_ data: inout Data, _ val: UInt32) {
        var le = val.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    nonisolated private static func appendUInt64(_ data: inout Data, _ val: UInt64) {
        var le = val.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
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
        let sendableAudio = SendableAudioSampleBuffer(buffer: audioSampleBuffer)
        Task { @MainActor [weak self] in
            guard let self, self.phase == .recording, let engine = self.writerEngine else { return }
            if self.audioFirstSampleTS == nil {
                self.audioFirstSampleTS = pts
            }
            self.writerQueue.async { [weak self] in
                do {
                    try engine.appendAudioSampleBuffer(sendableAudio.buffer)
                } catch {
                    let errMsg = error.localizedDescription
                    Task { @MainActor [weak self] in
                        self?.handleStreamFailure(stream: "audio", message: errMsg)
                    }
                }
            }
        }
    }

    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let timestampS = frame.timestamp
        let cTransform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        let deliveredW = CVPixelBufferGetWidth(frame.capturedImage)
        let deliveredH = CVPixelBufferGetHeight(frame.capturedImage)

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
        let copiedRGBBufferOpt = Self.clonePixelBuffer(frame.capturedImage, pool: nil).map { SendablePixelBuffer(buffer: $0) }

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

        let finalDepthCopy = depthCopyMM
        let finalConfCopy = confCopyU8
        let finalDepthW = depthW
        let finalDepthH = depthH
        let finalCenterDepthM = centerDepthM
        let finalCenterWorldPt = centerWorldPt

        Task { @MainActor [weak self] in
            guard let self, self.phase == .recording, let dir = self.captureDirURL, let engine = self.writerEngine else { return }

            // Record actual delivered buffer dimensions (`YS-03`, Blocker 5)
            self.actualWidth = deliveredW
            self.actualHeight = deliveredH

            // Record start_clock_pair at the FIRST delivered ARFrame (Major 11)
            if self.t0MonotonicS == nil {
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime]
                self.startedAtUTC = formatter.string(from: Date())
                self.t0MonotonicS = timestampS
            }

            // Detect upstream frame timestamp gaps (> 1.6x frame period) and prepare gap lines (YS-02, Major 14)
            var gapLines: [String] = []
            let expectedDt = 1.0 / Double(max(1, self.actualFPS))
            if let prevT = self.lastDeliveredFrameTimestampS {
                let dt = timestampS - prevT
                if dt > 1.6 * expectedDt {
                    let missedCount = min(30, max(1, Int((dt / expectedDt).rounded()) - 1))
                    for k in 1...missedCount {
                        let interpT = prevT + Double(k) * (dt / Double(missedCount + 1))
                        gapLines.append(
                            String(format: "%.6f,,true,upstream_frame_drop,%@,%@\n", interpT, trackingStr, reasonStr)
                        )
                    }
                    self.droppedFrameCount += missedCount
                }
            }
            let finalGapLines = gapLines
            self.lastDeliveredFrameTimestampS = timestampS

            self.latestTimestampS = timestampS
            self.latestCameraTransform = cMat
            self.pitchDeg = pitchDegrees
            self.rollDeg = rollDegrees
            self.trackingStateText = trackingStr
            self.crosshairDepthMeters = finalCenterDepthM
            self.latestCrosshairWorld = finalCenterWorldPt

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

            // Backpressure guard before dispatching heavy frame buffers
            let queueOverloaded = self.pendingWriterJobs > 8
            let forceArUco = self.pendingForceMarkerDetect
            self.pendingForceMarkerDetect = false
            let sizesSnapshot = self.markerSizesMM

            let rawQuat = simd_quatd(cMat)
            let normQuat = simd_normalize(rawQuat)
            let fx = Double(intrinsics.columns.0.x)
            let fy = Double(intrinsics.columns.1.y)
            let cx = Double(intrinsics.columns.2.x)
            let cy = Double(intrinsics.columns.2.y)
            let rgbW = deliveredW
            let rgbH = deliveredH
            let wD = finalDepthW
            let hD = finalDepthH

            let sessionHeartbeatData: Data?
            if timestampS - self.lastHeartbeatS >= 5.0 {
                self.lastHeartbeatS = timestampS
                sessionHeartbeatData = self.makeSessionJSONData(status: "recording", stopReason: nil, failedStream: nil)
            } else {
                sessionHeartbeatData = nil
            }

            self.pendingWriterJobs += 1
            self.writerQueue.async { [weak self] in
                guard let self, !engine.isClosed else { return }

                do {
                    // Write any upstream gap rows first
                    for gLine in finalGapLines {
                        try engine.appendLine(gLine, to: engine.framesFH)
                    }

                    guard let dPixels = finalDepthCopy,
                          let cPixels = finalConfCopy,
                          let sendableRGB = copiedRGBBufferOpt
                    else {
                        let dropLine = String(format: "%.6f,,true,missing_depth_buffer,%@,%@\n", timestampS, trackingStr, reasonStr)
                        try engine.appendLine(dropLine, to: engine.framesFH)
                        engine.droppedFrameCount += 1
                        let totalDropped = engine.droppedFrameCount + finalGapLines.count
                        Task { @MainActor in
                            self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                            self.droppedFrameCount = max(self.droppedFrameCount, totalDropped)
                        }
                        return
                    }

                    if queueOverloaded {
                        let dropLine = String(format: "%.6f,,true,writer_queue_full,%@,%@\n", timestampS, trackingStr, reasonStr)
                        try engine.appendLine(dropLine, to: engine.framesFH)
                        engine.droppedFrameCount += 1
                        Task { @MainActor in
                            self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                            self.droppedFrameCount += 1
                        }
                        return
                    }

                    guard let writer = engine.assetWriter,
                          let vInput = engine.videoWriterInput,
                          let adaptor = engine.pixelBufferAdaptor
                    else {
                        throw NSError(domain: "Yardstick", code: -30, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter not initialized"])
                    }

                    if writer.status == .failed {
                        throw writer.error ?? NSError(domain: "Yardstick", code: -31, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter status is failed"])
                    }

                    // Blocker 3: If encoder is temporarily not ready for media data, drop and count the frame;
                    // do NOT fail the capture unless the writer actually errors!
                    if engine.videoSessionStarted && !vInput.isReadyForMoreMediaData {
                        let dropLine = String(format: "%.6f,,true,encoder_not_ready,%@,%@\n", timestampS, trackingStr, reasonStr)
                        try engine.appendLine(dropLine, to: engine.framesFH)
                        engine.droppedFrameCount += 1
                        Task { @MainActor in
                            self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                            self.droppedFrameCount += 1
                        }
                        return
                    }

                    let pts = CMTime(seconds: timestampS, preferredTimescale: 600)
                    if !engine.videoSessionStarted {
                        writer.startSession(atSourceTime: pts)
                        engine.videoSessionStarted = true
                    }

                    let rgbPB = sendableRGB.buffer
                    guard adaptor.append(rgbPB, withPresentationTime: pts) else {
                        throw writer.error ?? NSError(domain: "Yardstick", code: -32, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter adaptor.append failed"])
                    }

                    let sIdx = engine.savedFrameCount
                    engine.savedFrameCount += 1

                    let fLine = String(format: "%.6f,%d,false,,%@,%@\n", timestampS, sIdx, trackingStr, reasonStr)
                    try engine.appendLine(fLine, to: engine.framesFH)

                    let pLine = String(
                        format: "%.6f,%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                        timestampS, sIdx,
                        camPosW.x, camPosW.y, camPosW.z,
                        normQuat.imag.x, normQuat.imag.y, normQuat.imag.z, normQuat.real,
                        fx, fy, cx, cy, rgbW, rgbH
                    )
                    try engine.appendLine(pLine, to: engine.posesFH)

                    let depthURL = dir.appendingPathComponent(String(format: "depth/%06d.png", sIdx))
                    let confURL = dir.appendingPathComponent(String(format: "confidence/%06d.png", sIdx))
                    guard Self.writeDepth16BitPNG(dPixels, width: wD, height: hD, to: depthURL) else {
                        throw NSError(domain: "Yardstick", code: -33, userInfo: [NSLocalizedDescriptionKey: "Failed to write \(depthURL.lastPathComponent)"])
                    }
                    guard Self.writeConfidence8BitPNG(cPixels, width: wD, height: hD, to: confURL) else {
                        throw NSError(domain: "Yardstick", code: -34, userInfo: [NSLocalizedDescriptionKey: "Failed to write \(confURL.lastPathComponent)"])
                    }

                    // Run ArUco detector & periodic sharp still selector on `writerQueue`
                    let shouldRunArUco = (sIdx % 3 == 0) || forceArUco
                    let shouldConsiderPeriodicStill = (timestampS - engine.lastStillSavedTimeS) >= 2.0
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
                                if sharpness >= 12.0 || (timestampS - engine.lastStillSavedTimeS) >= 3.0 {
                                    saveStillNow = true
                                }
                            }
                        }
                    }

                    // Write still JPEG AND its `stills.csv` row atomically on `writerQueue` so no still is ever orphaned when sealing starts!
                    if saveStillNow && !engine.savedStillsIndices.contains(sIdx) {
                        let stillRelPath = String(format: "stills/%06d.jpg", sIdx)
                        let stillURL = dir.appendingPathComponent(stillRelPath)
                        let ciImg = CIImage(cvPixelBuffer: rgbPB)
                        let cs = CGColorSpaceCreateDeviceRGB()
                        try engine.ciContext.writeJPEGRepresentation(
                            of: ciImg,
                            to: stillURL,
                            colorSpace: cs,
                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.92]
                        )
                        engine.savedStillsIndices.insert(sIdx)
                        engine.lastStillSavedTimeS = timestampS
                        let sLine = String(
                            format: "%.6f,%d,%@,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d\n",
                            timestampS, sIdx, stillRelPath,
                            camPosW.x, camPosW.y, camPosW.z,
                            normQuat.imag.x, normQuat.imag.y, normQuat.imag.z, normQuat.real,
                            fx, fy, cx, cy, rgbW, rgbH
                        )
                        try engine.appendLine(sLine, to: engine.stillsFH)
                    }

                    for m in detected3D {
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
                        try engine.appendLine(mLine, to: engine.markersFH)
                    }

                    if let sessionHeartbeatData {
                        Self.writeSessionJSONDataOnQueue(sessionHeartbeatData, in: dir)
                    }

                    let finalDetected = detected3D
                    let newSavedCount = engine.savedFrameCount
                    Task { @MainActor in
                        self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                        self.latestSavedIndex = sIdx
                        self.savedFrameCount = newSavedCount

                        for m in finalDetected {
                            self.detectedMarkerIDs.insert(m.markerID)
                            if m.markerID == 0 {
                                if self.marker0FirstPassPos == nil {
                                    self.marker0FirstPassPos = m.positionWorld
                                    self.marker0FirstPassTimeS = timestampS
                                    self.statusBanner = "Marker 0 Origin locked — walk yard & return to Marker 0 at end"
                                } else if let firstT = self.marker0FirstPassTimeS,
                                          let firstPos = self.marker0FirstPassPos,
                                          (timestampS - firstT) >= 3.0,
                                          (self.marker0MaxExcursionM >= 1.0 || (timestampS - firstT) >= 10.0),
                                          simd_length(camPosW - m.positionWorld) <= 1.50 {
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
                } catch {
                    let errMsg = error.localizedDescription
                    Task { @MainActor in
                        self.pendingWriterJobs = max(0, self.pendingWriterJobs - 1)
                        self.handleStreamFailure(stream: "rgb_video", message: errMsg)
                    }
                }
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
