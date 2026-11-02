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
import os
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
/// and non-Sendable AVFoundation/FileHandle objects never cross concurrency domains (Blockers 3, 4, 5, 6, `YS-68`, `YS-84`).
private final class CaptureWriterEngine: @unchecked Sendable {
    let captureDirURL: URL
    let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    let detectionCIContext = CIContext(options: [.useSoftwareRenderer: false])
    let digestRegistry = PrecomputedDigestRegistry()
    private let stillGateLock = OSAllocatedUnfairLock(initialState: StillCaptureGate())

    /// Field debug log (device walk 4, 2026-11-01): one row per live marker read and why it was or was not used.
    /// Lives in `Documents/debug/`, outside the capture folder, so the sealed capture and its manifest are unchanged.
    static let detectionDebugLock = OSAllocatedUnfairLock(initialState: ())
    static func appendDetectionDebug(captureDir: URL, line: String) {
        detectionDebugLock.withLock { _ in
            let debugDir = captureDir.deletingLastPathComponent().appendingPathComponent("debug", isDirectory: true)
            let url = debugDir.appendingPathComponent(captureDir.lastPathComponent + "-detections.csv")
            let fm = FileManager.default
            if !fm.fileExists(atPath: url.path) {
                try? fm.createDirectory(at: debugDir, withIntermediateDirectories: true)
                let header = "t,saved_index,event,marker_id,measured_3d,size_mm_measured,size_mm_printed,range_m,horiz_m,viewing_angle_deg,turn_rad_s,blur_px,tracking,gate_reason,angle_ok,size_ok,phase_target,eligible\n"
                fm.createFile(atPath: url.path, contents: header.data(using: .utf8))
            }
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile()
                h.write((line + "\n").data(using: .utf8)!)
                try? h.close()
            }
        }
    }
    private let coverageGridLock = OSAllocatedUnfairLock(initialState: GroundCoverageGrid())

    var assetWriter: AVAssetWriter?
    var videoWriterInput: AVAssetWriterInput?
    var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    var videoSessionStarted: Bool = false

    var framesFH: FileHandle?
    var posesFH: FileHandle?
    var stillsFH: FileHandle?
    var markersFH: FileHandle?
    var eventsFH: FileHandle?
    var guidanceFH: FileHandle?
    var audioFH: FileHandle?

    var framesDigest = IncrementalByteDigest()
    var posesDigest = IncrementalByteDigest()
    var stillsDigest = IncrementalByteDigest()
    var markersDigest = IncrementalByteDigest()
    var eventsDigest = IncrementalByteDigest()
    var guidanceDigest = IncrementalByteDigest()

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
    var savedStillsIndices: Set<Int> = []
    var isClosed: Bool = false
    var lastGuidanceTimestampS: Double = -1.0
    var lastGuidanceSavedIndex: Int = 0
    var lastEventTimestampS: Double = -1.0
    var lastEventSavedIndex: Int = 0

    func appendMonotonicGuidanceRow(
        timestampS: Double,
        savedIndex: Int?,
        event: String,
        phase: String,
        instruction: String,
        spoken: Bool,
        pitchDeg: Double,
        rollDeg: Double,
        trackingState: String,
        exposureDurationS: Double,
        iso: Double,
        turnRateRadS: Double,
        predictedBlurPx: Double,
        thermalState: String,
        coverageCells: Int? = nil
    ) throws {
        let t = max(timestampS, lastGuidanceTimestampS)
        lastGuidanceTimestampS = t
        let maxWrittenIdx = savedFrameCount > 0 ? (savedFrameCount - 1) : 0
        let resolvedIdx = savedIndex.map { min(maxWrittenIdx, max($0, lastGuidanceSavedIndex)) }
        if let rIdx = resolvedIdx {
            lastGuidanceSavedIndex = rIdx
        }
        let gLine = ExposureAndStillnessGate.formatGuidanceCSVRow(
            timestampS: t,
            savedIndex: resolvedIdx,
            event: event,
            phase: phase,
            instruction: instruction,
            spoken: spoken,
            pitchDeg: pitchDeg,
            rollDeg: rollDeg,
            trackingState: trackingState,
            exposureDurationS: exposureDurationS,
            iso: iso,
            turnRateRadS: turnRateRadS,
            predictedBlurPx: predictedBlurPx,
            thermalState: thermalState,
            coverageCells: coverageCells ?? coveredCellCount
        )
        try appendLine(gLine, to: guidanceFH)
    }

    func appendMonotonicEventRow(
        timestampS: Double,
        savedIndex: Int,
        kind: String,
        detail: String
    ) throws {
        let t = max(timestampS, lastEventTimestampS)
        lastEventTimestampS = t
        let maxWrittenIdx = savedFrameCount > 0 ? (savedFrameCount - 1) : 0
        let resolvedIdx = min(maxWrittenIdx, max(savedIndex, lastEventSavedIndex))
        lastEventSavedIndex = resolvedIdx
        let safeDetail = detail.replacingOccurrences(of: ",", with: ";").replacingOccurrences(of: "\n", with: " ")
        let line = String(format: "%.6f,%d,%@,%@\n", t, resolvedIdx, kind, safeDetail)
        try appendLine(line, to: eventsFH)
    }

    var lastAnyStillTimeS: Double {
        stillGateLock.withLock { $0.lastAnyStillTimeS }
    }

    func evaluateStillGate(
        timestampS: Double,
        detectedMarkerIDs: [Int],
        forceOperatorMarkerTap: Bool,
        sharpness: Double?
    ) -> (shouldSave: Bool, reason: String?, snapshotBefore: StillCaptureGate) {
        stillGateLock.withLock { gate in
            let before = gate
            let res = gate.evaluateFrame(
                timestampS: timestampS,
                detectedMarkerIDs: detectedMarkerIDs,
                forceOperatorMarkerTap: forceOperatorMarkerTap,
                sharpness: sharpness
            )
            return (res.shouldSave, res.reason, before)
        }
    }

    func restoreStillGate(_ snapshot: StillCaptureGate) {
        stillGateLock.withLock { gate in
            gate = snapshot
        }
    }

    var coveredCellCount: Int {
        coverageGridLock.withLock { $0.coveredCellCount }
    }

    func isCoverageUpdateDue(at timestampS: Double, minIntervalS: Double) -> Bool {
        coverageGridLock.withLock { $0.shouldUpdate(at: timestampS, minIntervalS: minIntervalS) }
    }

    func anchorGroundPlaneY(_ yWorld: Double) {
        coverageGridLock.withLock { grid in
            grid.anchorGroundPlaneY(yWorld)
        }
    }

    func anchorMarker0Frame(originWorld: SIMD3<Double>, xAxisWorld: SIMD3<Double>) {
        coverageGridLock.withLock { grid in
            grid.anchorMarker0Frame(originWorld: originWorld, xAxisWorld: xAxisWorld)
        }
    }

    func updateCoverageGridIfDue(
        timestampS: Double,
        minIntervalS: Double,
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
        cameraToWorld: simd_double4x4
    ) -> (didUpdate: Bool, snapshot: GroundCoverageGrid, validDepthSamples: Int, elevatedPointsAbove2m: Int)? {
        coverageGridLock.withLock { grid in
            guard grid.shouldUpdate(at: timestampS, minIntervalS: minIntervalS) else {
                return nil
            }
            let stats = grid.ingestDepthFrame(
                timestampS: timestampS,
                depthMapMM: depthMapMM,
                confidenceMap: confidenceMap,
                depthWidth: depthWidth,
                depthHeight: depthHeight,
                rgbWidth: rgbWidth,
                rgbHeight: rgbHeight,
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                cameraToWorld: cameraToWorld
            )
            return (true, grid, stats.validDepthSamples, stats.elevatedPointsAbove2m)
        }
    }

    init(captureDirURL: URL, width: Int, height: Int) throws {
        self.captureDirURL = captureDirURL
        let fm = FileManager.default
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("depth"), withIntermediateDirectories: true)
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("confidence"), withIntermediateDirectories: true)
        try fm.createDirectory(at: captureDirURL.appendingPathComponent("stills"), withIntermediateDirectories: true)

        try Self.createFileWithHeader(
            "t,saved_index,dropped,drop_reason,tracking_state,tracking_reason\n",
            at: captureDirURL.appendingPathComponent("frames.csv"),
            digest: &framesDigest,
            registry: digestRegistry
        )
        try Self.createFileWithHeader(
            "t,saved_index,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height\n",
            at: captureDirURL.appendingPathComponent("poses.csv"),
            digest: &posesDigest,
            registry: digestRegistry
        )
        try Self.createFileWithHeader(
            "t,saved_index,file,px,py,pz,qx,qy,qz,qw,fx,fy,cx,cy,width,height,exposure_duration_s,iso\n",
            at: captureDirURL.appendingPathComponent("stills.csv"),
            digest: &stillsDigest,
            registry: digestRegistry
        )
        try Self.createFileWithHeader(
            "t,saved_index,marker_id,size_mm_printed,size_mm_measured,x,y,z,qx,qy,qz,qw,c0_u,c0_v,c1_u,c1_v,c2_u,c2_v,c3_u,c3_v\n",
            at: captureDirURL.appendingPathComponent("markers.csv"),
            digest: &markersDigest,
            registry: digestRegistry
        )
        try Self.createFileWithHeader(
            "t,saved_index,kind,detail\n",
            at: captureDirURL.appendingPathComponent("events.csv"),
            digest: &eventsDigest,
            registry: digestRegistry
        )
        try Self.createFileWithHeader(
            ExposureAndStillnessGate.guidanceCSVHeader,
            at: captureDirURL.appendingPathComponent("guidance.csv"),
            digest: &guidanceDigest,
            registry: digestRegistry
        )
        try Self.initializeAudioWavHeader(at: captureDirURL.appendingPathComponent("audio.wav"))
        digestRegistry.addRecordedBytes(44)

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
        self.guidanceFH = try FileHandle(forWritingTo: captureDirURL.appendingPathComponent("guidance.csv"))
        try self.guidanceFH?.seekToEnd()
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

    private static func createFileWithHeader(
        _ header: String,
        at url: URL,
        digest: inout IncrementalByteDigest,
        registry: PrecomputedDigestRegistry
    ) throws {
        guard let data = header.data(using: .utf8) else {
            throw NSError(domain: "Yardstick", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to encode CSV header as UTF-8"])
        }
        try data.write(to: url, options: .atomic)
        digest.update(with: data)
        registry.addRecordedBytes(UInt64(data.count))
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
        if fh === framesFH {
            framesDigest.update(with: data)
        } else if fh === posesFH {
            posesDigest.update(with: data)
        } else if fh === stillsFH {
            stillsDigest.update(with: data)
        } else if fh === markersFH {
            markersDigest.update(with: data)
        } else if fh === eventsFH {
            eventsDigest.update(with: data)
        } else if fh === guidanceFH {
            guidanceDigest.update(with: data)
        }
        digestRegistry.addRecordedBytes(UInt64(data.count))
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
            digestRegistry.addRecordedBytes(UInt64(byteCount))
        }
    }

    /// Append zero-PCM silence padding to `audio.wav` during an experimental pause-and-resume (`YS-82` Experiment 5).
    func appendAudioPaddingBytes(_ paddingData: Data) throws {
        guard !paddingData.isEmpty else { return }
        guard let aFH = audioFH else {
            throw NSError(domain: "Yardstick", code: -18, userInfo: [NSLocalizedDescriptionKey: "audio.wav FileHandle is closed"])
        }
        try aFH.write(contentsOf: paddingData)
        audioPCMByteCount &+= UInt32(paddingData.count)
        digestRegistry.addRecordedBytes(UInt64(paddingData.count))
    }

    /// Finalize `audio.wav` RIFF/data chunk sizes, finish `rgb.mp4`, record incremental CSV digests (`YS-84`),
    /// and close all open file handles.
    /// Does NOT call `handleStreamFailure` so it can never cause recursive failure loops (Blocker 6).
    func finishStreamsAndCloseFiles() throws {
        guard !isClosed else { return }
        isClosed = true

        try? framesFH?.close()
        try? posesFH?.close()
        try? stillsFH?.close()
        try? markersFH?.close()
        try? eventsFH?.close()
        try? guidanceFH?.close()

        digestRegistry.record(framesDigest.finalize(), forRelativePath: "frames.csv")
        digestRegistry.record(posesDigest.finalize(), forRelativePath: "poses.csv")
        digestRegistry.record(stillsDigest.finalize(), forRelativePath: "stills.csv")
        digestRegistry.record(markersDigest.finalize(), forRelativePath: "markers.csv")
        digestRegistry.record(eventsDigest.finalize(), forRelativePath: "events.csv")
        digestRegistry.record(guidanceDigest.finalize(), forRelativePath: "guidance.csv")

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

/// Summary record of a completed capture session displayed in the root `Captures` list (`YS-80`, `DESIGN-SPEC-NATIVE.md` §2a, §2c, §3f).
public struct SavedCaptureRecord: Sendable, Identifiable, Equatable {
    public let id: String
    public var title: String
    public let dateText: String
    public let durationSeconds: Int
    public let savedFrameCount: Int
    public let coveredGroundCells: Set<GroundGridCell>
    public let coveredGroundAreaM2: Double
    public let detailSummary: CaptureDetailSummary
    public let archiveZipURL: URL?

    public init(
        id: String,
        title: String = "Back yard",
        dateText: String,
        durationSeconds: Int,
        savedFrameCount: Int,
        coveredGroundCells: Set<GroundGridCell>,
        coveredGroundAreaM2: Double,
        detailSummary: CaptureDetailSummary,
        archiveZipURL: URL?
    ) {
        self.id = id
        self.title = title
        self.dateText = dateText
        self.durationSeconds = durationSeconds
        self.savedFrameCount = savedFrameCount
        self.coveredGroundCells = coveredGroundCells
        self.coveredGroundAreaM2 = coveredGroundAreaM2
        self.detailSummary = detailSummary
        self.archiveZipURL = archiveZipURL
    }
}

/// Crash-safe, zero-`ARFrame`-retention LiDAR capture session recorder (`YS-00`–`YS-14`, `YS-21`, `YS-22`, `YS-30`–`YS-33`, `YS-40`–`YS-44`, `YS-68`, `YS-84`).
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
    @Published public private(set) var sealingProgress: Double = 0.0
    @Published public private(set) var sealingStatusText: String = "Saving your walk"
    @Published public private(set) var lowDiskSpaceWarning: String? = nil
    @Published public private(set) var latestExposureDurationS: Double = 0.0
    @Published public private(set) var latestISO: Double = 100.0
    @Published public private(set) var latestTurnRateRadS: Double = 0.0
    @Published public private(set) var latestPredictedBlurPx: Double = 0.0
    @Published public private(set) var liveMarkerSizeWarning: String? = nil
    @Published public private(set) var lastSizeMismatchMarkerID: Int? = nil
    @Published public private(set) var beaconProjections: [MarkerBeaconProjection] = []
    @Published public private(set) var isLoopClosed: Bool = false
    @Published public private(set) var guidedWalkPhase: GuidedWalkPhase = .openingMarker0
    @Published public private(set) var activeRingState: MarkerRingVisualState? = nil
    @Published public private(set) var lockedMarkerIDs: Set<Int> = []
    @Published public private(set) var isFinishAvailable: Bool = false {
        didSet {
            if !oldValue && isFinishAvailable {
                UIAccessibility.post(notification: .announcement, argument: "Finish button available")
            }
        }
    }
    @Published public private(set) var coveredGroundCellCount: Int = 0
    @Published public private(set) var coveredGroundAreaM2: Double = 0.0
    @Published public private(set) var coveredGroundCells: Set<GroundGridCell> = []
    @Published public private(set) var lookUpProgressFraction: Double = 0.0
    @Published public private(set) var isLookUpDone: Bool = false
    @Published public private(set) var thermalSchedule = ThermalAdaptationSchedule.evaluate(
        thermalLevel: .nominal,
        isMarkerHoldActive: false
    )
    @Published public private(set) var latestHapticEvent: GuidedHapticEvent? = nil
    @Published public private(set) var hapticDispatchCount: Int = 0
    public private(set) var dispatchedHaptics: [GuidedHapticEvent] = []
    @Published public private(set) var isRecordingVoiceNote: Bool = false
    /// Voice coach toggle (`YS-75`, `DESIGN-SPEC-NATIVE.md` §3g).
    @Published public var isVoiceCoachEnabled: Bool = true {
        didSet {
            guidedEngine.isVoiceCoachEnabled = isVoiceCoachEnabled
            if !isVoiceCoachEnabled {
                speechSynthesizer.stopSpeaking(at: .immediate)
            }
        }
    }
    /// Haptics toggle (`DESIGN-SPEC-NATIVE.md` §3g, §5: `"Make haptics optional"`).
    @Published public var isHapticsEnabled: Bool = true
    /// Developer live diagnostics overlay toggle (`YS-80`, `DESIGN-SPEC-NATIVE.md` §3c, §3g: off by default, never on the regular capture screen).
    @Published public var showDeveloperDiagnosticsOverlay: Bool = false
    /// `YS-82` experiment flags (all default off, logged in `guidance.csv`, off unless proven).
    @Published public var experimentFlags: CaptureExperimentFlags = .default
    public private(set) var exposureCapController = ExposureCapExperimentController()
    private var lastPauseStartedTimestampS: Double? = nil
    /// Completed capture records shown in the root `Captures` list (`YS-80`, `DESIGN-SPEC-NATIVE.md` §2a, §2c).
    @Published public var savedCaptureRecords: [SavedCaptureRecord] = []
    /// Live Walk Review report (`YS-72`, `DESIGN-SPEC-NATIVE.md` §3e, §4e).
    @Published public private(set) var latestWalkReviewReport: WalkReviewReport = WalkReviewEvaluator.evaluate(
        lockedMarkerIDs: [],
        isClosingMarker0Locked: false,
        unifiedLoopClosed: false,
        fixNowRequiresReturnToMarker0: false,
        endEarlyChosen: false,
        lookUpDone: false,
        lookUpSkipped: false,
        coveredGroundCellCount: 0
    )
    /// Injectable VoiceOver status provider (`YS-75`: routes coach lines through VoiceOver announcements instead of a second voice).
    public var isVoiceOverRunningProvider: () -> Bool = { UIAccessibility.isVoiceOverRunning }
    @Published public private(set) var spokenCoachHistory: [VoiceCoachOutputRoute] = []
    private let speechSynthesizer = AVSpeechSynthesizer()

    /// Pre-walk centre-to-centre tape distance (meters) between Marker 0 and Marker 1 (`YS-77`).
    @Published public var preWalkTapeMeters0To1: Double? = nil {
        didSet {
            guidedEngine.preWalkTapeMeters0To1 = preWalkTapeMeters0To1
            syncMarker0To1PreWalkTapeCheckIfNeeded()
            _ = evaluatePreFlightChecklist()
        }
    }
    @Published public var isPrintSizeConfirmed: Bool = false {
        didSet {
            _ = evaluatePreFlightChecklist()
        }
    }
    @Published public private(set) var latestLensSmudgeProbe: LensSmudgeProbeResult? = nil
    @Published public private(set) var preFlightReport: PreFlightChecklistReport = PreFlightChecklistEvaluator.evaluate(
        availableDiskBytes: 25_000_000_000,
        batteryLevelFraction: 0.85,
        thermalLevel: .nominal,
        exposureDurationS: 0.004,
        iso: 100.0,
        isLocationAuthorized: false
    )
    /// Injectable thermal level for deterministic testing of `YS-85` serious-thermal adaptation.
    public var injectedThermalLevelForTesting: CaptureThermalLevel? = nil

    private static let markerSizesDefaultsKey = "io.buildnexus.yardstick.markerSizesMM.v2"

    /// Per-marker printed size configuration (`180.0` mm default for all markers `0–5` per `YS-80` #3), persisted in `UserDefaults`.
    @Published public var markerSizesMM: [Int: Double] = CaptureSessionRecorder.loadPersistedMarkerSizes()

    public static func loadPersistedMarkerSizes(defaults: UserDefaults = .standard) -> [Int: Double] {
        var sizes: [Int: Double] = [
            0: 180.0, 1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 180.0,
        ]
        if let saved = defaults.dictionary(forKey: markerSizesDefaultsKey) {
            for mid in 0..<6 {
                if let num = saved[String(mid)] as? NSNumber {
                    let val = num.doubleValue
                    if val >= 80.0 && val <= 400.0 {
                        sizes[mid] = val
                    }
                }
            }
        }
        return sizes
    }

    /// Sets every marker ID (`0...5`) to the given printed size in millimeters (`YS-80` correction #3:
    /// "the marker-size setting sets every marker ID").
    public func setAllMarkersPrintedSizeMM(_ sizeMM: Double, defaults: UserDefaults = .standard) {
        guard sizeMM >= 80.0 && sizeMM <= 400.0 else { return }
        for mid in 0..<6 {
            markerSizesMM[mid] = sizeMM
        }
        persistMarkerSizes(defaults: defaults)
        _ = evaluatePreFlightChecklist()
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

    /// Atomic writer backlog counter updated strictly on `writerQueue` (`YS-68`).
    nonisolated public let writerBacklog = WriterBacklogCounter(maxBacklog: 8)
    /// Single-slot background detection scheduler that drops incoming work immediately when busy (`YS-68`).
    nonisolated public let detectionScheduler = SingleSlotDetectionScheduler()
    /// Injectable free-disk-bytes provider for mid-walk `ENOSPC` prevention and unit testing (`YS-84`).
    public var availableDiskBytesProvider: (@Sendable (URL) -> Int64?) = { DiskSpaceMonitor.querySystemAvailableBytes(at: $0) }

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

    private var requestedWidth: Int = 1920
    private var requestedHeight: Int = 1440
    private var requestedFPS: Int = 30
    public private(set) var actualWidth: Int = 1920
    public private(set) var actualHeight: Int = 1440
    private var actualFPS: Int = 30

    private var latestCameraTransform: simd_double4x4 = matrix_identity_double4x4
    private var latestCrosshairWorld: SIMD3<Double>?
    private var latestSavedIndex: Int = 0
    private var latestTimestampS: Double = 0.0
    private var latestTrueHeadingDeg: Double?
    private var latestMagneticHeadingDeg: Double?
    private var latestHeadingAccuracyDeg: Double?
    private var latestLocation: CLLocation?
    nonisolated private static let writerSignposter = OSSignposter(
        subsystem: "io.yardstick.capture",
        category: "WriterQueue"
    )

    private var locationAuthStatus: String = "unavailable"
    private var northSamples: [[String: Double]] = []
    private var phoneLoopState = PhoneLoopRuleState()
    private var guidedEngine = GuidedCaptureEngine()
    private var latestValidDepthSamplesCount: Int = 100
    private var latestElevatedPointsAbove2m: Int = 0
    private var marker0FirstPassPos: SIMD3<Double>? {
        get { phoneLoopState.marker0FirstPassPos }
        set { phoneLoopState.marker0FirstPassPos = newValue }
    }
    private var marker0LastPassPos: SIMD3<Double>? {
        get { phoneLoopState.marker0LastPassPos }
        set { phoneLoopState.marker0LastPassPos = newValue }
    }
    private var marker0FirstPassTimeS: Double? {
        get { phoneLoopState.marker0FirstPassTimeS }
        set { phoneLoopState.marker0FirstPassTimeS = newValue }
    }
    private var marker0LastPassTimeS: Double? {
        get { phoneLoopState.marker0LastPassTimeS }
        set { phoneLoopState.marker0LastPassTimeS = newValue }
    }
    private var marker0MaxExcursionM: Double {
        get { phoneLoopState.marker0MaxExcursionM }
        set { phoneLoopState.marker0MaxExcursionM = newValue }
    }
    private var tapeChecks: [[String: Any]] = []
    private var pendingTapeASavedIndex: Int = 0
    private var pendingTapeATimeS: Double = 0.0
    private var pendingTapeBSavedIndex: Int = 0
    private var pendingTapeBTimeS: Double = 0.0
    private var provisionalDistanceTolerance10m: Double = 0.05
    private var lastPreFlightProbeTimeS: Double = -10.0
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
        if UserDefaults.standard.dictionary(forKey: Self.markerSizesDefaultsKey) == nil {
            persistMarkerSizes()
        }
        let args = ProcessInfo.processInfo.arguments
        if args.contains("--force-thermal-serious") {
            injectedThermalLevelForTesting = .serious
        }
        if args.contains("--force-low-space") {
            availableDiskBytesProvider = { _ in 800 * 1024 * 1024 }
        }
        locationManager.delegate = self
        locationManager.headingFilter = kCLHeadingFilterNone
        UIDevice.current.isBatteryMonitoringEnabled = true
        speechSynthesizer.usesApplicationAudioSession = true
        #if canImport(ARKit)
        arSession.delegate = self
        #endif
        registerLifecycleObservers()
        _ = evaluatePreFlightChecklist()
    }

    /// Dispatch a `YS-75` spoken coach phrase, sharing the AR session's `AVAudioSession` without deactivating it,
    /// routing through VoiceOver announcements when VoiceOver is running, and staying silent while a voice note records.
    @discardableResult
    public func dispatchVoiceCoachIfNeeded(_ spokenPhrase: String?) -> Bool {
        guard let route = GuidedVoiceCatalog.resolveOutputRoute(
            phrase: spokenPhrase,
            isVoiceCoachEnabled: isVoiceCoachEnabled,
            isRecordingVoiceNote: guidedEngine.isRecordingVoiceNote,
            isVoiceOverRunning: isVoiceOverRunningProvider()
        ) else {
            return false
        }
        spokenCoachHistory.append(route)
        switch route {
        case .voiceOverAnnouncement(let phrase):
            UIAccessibility.post(notification: .announcement, argument: phrase)
        case .speechSynthesizer(let phrase):
            let utterance = AVSpeechUtterance(string: phrase)
            utterance.prefersAssistiveTechnologySettings = true
            if speechSynthesizer.isSpeaking {
                speechSynthesizer.stopSpeaking(at: .immediate)
            }
            speechSynthesizer.speak(utterance)
        }
        return true
    }

    /// Evaluate the `YS-76` pre-flight checklist and update `preFlightReport`.
    @discardableResult
    public func evaluatePreFlightChecklist(
        overrideAvailableBytes: Int64? = nil,
        overrideBatteryLevel: Double? = nil,
        overrideThermalLevel: CaptureThermalLevel? = nil,
        overrideExposureDurationS: Double? = nil,
        overrideISO: Double? = nil,
        overrideLocationAuthorized: Bool? = nil,
        overrideLensProbe: LensSmudgeProbeResult? = nil
    ) -> PreFlightChecklistReport {
        let docsURL = captureDirURL ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let availBytes = overrideAvailableBytes ?? availableDiskBytesProvider(docsURL) ?? 25_000_000_000
        let sysBatt = Double(UIDevice.current.batteryLevel)
        let battLevel = overrideBatteryLevel ?? (sysBatt < 0 ? 0.85 : sysBatt)
        let therm = overrideThermalLevel ?? injectedThermalLevelForTesting ?? CaptureThermalLevel.fromSystem(ProcessInfo.processInfo.thermalState)
        let expS = overrideExposureDurationS ?? (latestExposureDurationS > 0 ? latestExposureDurationS : 0.004)
        let isoV = overrideISO ?? (latestISO > 0 ? latestISO : 100.0)
        let locAuth = overrideLocationAuthorized
            ?? (locationAuthStatus == "ok" || locationAuthStatus == "accuracy_exceeded"
                || locationAuthStatus == "authorizedWhenInUse" || locationAuthStatus == "authorizedAlways")
        let lensProbe = overrideLensProbe ?? latestLensSmudgeProbe
        let printedMM = markerSizesMM[0] ?? 180.0

        let report = PreFlightChecklistEvaluator.evaluate(
            availableDiskBytes: availBytes,
            batteryLevelFraction: battLevel,
            thermalLevel: therm,
            exposureDurationS: expS,
            iso: isoV,
            isLocationAuthorized: locAuth,
            lensProbe: lensProbe,
            printedMarkerSizeMM: printedMM,
            isPrintSizeConfirmed: isPrintSizeConfirmed,
            preWalkTapeMeters0To1: preWalkTapeMeters0To1
        )
        preFlightReport = report
        return report
    }

    /// Clear any previous lens smudge warning (`YS-76`) and force an immediate fresh pre-flight probe (`Check Again`).
    @discardableResult
    public func resetLensSmudgeProbeAndCheckAgain() -> PreFlightChecklistReport {
        latestLensSmudgeProbe = nil
        lastPreFlightProbeTimeS = -10.0
        return evaluatePreFlightChecklist()
    }

    /// Start a lightweight pre-flight camera probe on the setup screen (`YS-76`) if camera access is already authorized,
    /// without prompting prematurely before `Start Walk`.
    public func startPreFlightCameraProbeIfAuthorized() {
        #if canImport(ARKit) && !targetEnvironment(simulator)
        guard phase == .setup, Self.isLiDARSupported else { return }
        if AVCaptureDevice.authorizationStatus(for: .video) == .authorized {
            let config = ARWorldTrackingConfiguration()
            config.worldAlignment = .gravity
            arSession.run(config, options: [])
        }
        #endif
    }

    /// Update the live lens smudge probe (`YS-76`) from a camera luma frame.
    @discardableResult
    public func updateLensSmudgeProbe(
        lumaBytes: [UInt8],
        width: Int,
        height: Int,
        turnRateRadS: Double,
        predictedBlurPx: Double,
        cameraVelocityMps: Double = 0.0
    ) -> LensSmudgeProbeResult {
        let probe = LensSmudgeDetector.evaluate(
            lumaBytes: lumaBytes,
            width: width,
            height: height,
            turnRateRadS: turnRateRadS,
            predictedBlurPx: predictedBlurPx,
            cameraVelocityMps: cameraVelocityMps
        )
        latestLensSmudgeProbe = probe
        _ = evaluatePreFlightChecklist()
        return probe
    }

    /// Where the tape lay (YS-86): on grass it bows and twists (~5 cm, Martin's field number); along a wall it stays
    /// straight (~2 cm). Recorded per tape check as `tape_uncertainty_m`; the grade discounts only that much.
    public enum TapeSurface: String, CaseIterable, Sendable {
        case grass, wall
        public var uncertaintyM: Double { self == .grass ? 0.05 : 0.02 }
        public var label: String { self == .grass ? "On grass (±5 cm)" : "Along a wall (±2 cm)" }
    }
    private static let tapeSurfaceDefaultsKey = "io.buildnexus.yardstick.tapeSurface.v1"
    @Published public var tapeSurface: TapeSurface =
        TapeSurface(rawValue: UserDefaults.standard.string(forKey: CaptureSessionRecorder.tapeSurfaceDefaultsKey) ?? "") ?? .grass {
        didSet {
            UserDefaults.standard.set(tapeSurface.rawValue, forKey: Self.tapeSurfaceDefaultsKey)
            for i in tapeChecks.indices { tapeChecks[i]["tape_uncertainty_m"] = tapeSurface.uncertaintyM }
            updateProvisionalGrade()
        }
    }

    /// Set the pre-walk centre-to-centre tape distance (`Marker 0 -> Marker 1`, `YS-77`) and synchronize `session.json["tape_checks"]`
    /// once both Marker 0 and Marker 1 are locked.
    public func setPreWalkTapeMeters0To1(_ meters: Double?) {
        preWalkTapeMeters0To1 = (meters != nil && meters! > 0) ? meters : nil
    }

    /// Synchronize the automatic `YS-77` centre-to-centre `Marker 0 -> Marker 1` tape entry into `tapeChecks` (`session.json["tape_checks"]`)
    /// without requiring any mid-walk typing or writing an operator tap event (`events.csv`).
    public func syncMarker0To1PreWalkTapeCheckIfNeeded() {
        guard let autoEntry = guidedEngine.generatedMarker0To1TapeCheck else { return }
        var dict = autoEntry.toJSONDictionary()
        dict["tape_uncertainty_m"] = tapeSurface.uncertaintyM
        let summaryStr = String(
            format: "Marker 0→1 (%@): tape %.2f m · phone %.2f m (diff %.1f cm)",
            autoEntry.role, autoEntry.tapeM, autoEntry.measuredM, autoEntry.residualM * 100.0
        )
        if !tapeChecks.isEmpty, (tapeChecks[0]["id"] as? Int) == 0 {
            tapeChecks[0] = dict
            if !tapeCheckSummaries.isEmpty {
                tapeCheckSummaries[0] = summaryStr
            } else {
                tapeCheckSummaries.append(summaryStr)
            }
        } else {
            tapeChecks.insert(dict, at: 0)
            for idx in 0..<tapeChecks.count {
                tapeChecks[idx]["id"] = idx
            }
            tapeCheckSummaries.insert(summaryStr, at: 0)
        }
        tapeCheckCount = tapeChecks.count
        updateProvisionalGrade()
    }

    /// Dispatch a `GuidedHapticEvent` through native `UIKit` feedback generators while respecting
    /// voice-note recording suppression and the `isHapticsEnabled` setting (`YS-74`, `DESIGN-SPEC-NATIVE.md` §3g, §5).
    public func dispatchHapticIfNeeded(_ event: GuidedHapticEvent?) {
        guard let event, isHapticsEnabled, !guidedEngine.isRecordingVoiceNote else { return }
        latestHapticEvent = event
        hapticDispatchCount += 1
        dispatchedHaptics.append(event)
        #if canImport(UIKit) && !targetEnvironment(macCatalyst)
        switch event {
        case .sessionStart, .sessionStop, .markerFirstSeen:
            let gen = UIImpactFeedbackGenerator(style: .light)
            gen.prepare()
            gen.impactOccurred()
        case .holdSegmentConfirmed(_, _, let intensity):
            let gen = UIImpactFeedbackGenerator(style: .rigid)
            gen.prepare()
            gen.impactOccurred(intensity: CGFloat(min(1.0, max(0.2, intensity))))
        case .markerLocked, .loopClosedPathComplete:
            let gen = UINotificationFeedbackGenerator()
            gen.prepare()
            gen.notificationOccurred(.success)
        case .warning:
            let gen = UINotificationFeedbackGenerator()
            gen.prepare()
            gen.notificationOccurred(.warning)
        case .error:
            let gen = UINotificationFeedbackGenerator()
            gen.prepare()
            gen.notificationOccurred(.error)
        }
        #endif
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
        try? AVAudioSession.sharedInstance().setAllowHapticsAndSystemSoundsDuringRecording(true)
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
        let dir = docs.appendingPathComponent(captureID, isDirectory: true).resolvingSymlinksInPath().standardizedFileURL
        captureDirURL = dir
        savedFrameCount = 0
        droppedFrameCount = 0
        latestSavedIndex = 0
        latestTimestampS = 0.0
        lastHeartbeatS = 0.0
        detectedMarkerIDs.removeAll()
        northSamples.removeAll()
        northSampleCount = 0
        tapeChecks.removeAll()
        tapeCheckCount = 0
        tapeCheckSummaries.removeAll()
        pendingTapeEndpointAWorld = nil
        pendingTapeEndpointBWorld = nil
        phoneLoopState = PhoneLoopRuleState()
        loopClosureGapMeters = nil
        t0MonotonicS = nil
        audioFirstSampleTS = nil
        lastNorthSampleTimeS = -10.0
        lastDeliveredFrameTimestampS = nil
        writerBacklog.reset()
        detectionScheduler.resetStats()
        provisionalGrade = "—"
        provisionalReasons = []
        provisionalDistanceTolerance10m = 0.05
        archiveZipURL = nil
        sealingProgress = 0.0
        sealingStatusText = "Saving your walk"
        lowDiskSpaceWarning = nil
        latestExposureDurationS = 0.0
        latestISO = 100.0
        latestTurnRateRadS = 0.0
        latestPredictedBlurPx = 0.0
        liveMarkerSizeWarning = nil
        lastSizeMismatchMarkerID = nil
        beaconProjections = []
        exposureCapController = ExposureCapExperimentController()
        lastPauseStartedTimestampS = nil
        isLoopClosed = false
        guidedEngine = GuidedCaptureEngine(preWalkTapeMeters0To1: preWalkTapeMeters0To1)
        guidedEngine.isVoiceCoachEnabled = isVoiceCoachEnabled
        spokenCoachHistory = []
        guidedWalkPhase = .openingMarker0
        activeRingState = nil
        lockedMarkerIDs = []
        isFinishAvailable = false
        coveredGroundCellCount = 0
        coveredGroundAreaM2 = 0.0
        coveredGroundCells = []
        lookUpProgressFraction = 0.0
        isLookUpDone = false
        isRecordingVoiceNote = false
        thermalSchedule = ThermalAdaptationSchedule.evaluate(thermalLevel: .nominal, isMarkerHoldActive: false)
        hapticDispatchCount = 0
        dispatchedHaptics = []
        dispatchHapticIfNeeded(.sessionStart)
        latestValidDepthSamplesCount = 0
        latestElevatedPointsAbove2m = 0

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
            let expSummary = experimentFlags.guidanceLogSummary
            writerQueue.async {
                Self.writeSessionJSONDataOnQueue(snapData, in: dir, registry: engine.digestRegistry)
                try? engine.appendMonotonicGuidanceRow(
                    timestampS: 0.0,
                    savedIndex: 0,
                    event: "experiments",
                    phase: "opening_marker_0",
                    instruction: expSummary,
                    spoken: false,
                    pitchDeg: -30.0,
                    rollDeg: 0.0,
                    trackingState: "normal",
                    exposureDurationS: 0.0,
                    iso: 100.0,
                    turnRateRadS: 0.0,
                    predictedBlurPx: 0.0,
                    thermalState: "nominal",
                    coverageCells: 0
                )
            }
            phase = .recording
            statusBanner = guidedEngine.activeInstruction
            _ = checkMidWalkDiskSpace(recordedBytes: 0)
        } catch {
            UIApplication.shared.isIdleTimerDisabled = false
            phase = .failed
            statusBanner = "Failed to initialize capture: \(error.localizedDescription)"
        }
    }

    /// Evaluate free disk space during a walk and warn `"Almost out of space. Finish soon."` before sealing can fail with `ENOSPC` (`YS-84`).
    @discardableResult
    public func checkMidWalkDiskSpace(recordedBytes: Int64) -> Bool {
        let queryURL = captureDirURL ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        guard let avail = availableDiskBytesProvider(queryURL) else { return false }
        let eval = DiskSpaceMonitor.evaluateMidWalkStorage(
            availableBytes: avail,
            recordedCaptureBytes: recordedBytes
        )
        if eval.isLowSpace {
            let msg = eval.warningMessage ?? "Almost out of space. Finish soon."
            lowDiskSpaceWarning = msg
            statusBanner = msg
            return true
        } else {
            lowDiskSpaceWarning = nil
            return false
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
        if ["marker", "note", "tape_a", "tape_b"].contains(kind) {
            phoneLoopState.recordOperatorTap(at: t)
            isLoopClosed = phoneLoopState.unifiedLoopClosed
            loopClosureGapMeters = phoneLoopState.unifiedLoopClosureGapMeters
        } else if kind == "loop_closed" {
            phoneLoopState.recordLoopClosedEvent(at: t)
            isLoopClosed = phoneLoopState.unifiedLoopClosed
            loopClosureGapMeters = phoneLoopState.unifiedLoopClosureGapMeters
        }
        writerQueue.async { [weak self] in
            do {
                try engine.appendMonotonicEventRow(timestampS: t, savedIndex: sIdx, kind: kind, detail: detail)
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "events", message: error.localizedDescription)
                }
            }
        }
    }

    /// Toggle or set voice-note recording state (`YS-74`, `YS-75`), suppressing haptics and voice coach while active.
    public func setRecordingVoiceNote(_ active: Bool) {
        if active && speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        if active && !guidedEngine.isRecordingVoiceNote {
            dispatchHapticIfNeeded(.sessionStart)
        }
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        guidedEngine.setRecordingVoiceNote(active, at: t)
        isRecordingVoiceNote = guidedEngine.isRecordingVoiceNote
        statusBanner = guidedEngine.activeInstruction
        if !active {
            dispatchHapticIfNeeded(.sessionStop)
        }
        if active {
            logEvent(kind: "note", detail: "Spoken note anchor")
        }
        guard phase == .recording, let engine = writerEngine else { return }
        let evName = active ? "note_started" : "note_stopped"
        let phaseStr = guidedEngine.phase.rawValue
        let instr = guidedEngine.activeInstruction
        let pDeg = pitchDeg
        let rDeg = rollDeg
        let trk = trackingStateText
        let expS = latestExposureDurationS
        let isoV = latestISO
        let trS = latestTurnRateRadS
        let blurP = latestPredictedBlurPx
        let thermS = thermalSchedule.thermalLevel.rawValue
        let covC = coveredGroundCellCount
        writerQueue.async { [weak self] in
            do {
                try engine.appendMonotonicGuidanceRow(
                    timestampS: t,
                    savedIndex: sIdx,
                    event: evName,
                    phase: phaseStr,
                    instruction: instr,
                    spoken: false,
                    pitchDeg: pDeg,
                    rollDeg: rDeg,
                    trackingState: trk,
                    exposureDurationS: expS,
                    iso: isoV,
                    turnRateRadS: trS,
                    predictedBlurPx: blurP,
                    thermalState: thermS,
                    coverageCells: covC
                )
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "guidance", message: error.localizedDescription)
                }
            }
        }
    }

    /// Mark `"End early"` chosen (`YS-71`, `YS-72`, `YS-74`), synchronizing `PhoneLoopRuleState` and `GuidedCaptureEngine`
    /// and logging `event = "end_early"` to `guidance.csv`.
    public func markEndEarlyChosen() {
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        phoneLoopState.endEarlyChosen = true
        phoneLoopState.evaluateUnifiedLoopClosure()
        guidedEngine.markEndEarlyChosen(at: t)
        let didSpeak = dispatchVoiceCoachIfNeeded(guidedEngine.lastSpokenPhraseThisStep)
        guidedWalkPhase = guidedEngine.phase
        isLoopClosed = phoneLoopState.unifiedLoopClosed
        isFinishAvailable = guidedEngine.isFinishAvailable(unifiedLoopClosed: phoneLoopState.unifiedLoopClosed)
        statusBanner = guidedEngine.activeInstruction

        guard phase == .recording, let engine = writerEngine else { return }
        let phaseStr = guidedEngine.phase.rawValue
        let instr = guidedEngine.activeInstruction
        let pDeg = pitchDeg
        let rDeg = rollDeg
        let trk = trackingStateText
        let expS = latestExposureDurationS
        let isoV = latestISO
        let trS = latestTurnRateRadS
        let blurP = latestPredictedBlurPx
        let thermS = thermalSchedule.thermalLevel.rawValue
        let covC = coveredGroundCellCount
        writerQueue.async { [weak self] in
            do {
                try engine.appendMonotonicGuidanceRow(
                    timestampS: t,
                    savedIndex: sIdx,
                    event: "end_early",
                    phase: phaseStr,
                    instruction: instr,
                    spoken: didSpeak,
                    pitchDeg: pDeg,
                    rollDeg: rDeg,
                    trackingState: trk,
                    exposureDurationS: expS,
                    iso: isoV,
                    turnRateRadS: trS,
                    predictedBlurPx: blurP,
                    thermalState: thermS,
                    coverageCells: covC
                )
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "guidance", message: error.localizedDescription)
                }
            }
        }
    }

    /// Begin a `"Fix now"` action (`YS-71`, `YS-72`, `YS-74`), unlocking `targetMarkerID` (if in `1...3`)
    /// or re-opening the look-up step (`reopenLookUp == true`), clearing `loopClosedEventTimeS` in `PhoneLoopRuleState`,
    /// and requiring a fresh closing hold on Marker 0.
    public func beginFixNow(targetMarkerID: Int? = nil, reopenLookUp: Bool = false) {
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        phoneLoopState.clearLoopClosedForFixNow()
        guidedEngine.beginFixNowRequiringReturnToMarker0(
            targetMarkerID: targetMarkerID,
            reopenLookUp: reopenLookUp,
            at: t
        )
        let didSpeak = dispatchVoiceCoachIfNeeded(guidedEngine.lastSpokenPhraseThisStep)
        guidedWalkPhase = guidedEngine.phase
        lockedMarkerIDs = guidedEngine.lockedMarkerIDs
        isLookUpDone = guidedEngine.lookUpTracker.isDone
        lookUpProgressFraction = guidedEngine.lookUpTracker.progressFraction
        isLoopClosed = phoneLoopState.unifiedLoopClosed
        isFinishAvailable = guidedEngine.isFinishAvailable(unifiedLoopClosed: phoneLoopState.unifiedLoopClosed)
        statusBanner = guidedEngine.activeInstruction
        _ = evaluateWalkReview()

        guard phase == .recording, let engine = writerEngine else { return }
        let phaseStr = guidedEngine.phase.rawValue
        let instr = guidedEngine.activeInstruction
        let pDeg = pitchDeg
        let rDeg = rollDeg
        let trk = trackingStateText
        let expS = latestExposureDurationS
        let isoV = latestISO
        let trS = latestTurnRateRadS
        let blurP = latestPredictedBlurPx
        let thermS = thermalSchedule.thermalLevel.rawValue
        let covC = coveredGroundCellCount
        writerQueue.async { [weak self] in
            do {
                try engine.appendMonotonicGuidanceRow(
                    timestampS: t,
                    savedIndex: sIdx,
                    event: "fix_now",
                    phase: phaseStr,
                    instruction: instr,
                    spoken: didSpeak,
                    pitchDeg: pDeg,
                    rollDeg: rDeg,
                    trackingState: trk,
                    exposureDurationS: expS,
                    iso: isoV,
                    turnRateRadS: trS,
                    predictedBlurPx: blurP,
                    thermalState: thermS,
                    coverageCells: covC
                )
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "guidance", message: error.localizedDescription)
                }
            }
        }
    }

    /// One-tap `Skip` for the `YS-79` house-wall and roof-edge look-up step, recorded in `guidance.csv`.
    public func skipLookUpStep() {
        let t = latestTimestampS
        let sIdx = max(0, latestSavedIndex)
        guidedEngine.skipLookUpStep(at: t)
        let didSpeak = dispatchVoiceCoachIfNeeded(guidedEngine.lastSpokenPhraseThisStep)
        guidedWalkPhase = guidedEngine.phase
        isLookUpDone = guidedEngine.lookUpTracker.isDone
        lookUpProgressFraction = guidedEngine.lookUpTracker.progressFraction
        statusBanner = guidedEngine.activeInstruction

        guard phase == .recording, let engine = writerEngine else { return }
        let phaseStr = guidedEngine.phase.rawValue
        let instr = guidedEngine.activeInstruction
        let pDeg = pitchDeg
        let rDeg = rollDeg
        let trk = trackingStateText
        let expS = latestExposureDurationS
        let isoV = latestISO
        let trS = latestTurnRateRadS
        let blurP = latestPredictedBlurPx
        let thermS = thermalSchedule.thermalLevel.rawValue
        let covC = coveredGroundCellCount
        writerQueue.async { [weak self] in
            do {
                try engine.appendMonotonicGuidanceRow(
                    timestampS: t,
                    savedIndex: sIdx,
                    event: "lookup_skipped",
                    phase: phaseStr,
                    instruction: instr,
                    spoken: didSpeak,
                    pitchDeg: pDeg,
                    rollDeg: rDeg,
                    trackingState: trk,
                    exposureDurationS: expS,
                    iso: isoV,
                    turnRateRadS: trS,
                    predictedBlurPx: blurP,
                    thermalState: thermS,
                    coverageCells: covC
                )
            } catch {
                Task { @MainActor in
                    self?.handleStreamFailure(stream: "guidance", message: error.localizedDescription)
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
            "tape_uncertainty_m": tapeSurface.uncertaintyM,
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
        detectionScheduler.scheduleAfterPendingWork { [writerQueue] in
            writerQueue.async {
                try? engine?.finishStreamsAndCloseFiles()
                Self.writeSessionJSONDataOnQueue(snapData, in: dir, registry: engine?.digestRegistry)
            }
        }
    }

    /// Evaluate the live Walk Review sheet (`YS-72`, `YS-80` #1) without stopping or pausing recording.
    @discardableResult
    public func evaluateWalkReview() -> WalkReviewReport {
        let sizeMismatchID: Int? = (liveMarkerSizeWarning != nil) ? (lastSizeMismatchMarkerID ?? 0) : nil
        let didGetDark = latestExposureDurationS > 0.025 || latestISO > 800.0
        let report = guidedEngine.evaluateWalkReview(
            unifiedLoopClosed: phoneLoopState.unifiedLoopClosed,
            coveredGroundCellCount: coveredGroundCellCount,
            minExpectedCoverageCells: 36,
            coveredGroundCells: coveredGroundCells,
            missingCoverageNearMarkerID: nil,
            sizeMismatchMarkerID: sizeMismatchID,
            didGetDark: didGetDark
        )
        latestWalkReviewReport = report
        return report
    }

    /// Execute a verb action from the live Walk Review sheet (`YS-72`, `DESIGN-SPEC-NATIVE.md` §3e, §4e),
    /// returning to the same walk and requiring a fresh closing hold at Marker 0 before saving.
    public func performWalkReviewAction(_ action: WalkReviewAction) {
        switch action {
        case .returnToMarker0, .fillCoverageGap:
            beginFixNow(targetMarkerID: nil, reopenLookUp: false)
        case .captureMissingMarker(let mid):
            beginFixNow(targetMarkerID: mid, reopenLookUp: false)
        case .performLookUp:
            beginFixNow(targetMarkerID: nil, reopenLookUp: true)
        case .checkMarkerPrintSize:
            beginFixNow(targetMarkerID: lastSizeMismatchMarkerID ?? 0, reopenLookUp: false)
        }
    }

    /// Handle a hardware Camera Control / volume button press (`YS-82` Experiment 3).
    /// Returns `.openReviewWalkFinish` when the caller should present the Review Walk sheet.
    @discardableResult
    public func handleHardwareCaptureButtonPress() -> HardwareButtonExperimentAction {
        let action = HardwareButtonExperimentRouter.resolvePress(
            isFlagEnabled: experimentFlags.hardwareButtonControl,
            isRecordingVoiceNote: isRecordingVoiceNote,
            isFinishAvailable: isFinishAvailable
        )
        switch action {
        case .startVoiceNote:
            setRecordingVoiceNote(true)
        case .stopVoiceNote:
            setRecordingVoiceNote(false)
        case .openReviewWalkFinish:
            _ = evaluateWalkReview()
        case .ignoredFlagOff:
            return action
        }
        if phase == .recording, let engine = writerEngine {
            let t = latestTimestampS
            let sIdx = savedFrameCount
            let phaseStr = guidedEngine.phase.rawValue
            let actStr = action.rawValue
            let p = pitchDeg
            let r = rollDeg
            let tr = trackingStateText
            let expS = latestExposureDurationS
            let iso = latestISO
            let trn = latestTurnRateRadS
            let blr = latestPredictedBlurPx
            let thm = thermalSchedule.thermalLevel.rawValue
            let cov = coveredGroundCellCount
            writerQueue.async {
                guard !engine.isClosed else { return }
                try? engine.appendMonotonicGuidanceRow(
                    timestampS: t,
                    savedIndex: sIdx,
                    event: "exp_hardware_button",
                    phase: phaseStr,
                    instruction: actStr,
                    spoken: false,
                    pitchDeg: p,
                    rollDeg: r,
                    trackingState: tr,
                    exposureDurationS: expS,
                    iso: iso,
                    turnRateRadS: trn,
                    predictedBlurPx: blr,
                    thermalState: thm,
                    coverageCells: cov
                )
            }
        }
        return action
    }

    /// Record the start of a short interruption when `YS-82` `pauseAndResumeWithPaddedAudio` is enabled (`off` by default).
    @discardableResult
    public func beginExperimentalPauseIfEnabled(at timestampS: Double) -> Bool {
        guard experimentFlags.pauseAndResumeWithPaddedAudio, phase == .recording else {
            return false
        }
        lastPauseStartedTimestampS = timestampS
        return true
    }

    /// Resume from a short interruption (`≤ 15.0 s`) by padding `audio.wav` with zero-PCM silence and declaring the gap in `guidance.csv` (`YS-82`).
    /// If the flag is off or the interruption exceeded `15.0 s`, seals and saves the walk via `handleInterruption`.
    @discardableResult
    public func resumeFromExperimentalPauseIfPossible(at resumeTimestampS: Double) -> PauseResumeAudioPadder.PauseResumeDecision {
        let pauseT = lastPauseStartedTimestampS ?? resumeTimestampS
        lastPauseStartedTimestampS = nil
        let decision = PauseResumeAudioPadder.evaluateResume(
            isFlagEnabled: experimentFlags.pauseAndResumeWithPaddedAudio,
            pauseTimestampS: pauseT,
            resumeTimestampS: resumeTimestampS
        )
        guard decision.canResume, phase == .recording, let engine = writerEngine else {
            handleInterruption(stopReason: "pause_exceeded_or_disabled")
            return decision
        }
        let padData = PauseResumeAudioPadder.makeZeroPCMPaddingData(sampleCount: decision.zeroPCMSampleCount)
        let sIdx = savedFrameCount
        let phaseStr = guidedEngine.phase.rawValue
        let detail = decision.guidanceEventDetail
        let p = pitchDeg
        let r = rollDeg
        let tr = trackingStateText
        let expS = latestExposureDurationS
        let iso = latestISO
        let trn = latestTurnRateRadS
        let blr = latestPredictedBlurPx
        let thm = thermalSchedule.thermalLevel.rawValue
        let cov = coveredGroundCellCount
        writerQueue.async {
            guard !engine.isClosed else { return }
            try? engine.appendAudioPaddingBytes(padData)
            try? engine.appendMonotonicGuidanceRow(
                timestampS: resumeTimestampS,
                savedIndex: sIdx,
                event: "exp_pause_resume",
                phase: phaseStr,
                instruction: detail,
                spoken: false,
                pitchDeg: p,
                rollDeg: r,
                trackingState: tr,
                exposureDurationS: expS,
                iso: iso,
                turnRateRadS: trn,
                predictedBlurPx: blr,
                thermalState: thm,
                coverageCells: cov
            )
        }
        return decision
    }

    /// Plain-language Capture Detail summary for the current session (`YS-80`, `DESIGN-SPEC-NATIVE.md` §3f).
    public var currentCaptureDetailSummary: CaptureDetailSummary {
        let statusStr: String = {
            if phase == .failed { return "failed" }
            if provisionalReasons.contains(where: { $0.hasPrefix("recording_interrupted") }) { return "interrupted" }
            return "stopped"
        }()
        let primaryTapeM: Double? = {
            if let pre = preWalkTapeMeters0To1, pre > 0 { return pre }
            for item in tapeChecks where (item["role"] as? String) == "primary" {
                if let tm = item["tape_m"] as? Double { return tm }
            }
            return nil
        }()
        return CaptureDetailSummary.build(
            provisionalGrade: provisionalGrade,
            status: statusStr,
            distanceTolerance10mM: provisionalDistanceTolerance10m,
            lockedMarkerIDs: lockedMarkerIDs,
            isLoopClosed: isLoopClosed,
            isLookUpDone: isLookUpDone,
            wasLookUpSkipped: guidedEngine.lookUpTracker.isSkipped,
            primaryTapeMeters: primaryTapeM
        )
    }

    private func appendCompletedCaptureRecord(status: String, zipURL: URL?) {
        let t0 = t0MonotonicS ?? latestTimestampS
        let durS = max(1, Int((max(t0, latestTimestampS) - t0).rounded()))
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "EEE, MMM d"
        let mins = max(1, (durS + 30) / 60)
        let dateLine = "\(dateFormatter.string(from: Date())) · \(mins) min"
        let record = SavedCaptureRecord(
            id: captureID.isEmpty ? UUID().uuidString : captureID,
            title: "Back yard",
            dateText: dateLine,
            durationSeconds: durS,
            savedFrameCount: savedFrameCount,
            coveredGroundCells: coveredGroundCells,
            coveredGroundAreaM2: coveredGroundAreaM2,
            detailSummary: currentCaptureDetailSummary,
            archiveZipURL: zipURL
        )
        savedCaptureRecords.insert(record, at: 0)
    }

    /// Reset recorder state back to `.setup` for starting a new walk from the root `Captures` screen (`YS-80`).
    public func resetToSetupForNewCapture() {
        phase = .setup
        statusBanner = "Stand over marker 0 and point at it"
        lowDiskSpaceWarning = nil
        liveMarkerSizeWarning = nil
        lastSizeMismatchMarkerID = nil
        beaconProjections = []
        _ = evaluatePreFlightChecklist()
    }

    /// Handle backgrounding or session interruption by finishing streams, writing `status: "interrupted"`,
    /// and sealing `manifest.json` + `.zip` inside a UIKit background task (`YS-08`, `YS-80` #2, `YS-84`, Blocker 1).
    public func handleInterruption(stopReason: String) {
        guard phase == .recording, let dir = captureDirURL, let engine = writerEngine else { return }
        phase = .sealing
        sealingProgress = 0.0
        sealingStatusText = "Saving your walk"
        // Keep idle timer disabled until `.zip` finishes (`YS-84`)
        UIApplication.shared.isIdleTimerDisabled = true
        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()
        guidedEngine.markWalkInterrupted(at: latestTimestampS)
        provisionalGrade = "fail"
        provisionalReasons = ["recording_interrupted: \(stopReason)"]
        // YS-80 correction #2: an interruption reads "Walk interrupted. Saved up to this point.", never "Walk paused"
        statusBanner = "Walk interrupted. Saved up to this point."

        beginSealingBackgroundTask(name: "YardstickSealInterrupted")
        let snapData = makeSessionJSONData(status: "interrupted", stopReason: stopReason, failedStream: nil)
        detectionScheduler.scheduleAfterPendingWork { [weak self] in
            guard let self else { return }
            self.writerQueue.async { [weak self] in
                guard let self else { return }
                try? engine.finishStreamsAndCloseFiles()
                Self.writeSessionJSONDataOnQueue(snapData, in: dir, registry: engine.digestRegistry)
                let precomputed = engine.digestRegistry.snapshot()
                let zipURL = try? Self.sealManifestAndPackZip(
                    in: dir,
                    precomputedDigests: precomputed
                ) { [self] frac, label in
                    Task { @MainActor [self] in
                        self.sealingProgress = frac
                        self.sealingStatusText = label
                    }
                }
                Task { @MainActor [self] in
                    self.sealingProgress = 1.0
                    self.archiveZipURL = zipURL
                    self.phase = .completed
                    self.statusBanner = "Walk interrupted. Saved up to this point."
                    self.appendCompletedCaptureRecord(status: "interrupted", zipURL: zipURL)
                    UIApplication.shared.isIdleTimerDisabled = false
                    self.endSealingBackgroundTask()
                }
            }
        }
    }

    /// Cleanly stop the recording, finish `rgb.mp4`, seal `manifest.json` using incremental digests (`YS-84`), and build the store-mode `.zip` (`YS-21`, `YS-22`, `YS-84`).
    public func stopAndSealCapture() {
        guard let dir = captureDirURL, phase == .recording, let engine = writerEngine else { return }
        dispatchHapticIfNeeded(.sessionStop)
        // Keep the screen awake with a determinate "Saving your walk" progress bar until `.zip` finishes (`YS-84`)
        UIApplication.shared.isIdleTimerDisabled = true
        logEvent(kind: "done", detail: "User tapped Done")
        phase = .sealing
        sealingProgress = 0.0
        sealingStatusText = "Saving your walk"
        statusBanner = "Saving your walk"

        #if canImport(ARKit)
        arSession.pause()
        #endif
        locationManager.stopUpdatingHeading()
        locationManager.stopUpdatingLocation()

        beginSealingBackgroundTask(name: "YardstickSealDone")
        // YS-88: the sealed gap includes every pose jump of the whole walk, exactly as the toolkit computes it.
        phoneLoopState.finalizeLoopGapForSeal()
        isLoopClosed = phoneLoopState.unifiedLoopClosed
        loopClosureGapMeters = phoneLoopState.unifiedLoopClosureGapMeters
        updateProvisionalGrade()
        let snapData = makeSessionJSONData(status: "stopped", stopReason: "user_done", failedStream: nil)

        detectionScheduler.scheduleAfterPendingWork { [weak self] in
            guard let self else { return }
            self.writerQueue.async { [weak self] in
                guard let self else { return }
                do {
                    try engine.finishStreamsAndCloseFiles()
                    Self.writeSessionJSONDataOnQueue(snapData, in: dir, registry: engine.digestRegistry)
                    let precomputed = engine.digestRegistry.snapshot()
                    let zipURL = try Self.sealManifestAndPackZip(
                        in: dir,
                        precomputedDigests: precomputed
                    ) { [self] frac, label in
                        Task { @MainActor [self] in
                            self.sealingProgress = frac
                            self.sealingStatusText = label
                        }
                    }
                    Task { @MainActor [self] in
                        self.sealingProgress = 1.0
                        self.archiveZipURL = zipURL
                        self.phase = .completed
                        self.dispatchHapticIfNeeded(.markerLocked(markerID: 0))
                        self.statusBanner = self.currentCaptureDetailSummary.verdictTitle
                        self.appendCompletedCaptureRecord(status: "stopped", zipURL: zipURL)
                        UIApplication.shared.isIdleTimerDisabled = false
                        self.endSealingBackgroundTask()
                    }
                } catch {
                    let errMsg = error.localizedDescription
                    Task { @MainActor [self] in
                        self.handleStreamFailure(stream: "manifest", message: errMsg)
                        self.endSealingBackgroundTask()
                    }
                }
            }
        }
    }

    /// Same grade as the toolkit (YS-44, YS-111): `GradeModel` is a line-for-line port of `grading.py`.
    private func updateProvisionalGrade() {
        var inp = GradeModel.Inputs()
        inp.hasMarker0 = detectedMarkerIDs.contains(0)
        inp.tapeChecks = tapeChecks.compactMap { item -> GradeModel.TapeCheck? in
            guard let tape = item["tape_m"] as? Double, let res = item["residual_m"] as? Double else { return nil }
            let role = (item["role"] as? String) ?? "primary"
            let u = (item["tape_uncertainty_m"] as? Double) ?? 0.0
            return GradeModel.TapeCheck(tapeM: tape, residualM: res, isHeldOut: role == "held_out", tapeUncertaintyM: u)
        }
        var sizes: [Int: (printedMM: Double, measuredMM: [Double])] = [:]
        for s in phoneLoopState.allSightings where s.sighting.sizeMMMeasured > 0 {
            let mid = s.sighting.markerID
            var e = sizes[mid] ?? (s.sighting.sizeMMPrinted, [])
            e.measuredMM.append(s.sighting.sizeMMMeasured)
            sizes[mid] = e
        }
        inp.scaleErrorEff = GradeModel.scaleErrorEff(sizesByMarker: sizes)
        inp.rawMarker0GapM = loopClosureGapMeters
        inp.poseJumpSumM = phoneLoopState.poseJumpSumM
        inp.walkLengthM = phoneLoopState.walkLengthM
        inp.limitedTrackingShare = phoneLoopState.limitedTrackingShare

        let result = GradeModel.grade(inp)
        provisionalDistanceTolerance10m = result.distanceTolerance10mM
        provisionalGrade = result.grade
        provisionalReasons = result.reasons
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
            "app_version": "0.2.0",
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

    nonisolated private static func writeSessionJSONDataOnQueue(
        _ data: Data?,
        in dir: URL,
        registry: PrecomputedDigestRegistry? = nil
    ) {
        guard let data else { return }
        do {
            try data.write(to: dir.appendingPathComponent("session.json"), options: .atomic)
            if let registry {
                let meta = IncrementalByteDigest.compute(for: data)
                registry.record(meta, forRelativePath: "session.json")
            }
        } catch {
            // Ignored on heartbeat; final seal checks files directly
        }
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

    /// Compute the relative path of `fileURL` inside `baseDir`, resolving symlinks (`/private/var` vs `/var` on iOS)
    /// on both URLs so prefix stripping never produces a mangled path (`YS-65`).
    nonisolated public static func relativePath(of fileURL: URL, inside baseDir: URL) -> String? {
        let baseResolved = baseDir.resolvingSymlinksInPath().standardizedFileURL.path
        let fileResolved = fileURL.resolvingSymlinksInPath().standardizedFileURL.path
        let prefix = baseResolved.hasSuffix("/") ? baseResolved : (baseResolved + "/")
        guard fileResolved.hasPrefix(prefix) else { return nil }
        let rel = String(fileResolved.dropFirst(prefix.count))
        return rel.isEmpty ? nil : rel
    }

    /// Seal `manifest.json` and build the store-mode `.zip` archive (`YS-21`, `YS-22`, `YS-84`).
    ///
    /// When `precomputedDigests` is provided (`YS-84`), files whose incremental SHA-256, CRC-32, and
    /// byte size were computed as they were written are never re-opened or re-read during the manifest pass.
    nonisolated public static func sealManifestAndPackZip(
        in dir: URL,
        precomputedDigests: [String: FileDigestMeta] = [:],
        progressHandler: (@Sendable (Double, String) -> Void)? = nil
    ) throws -> URL {
        progressHandler?(0.02, "Saving your walk")
        let resolvedDir = dir.resolvingSymlinksInPath().standardizedFileURL
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: resolvedDir, includingPropertiesForKeys: [.isRegularFileKey])
        var filesArray: [[String: Any]] = []
        var precomputedCRCByRelPath: [String: (crc32: UInt32, sizeBytes: UInt64)] = [:]

        while let fileURL = enumerator?.nextObject() as? URL {
            guard let rel = Self.relativePath(of: fileURL, inside: resolvedDir) else { continue }
            if rel == "manifest.json" || rel.hasPrefix("standard/") || rel.hasPrefix(".") {
                continue
            }
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard vals.isRegularFile == true else { continue }

            let hex: String
            let crc: UInt32
            let sizeBytes: UInt64
            if let pre = precomputedDigests[rel] {
                hex = pre.sha256
                crc = pre.crc32
                sizeBytes = pre.sizeBytes
            } else {
                let computed = try Self.streamHashAndCRC32(of: fileURL)
                hex = computed.sha256
                crc = computed.crc32
                sizeBytes = computed.sizeBytes
            }
            filesArray.append([
                "path": rel,
                "size_bytes": sizeBytes,
                "sha256": hex,
            ])
            precomputedCRCByRelPath[rel] = (crc, sizeBytes)
        }
        filesArray.sort { ($0["path"] as? String ?? "") < ($1["path"] as? String ?? "") }
        progressHandler?(0.15, "Saving your walk")

        var streamsDeclared = [
            "session", "frames", "rgb_video", "depth", "confidence",
            "poses", "audio", "events", "stills", "markers",
        ]
        if precomputedCRCByRelPath["guidance.csv"] != nil {
            streamsDeclared.append("guidance")
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let manifestObj: [String: Any] = [
            "contract": "yardstick-capture/1",
            "capture_id": resolvedDir.lastPathComponent,
            "sealed_at_utc": formatter.string(from: Date()),
            "streams_declared": streamsDeclared,
            "files": filesArray,
        ]
        let manifestURL = resolvedDir.appendingPathComponent("manifest.json")
        let manifestData = try JSONSerialization.data(withJSONObject: manifestObj, options: [.prettyPrinted, .sortedKeys])
        try manifestData.write(to: manifestURL, options: .atomic)
        let manifestMeta = IncrementalByteDigest.compute(for: manifestData)
        precomputedCRCByRelPath["manifest.json"] = (manifestMeta.crc32, manifestMeta.sizeBytes)

        let zipURL = resolvedDir.deletingLastPathComponent().appendingPathComponent("\(resolvedDir.lastPathComponent).zip")
        try Self.writeStoreModeZipArchive(
            sourceDir: resolvedDir,
            destinationZipURL: zipURL,
            precomputedMeta: precomputedCRCByRelPath,
            progressHandler: { zipFrac, label in
                let combined = 0.15 + 0.85 * min(1.0, max(0.0, zipFrac))
                progressHandler?(combined, label)
            }
        )
        progressHandler?(1.0, "Saving your walk")
        return zipURL
    }

    /// Encode 16-bit depth millimetres (`[UInt16]`) into in-memory PNG `Data` (`YS-04`, `YS-84`).
    nonisolated public static func encodeDepth16BitPNGData(_ pixels: [UInt16], width: Int, height: Int) -> Data? {
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
              let outData = CFDataCreateMutable(kCFAllocatorDefault, 0),
              let dest = CGImageDestinationCreateWithData(outData, UTType.png.identifier as CFString, 1, nil)
        else {
            return nil
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return outData as Data
    }

    /// Encode 16-bit depth millimetres (`[UInt16]`) into a 16-bit linear grayscale PNG and optionally
    /// record its incremental SHA-256 + CRC-32 digest in memory (`YS-04`, `YS-84`).
    @discardableResult
    nonisolated public static func writeDepth16BitPNG(
        _ pixels: [UInt16],
        width: Int,
        height: Int,
        to url: URL,
        registry: PrecomputedDigestRegistry? = nil,
        relativePath: String? = nil
    ) -> Bool {
        guard let pngData = encodeDepth16BitPNGData(pixels, width: width, height: height) else {
            return false
        }
        do {
            try pngData.write(to: url)
            if let registry, let relativePath {
                let meta = IncrementalByteDigest.compute(for: pngData)
                registry.record(meta, forRelativePath: relativePath)
            }
            return true
        } catch {
            return false
        }
    }

    /// Encode 8-bit LiDAR confidence levels (`0, 1, 2`) into in-memory PNG `Data` (`YS-04`, `YS-84`).
    nonisolated public static func encodeConfidence8BitPNGData(_ pixels: [UInt8], width: Int, height: Int) -> Data? {
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
              let outData = CFDataCreateMutable(kCFAllocatorDefault, 0),
              let dest = CGImageDestinationCreateWithData(outData, UTType.png.identifier as CFString, 1, nil)
        else {
            return nil
        }
        CGImageDestinationAddImage(dest, cgImg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return outData as Data
    }

    /// Encode 8-bit LiDAR confidence levels (`0, 1, 2`) into an 8-bit linear grayscale PNG and optionally
    /// record its incremental SHA-256 + CRC-32 digest in memory (`YS-04`, `YS-84`).
    @discardableResult
    nonisolated public static func writeConfidence8BitPNG(
        _ pixels: [UInt8],
        width: Int,
        height: Int,
        to url: URL,
        registry: PrecomputedDigestRegistry? = nil,
        relativePath: String? = nil
    ) -> Bool {
        guard let pngData = encodeConfidence8BitPNGData(pixels, width: width, height: height) else {
            return false
        }
        do {
            try pngData.write(to: url)
            if let registry, let relativePath {
                let meta = IncrementalByteDigest.compute(for: pngData)
                registry.record(meta, forRelativePath: relativePath)
            }
            return true
        } catch {
            return false
        }
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

    /// Streaming, table-CRC, ZIP64-capable `ZIP_STORED` (`compressionMethod = 0`) archiver (`YS-22`, `YS-84`, Blocker 4).
    nonisolated public static func writeStoreModeZipArchive(
        sourceDir: URL,
        destinationZipURL: URL,
        precomputedMeta: [String: (crc32: UInt32, sizeBytes: UInt64)] = [:],
        progressHandler: (@Sendable (Double, String) -> Void)? = nil
    ) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destinationZipURL.path) {
            try fm.removeItem(at: destinationZipURL)
        }
        fm.createFile(atPath: destinationZipURL.path, contents: nil)
        let dstFH = try FileHandle(forWritingTo: destinationZipURL)
        defer { try? dstFH.close() }

        let resolvedSourceDir = sourceDir.resolvingSymlinksInPath().standardizedFileURL
        let baseFolderName = resolvedSourceDir.lastPathComponent
        let enumerator = fm.enumerator(at: resolvedSourceDir, includingPropertiesForKeys: [.isRegularFileKey])
        var relPaths: [String] = []
        while let fileURL = enumerator?.nextObject() as? URL {
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            if vals.isRegularFile == true,
               let rel = Self.relativePath(of: fileURL, inside: resolvedSourceDir),
               !rel.hasPrefix(".") {
                relPaths.append(rel)
            }
        }
        relPaths.sort()
        let totalFiles = max(1, relPaths.count)

        struct CentralRecord {
            let nameData: Data
            let crc32: UInt32
            let size: UInt64
            let localHeaderOffset: UInt64
            let usesZip64: Bool
        }
        var centralRecords: [CentralRecord] = []
        var offset: UInt64 = 0

        for (fileIdx, rel) in relPaths.enumerated() {
            let fileURL = resolvedSourceDir.appendingPathComponent(rel)
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
            if fileIdx % 32 == 0 || fileIdx + 1 == totalFiles {
                progressHandler?(Double(fileIdx + 1) / Double(totalFiles), "Saving your walk")
            }
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

    nonisolated private static func extractCameraISO(from frame: ARFrame) -> Double {
        if let exifDict = CMGetAttachment(frame.capturedImage, key: kCGImagePropertyExifDictionary, attachmentModeOut: nil) as? [CFString: Any] {
            if let isoArr = exifDict[kCGImagePropertyExifISOSpeedRatings] as? [NSNumber], let firstISO = isoArr.first?.doubleValue, firstISO > 0 {
                return max(16.0, min(12800.0, firstISO))
            }
            if let isoNum = exifDict[kCGImagePropertyExifISOSpeedRatings] as? NSNumber, isoNum.doubleValue > 0 {
                return max(16.0, min(12800.0, isoNum.doubleValue))
            }
        }
        return max(32.0, min(6400.0, 100.0 * pow(2.0, Double(frame.camera.exposureOffset))))
    }

    nonisolated public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let timestampS = frame.timestamp
        let cTransform = frame.camera.transform
        let intrinsics = frame.camera.intrinsics
        let deliveredW = CVPixelBufferGetWidth(frame.capturedImage)
        let deliveredH = CVPixelBufferGetHeight(frame.capturedImage)
        let exposureDurationS = max(1e-5, Double(frame.camera.exposureDuration))
        let isoVal = Self.extractCameraISO(from: frame)

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
        var frameValidDepthSampleCount = 0

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
                        let confVal = min(2, cRow[u])
                        cArr[baseIdx + u] = confVal
                        if z.isFinite && z > 0.0 && z < 65.535 {
                            let mm = UInt16(clamping: Int((z * 1000.0).rounded()))
                            dArr[baseIdx + u] = mm
                            if (v & 7 == 0) && (u & 7 == 0) && confVal >= 1 && mm >= 500 && mm <= 4500 {
                                frameValidDepthSampleCount += 1
                            }
                        }
                    }
                }
                depthCopyMM = dArr
                confCopyU8 = cArr

                // 5x5 center crosshair depth patch (`YS-40`, `YS-79` — requires confidence >= 1 so low-confidence
                // sky fill never masks `"Point at the roof edge, not the sky"`)
                var validDepths: [UInt16] = []
                let uC = depthW / 2
                let vC = depthH / 2
                for dv in -2...2 {
                    let v = min(depthH - 1, max(0, vC + dv))
                    for du in -2...2 {
                        let u = min(depthW - 1, max(0, uC + du))
                        let idx = v * depthW + u
                        let mm = dArr[idx]
                        let conf = cArr[idx]
                        if conf >= 1 && mm >= 200 && mm <= 4500 {
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
        let finalFrameValidDepthCount = frameValidDepthSampleCount

        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.phase == .setup {
                if timestampS - self.lastPreFlightProbeTimeS >= 0.50 {
                    let prevT = self.lastDeliveredFrameTimestampS
                    let prevTransform = (prevT != nil) ? self.latestCameraTransform : nil
                    let dt = prevT.map { max(1e-4, timestampS - $0) } ?? (1.0 / 30.0)
                    self.lastDeliveredFrameTimestampS = timestampS
                    self.lastPreFlightProbeTimeS = timestampS
                    self.latestCameraTransform = cMat
                    self.latestExposureDurationS = exposureDurationS
                    self.latestISO = isoVal

                    let fx = Double(intrinsics.columns.0.x)
                    let fy = Double(intrinsics.columns.1.y)
                    let meanFocalPx = 0.5 * (fx + fy)
                    let turnRateRadS = ExposureAndStillnessGate.computeTurnRateRadS(
                        previousTransform: prevTransform,
                        currentTransform: cMat,
                        dt: dt
                    )
                    let predictedBlurPx = ExposureAndStillnessGate.computePredictedBlurPx(
                        focalLengthPx: meanFocalPx,
                        turnRateRadS: turnRateRadS,
                        exposureDurationS: exposureDurationS
                    )
                    let camVelMps: Double = {
                        guard let prevM = prevTransform else { return 0.0 }
                        let prevPos = SIMD3<Double>(prevM.columns.3.x, prevM.columns.3.y, prevM.columns.3.z)
                        let curPos = SIMD3<Double>(cMat.columns.3.x, cMat.columns.3.y, cMat.columns.3.z)
                        return simd_length(curPos - prevPos) / dt
                    }()
                    self.latestTurnRateRadS = turnRateRadS
                    self.latestPredictedBlurPx = predictedBlurPx

                    if let sendableRGB = copiedRGBBufferOpt,
                       let luma = Self.extractLumaBytes(from: sendableRGB.buffer) {
                        _ = self.updateLensSmudgeProbe(
                            lumaBytes: luma,
                            width: deliveredW,
                            height: deliveredH,
                            turnRateRadS: turnRateRadS,
                            predictedBlurPx: predictedBlurPx,
                            cameraVelocityMps: camVelMps
                        )
                    } else {
                        _ = self.evaluatePreFlightChecklist()
                    }
                }
                return
            }
            guard self.phase == .recording, let dir = self.captureDirURL, let engine = self.writerEngine else { return }

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
            let prevTimestampS = self.lastDeliveredFrameTimestampS
            let prevTransform = (prevTimestampS != nil) ? self.latestCameraTransform : nil
            let frameDt = prevTimestampS.map { max(1e-4, timestampS - $0) } ?? expectedDt
            self.lastDeliveredFrameTimestampS = timestampS

            let fx = Double(intrinsics.columns.0.x)
            let fy = Double(intrinsics.columns.1.y)
            let cx = Double(intrinsics.columns.2.x)
            let cy = Double(intrinsics.columns.2.y)
            let meanFocalPx = 0.5 * (fx + fy)
            let turnRateRadS = ExposureAndStillnessGate.computeTurnRateRadS(
                previousTransform: prevTransform,
                currentTransform: cMat,
                dt: frameDt
            )
            let predictedBlurPx = ExposureAndStillnessGate.computePredictedBlurPx(
                focalLengthPx: meanFocalPx,
                turnRateRadS: turnRateRadS,
                exposureDurationS: exposureDurationS
            )

            self.latestTimestampS = timestampS
            self.latestCameraTransform = cMat
            self.pitchDeg = pitchDegrees
            self.rollDeg = rollDegrees
            self.trackingStateText = trackingStr
            self.crosshairDepthMeters = finalCenterDepthM
            self.latestCrosshairWorld = finalCenterWorldPt
            self.latestExposureDurationS = exposureDurationS
            self.latestISO = isoVal
            self.latestTurnRateRadS = turnRateRadS
            self.latestPredictedBlurPx = predictedBlurPx
            self.latestValidDepthSamplesCount = finalFrameValidDepthCount

            let activeThermalLevel = self.injectedThermalLevelForTesting ?? CaptureThermalLevel.fromSystem(ProcessInfo.processInfo.thermalState)
            let effectiveThermalStr = activeThermalLevel.rawValue
            let prevInstructionKey = self.guidedEngine.activeInstructionKey
            let stepOut = self.guidedEngine.step(
                timestampS: timestampS,
                savedIndex: self.savedFrameCount,
                dt: frameDt,
                trackingState: trackingStr,
                trackingReason: reasonStr,
                pitchDeg: pitchDegrees,
                turnRateRadS: turnRateRadS,
                predictedBlurPx: predictedBlurPx,
                cameraPositionWorld: SIMD3<Double>(cMat.columns.3.x, cMat.columns.3.y, cMat.columns.3.z),
                confirmedSightings: [],
                rawSightings: [],
                unifiedLoopClosed: self.phoneLoopState.unifiedLoopClosed,
                thermalLevel: activeThermalLevel,
                lowDiskSpaceWarning: self.lowDiskSpaceWarning != nil,
                hasValidDepthForLookUp: finalFrameValidDepthCount >= 8 || finalCenterDepthM != nil,
                elevatedPointsAbove2m: self.latestElevatedPointsAbove2m,
                didRunMarkerDetection: false
            )
            self.latestElevatedPointsAbove2m = 0
            self.guidedWalkPhase = self.guidedEngine.phase
            self.activeRingState = self.guidedEngine.activeRingState
            self.lockedMarkerIDs = self.guidedEngine.lockedMarkerIDs
            self.isFinishAvailable = self.guidedEngine.isFinishAvailable(unifiedLoopClosed: self.phoneLoopState.unifiedLoopClosed)
            self.lookUpProgressFraction = self.guidedEngine.lookUpTracker.progressFraction
            self.isLookUpDone = self.guidedEngine.lookUpTracker.isDone
            self.thermalSchedule = stepOut.thermalSchedule
            let isHoldCountingNow = (self.activeRingState != nil && !(self.activeRingState?.isLocked ?? true))
            let expCapStep = self.exposureCapController.step(
                timestampS: timestampS,
                isFlagEnabled: self.experimentFlags.exposureCapDuringHolds,
                isMarkerHoldActive: isHoldCountingNow,
                trackingState: trackingStr
            )
            if expCapStep.didViolateSafetyWindow {
                self.experimentFlags.exposureCapDuringHolds = false
                if let engine = self.writerEngine {
                    let violSIdx = self.savedFrameCount
                    let violPhase = self.guidedEngine.phase.rawValue
                    self.writerQueue.async {
                        guard !engine.isClosed else { return }
                        try? engine.appendMonotonicGuidanceRow(
                            timestampS: timestampS,
                            savedIndex: violSIdx,
                            event: "exp_exposure_cap_violated",
                            phase: violPhase,
                            instruction: "auto_disabled_limited_tracking",
                            spoken: false,
                            pitchDeg: pitchDegrees,
                            rollDeg: rollDegrees,
                            trackingState: trackingStr,
                            exposureDurationS: exposureDurationS,
                            iso: isoVal,
                            turnRateRadS: turnRateRadS,
                            predictedBlurPx: predictedBlurPx,
                            thermalState: effectiveThermalStr
                        )
                    }
                }
            }
            var newBeacons: [MarkerBeaconProjection] = []
            for mid in self.guidedEngine.knownMarkerPositionsWorld.keys.sorted() {
                guard self.activeRingState?.markerID != mid,
                      let posW = self.guidedEngine.knownMarkerPositionsWorld[mid]
                else { continue }
                let isDone = self.guidedEngine.lockedMarkerIDs.contains(mid)
                let proj = MarkerScreenProjector.project(
                    markerID: mid,
                    markerWorldPos: posW,
                    isDone: isDone,
                    cameraToWorld: cMat,
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    rgbWidth: Double(deliveredW),
                    rgbHeight: Double(deliveredH)
                )
                let isNextNeededTarget: Bool = {
                    if self.guidedEngine.phase == .closingMarker0 && !self.guidedEngine.isClosingMarker0Locked {
                        return mid == 0
                    }
                    return !isDone
                }()
                if proj.isOnScreen || isNextNeededTarget {
                    newBeacons.append(proj)
                }
            }
            self.beaconProjections = newBeacons
            self.dispatchHapticIfNeeded(stepOut.haptic)
            let didSpeakFrame = self.dispatchVoiceCoachIfNeeded(self.guidedEngine.lastSpokenPhraseThisStep)
            self.statusBanner = stepOut.instruction
            let currentInstruction = self.statusBanner
            let currentGuidedPhaseStr = self.guidedEngine.phase.rawValue
            let detectionStride = stepOut.thermalSchedule.detectionFrameStrideAt30FPS
            let coverageMinIntervalS = stepOut.thermalSchedule.coverageMinIntervalS
            let lockedMarkersSnapshot = self.guidedEngine.lockedMarkerIDs
            let isClosingM0LockedSnapshot = self.guidedEngine.isClosingMarker0Locked
            let guidedPhaseSnapshot = self.guidedEngine.phase
            let emittedSafetyEvent: String? = {
                guard stepOut.instructionKey != prevInstructionKey else { return nil }
                if stepOut.instructionKey == "thermal_serious" { return "thermal_serious" }
                if stepOut.instructionKey == "storage_low" { return "storage_low" }
                return nil
            }()

            // Track camera excursion from Marker 0 Pass 1 for multi-pass loop closure separation (`YS-66`, `YS-71`)
            let camPosW = SIMD3<Double>(cMat.columns.3.x, cMat.columns.3.y, cMat.columns.3.z)
            self.phoneLoopState.updateCameraExcursion(
                cameraPositionWorld: camPosW,
                savedIndex: self.savedFrameCount,
                timestampS: timestampS,
                trackingNormal: trackingStr == "normal"
            )

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

            // Atomic writer-side backpressure guard (`YS-68`) — never decremented through `@MainActor`
            let (queueOverloaded, _) = self.writerBacklog.acquireSlot()
            let forceArUco = self.pendingForceMarkerDetect
            self.pendingForceMarkerDetect = false
            let sizesSnapshot = self.markerSizesMM
            let hiResLockExperimentEnabled = self.experimentFlags.hiResStillAtLock

            let rawQuat = simd_quatd(cMat)
            let normQuat = simd_normalize(rawQuat)
            let rgbW = deliveredW
            let rgbH = deliveredH
            let wD = finalDepthW
            let hD = finalDepthH

            let sessionHeartbeatData: Data?
            if timestampS - self.lastHeartbeatS >= 5.0 {
                self.lastHeartbeatS = timestampS
                sessionHeartbeatData = self.makeSessionJSONData(status: "recording", stopReason: nil, failedStream: nil)
                _ = self.checkMidWalkDiskSpace(recordedBytes: Int64(engine.digestRegistry.totalRecordedBytes))
            } else {
                sessionHeartbeatData = nil
            }

            self.writerQueue.async { [weak self] in
                guard let self else { return }
                defer { self.writerBacklog.releaseSlot() }
                guard !engine.isClosed else { return }
                let sp = Self.writerSignposter
                let jobState = sp.beginInterval("WriterJob")
                defer { sp.endInterval("WriterJob", jobState) }

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
                            self.droppedFrameCount = max(self.droppedFrameCount, totalDropped)
                        }
                        return
                    }

                    if queueOverloaded {
                        let dropLine = String(format: "%.6f,,true,writer_queue_full,%@,%@\n", timestampS, trackingStr, reasonStr)
                        try engine.appendLine(dropLine, to: engine.framesFH)
                        engine.droppedFrameCount += 1
                        Task { @MainActor in
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
                    let videoState = sp.beginInterval("VideoAppend")
                    let appendedOK = adaptor.append(rgbPB, withPresentationTime: pts)
                    sp.endInterval("VideoAppend", videoState)
                    guard appendedOK else {
                        throw writer.error ?? NSError(domain: "Yardstick", code: -32, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter adaptor.append failed"])
                    }

                    let sIdx = engine.savedFrameCount
                    engine.savedFrameCount += 1

                    let csvState = sp.beginInterval("CSVAppend")
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

                    try engine.appendMonotonicGuidanceRow(
                        timestampS: timestampS,
                        savedIndex: sIdx,
                        event: "frame",
                        phase: currentGuidedPhaseStr,
                        instruction: currentInstruction,
                        spoken: didSpeakFrame && (emittedSafetyEvent == nil),
                        pitchDeg: pitchDegrees,
                        rollDeg: rollDegrees,
                        trackingState: trackingStr,
                        exposureDurationS: exposureDurationS,
                        iso: isoVal,
                        turnRateRadS: turnRateRadS,
                        predictedBlurPx: predictedBlurPx,
                        thermalState: effectiveThermalStr
                    )
                    if let safetyEv = emittedSafetyEvent {
                        try engine.appendMonotonicGuidanceRow(
                            timestampS: timestampS,
                            savedIndex: sIdx,
                            event: safetyEv,
                            phase: currentGuidedPhaseStr,
                            instruction: currentInstruction,
                            spoken: didSpeakFrame,
                            pitchDeg: pitchDegrees,
                            rollDeg: rollDegrees,
                            trackingState: trackingStr,
                            exposureDurationS: exposureDurationS,
                            iso: isoVal,
                            turnRateRadS: turnRateRadS,
                            predictedBlurPx: predictedBlurPx,
                            thermalState: effectiveThermalStr
                        )
                    }
                    sp.endInterval("CSVAppend", csvState)

                    let pngState = sp.beginInterval("DepthConfidencePNG")
                    let depthRelPath = String(format: "depth/%06d.png", sIdx)
                    let confRelPath = String(format: "confidence/%06d.png", sIdx)
                    let depthURL = dir.appendingPathComponent(depthRelPath)
                    let confURL = dir.appendingPathComponent(confRelPath)
                    let depthOK = Self.writeDepth16BitPNG(
                        dPixels,
                        width: wD,
                        height: hD,
                        to: depthURL,
                        registry: engine.digestRegistry,
                        relativePath: depthRelPath
                    )
                    let confOK = depthOK ? Self.writeConfidence8BitPNG(
                        cPixels,
                        width: wD,
                        height: hD,
                        to: confURL,
                        registry: engine.digestRegistry,
                        relativePath: confRelPath
                    ) : false
                    sp.endInterval("DepthConfidencePNG", pngState)
                    guard depthOK else {
                        throw NSError(domain: "Yardstick", code: -33, userInfo: [NSLocalizedDescriptionKey: "Failed to write \(depthURL.lastPathComponent)"])
                    }
                    guard confOK else {
                        throw NSError(domain: "Yardstick", code: -34, userInfo: [NSLocalizedDescriptionKey: "Failed to write \(confURL.lastPathComponent)"])
                    }

                    if let sessionHeartbeatData {
                        Self.writeSessionJSONDataOnQueue(sessionHeartbeatData, in: dir, registry: engine.digestRegistry)
                    }

                    let newSavedCount = engine.savedFrameCount
                    Task { @MainActor in
                        self.latestSavedIndex = sIdx
                        self.savedFrameCount = newSavedCount
                    }

                    // Offload ArUco detection, 3D plane measurement, stillness/distance gate (`YS-73`),
                    // rate-limited still gate (`YS-68`), OpenCV JPEG confirmation (`YS-70`), and 2 Hz ground coverage
                    // (`YS-78`, adapted by `YS-85` thermal schedule) onto the single-slot `detectionScheduler`!
                    let shouldRunArUco = (sIdx % detectionStride == 0) || forceArUco
                    let isSteadyEnoughForStill = (predictedBlurPx <= ExposureAndStillnessGate.maxPredictedBlurPx)
                        && (turnRateRadS <= ExposureAndStillnessGate.maxTurnRateRadS)
                    let shouldConsiderPeriodicStill = isSteadyEnoughForStill && ((timestampS - engine.lastAnyStillTimeS) >= 2.0)
                    let shouldConsiderCoverage = engine.isCoverageUpdateDue(at: timestampS, minIntervalS: coverageMinIntervalS)
                    if shouldRunArUco || shouldConsiderPeriodicStill || shouldConsiderCoverage {
                        self.detectionScheduler.submitIfIdle { [weak self] in
                            guard let self, !engine.isClosed else { return }
                            let detectState = sp.beginInterval("ArUcoDetectAndMeasure")
                            var rawDetected3D: [DetectedMarkerSighting] = []
                            var eligibleSightings: [DetectedMarkerSighting] = []
                            var sharpnessVal: Double? = nil

                            if (shouldRunArUco || shouldConsiderPeriodicStill),
                               let luma = Self.extractLumaBytes(from: sendableRGB.buffer) {
                                if shouldRunArUco {
                                    let dets2D = ArUcoDetector.detectMarkers2D(lumaBytes: luma, width: rgbW, height: rgbH)
                                    if dets2D.isEmpty {
                                        CaptureWriterEngine.appendDetectionDebug(captureDir: dir, line: String(format: "%.3f,%d,no_marker_2d,,,,,,,,%.3f,%.2f,%@,,,,,", timestampS, sIdx, turnRateRadS, predictedBlurPx, trackingStr))
                                    }
                                    for d2 in dets2D {
                                        let pSize = sizesSnapshot[d2.markerID] ?? 250.0
                                        var didMeasure3D = false
                                        defer {
                                            if !didMeasure3D {
                                                CaptureWriterEngine.appendDetectionDebug(captureDir: dir, line: String(format: "%.3f,%d,seen_2d_no_3d,%d,false,,%.1f,,,,%.3f,%.2f,%@,,,,,", timestampS, sIdx, d2.markerID, pSize, turnRateRadS, predictedBlurPx, trackingStr))
                                            }
                                        }
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
                                            rawDetected3D.append(s3D)
                                            didMeasure3D = true
                                            // Anchor Marker 0 ground plane BEFORE updating ground coverage (`YS-78`)
                                            if s3D.markerID == 0 {
                                                let mq = simd_quatd(ix: s3D.quaternionXYZW.x, iy: s3D.quaternionXYZW.y, iz: s3D.quaternionXYZW.z, r: s3D.quaternionXYZW.w)
                                                let m0XAxis = simd_act(mq, SIMD3<Double>(1.0, 0.0, 0.0))
                                                engine.anchorMarker0Frame(originWorld: s3D.positionWorld, xAxisWorld: m0XAxis)
                                            }
                                            let rngXZ = hypot(
                                                camPosW.x - s3D.positionWorld.x,
                                                camPosW.z - s3D.positionWorld.z
                                            )
                                            let verdict = ExposureAndStillnessGate.evaluateMarkerSighting(
                                                predictedBlurPx: predictedBlurPx,
                                                turnRateRadS: turnRateRadS,
                                                rangeMeters3D: s3D.rangeMeters,
                                                horizontalRangeMeters: rngXZ
                                            )
                                            let angleOK = s3D.viewingAngleDeg <= GuidedCaptureEngine.maxViewingAngleDeg
                                            let sizeOK = s3D.sizeMMPrinted > 0
                                                ? (abs(s3D.sizeMMMeasured - s3D.sizeMMPrinted) / s3D.sizeMMPrinted) <= GuidedCaptureEngine.maxSizeDeviationRatio
                                                : true
                                            let isUnlockedPhaseTarget: Bool = {
                                                switch guidedPhaseSnapshot {
                                                case .openingMarker0:
                                                    return s3D.markerID == 0 && !lockedMarkersSnapshot.contains(0)
                                                case .markers1To3:
                                                    return (1...3).contains(s3D.markerID) && !lockedMarkersSnapshot.contains(s3D.markerID)
                                                case .houseWallLookUp, .closingMarker0, .readyToFinish:
                                                    return s3D.markerID == 0 && !isClosingM0LockedSnapshot
                                                }
                                            }()
                                            let isEligible = (trackingStr == "normal" && verdict.passes && angleOK && sizeOK && isUnlockedPhaseTarget) || forceArUco
                                            if isEligible {
                                                eligibleSightings.append(s3D)
                                            }
                                            CaptureWriterEngine.appendDetectionDebug(captureDir: dir, line: String(
                                                format: "%.3f,%d,seen_3d,%d,true,%.2f,%.1f,%.3f,%.3f,%.1f,%.3f,%.2f,%@,%@,%@,%@,%@,%@",
                                                timestampS, sIdx, s3D.markerID, s3D.sizeMMMeasured, s3D.sizeMMPrinted, s3D.rangeMeters, rngXZ,
                                                s3D.viewingAngleDeg, turnRateRadS, predictedBlurPx, trackingStr,
                                                verdict.reason ?? "pass", angleOK ? "true" : "false", sizeOK ? "true" : "false",
                                                isUnlockedPhaseTarget ? "true" : "false", isEligible ? "true" : "false"))
                                        }
                                    }
                                }
                                if eligibleSightings.isEmpty && shouldConsiderPeriodicStill {
                                    sharpnessVal = ArUcoDetector.computeLaplacianSharpness(lumaBytes: luma, width: rgbW, height: rgbH)
                                }
                            }
                            sp.endInterval("ArUcoDetectAndMeasure", detectState)

                            let covResult = engine.updateCoverageGridIfDue(
                                timestampS: timestampS,
                                minIntervalS: coverageMinIntervalS,
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
                            )

                            let gateDecision = engine.evaluateStillGate(
                                timestampS: timestampS,
                                detectedMarkerIDs: eligibleSightings.map(\.markerID),
                                forceOperatorMarkerTap: forceArUco,
                                sharpness: sharpnessVal
                            )

                            // Encode JPEG, compute SHA-256 + CRC-32, and re-confirm markers on the encoded JPEG bytes (`YS-68`, `YS-70`, `YS-84`).
                            // Under YS-70: No row goes in `markers.csv`, and no marker-triggered still goes in `stills.csv`,
                            // unless OpenCV confirms it on the saved JPEG frame!
                            var encodedStill: (relPath: String, data: Data, meta: FileDigestMeta, csvLine: String)? = nil
                            var confirmedSightings: [DetectedMarkerSighting] = []

                            if gateDecision.shouldSave {
                                let stillState = sp.beginInterval("StillJPEGEncode")
                                let stillRelPath = String(format: "stills/%06d.jpg", sIdx)
                                let ciImg = CIImage(cvPixelBuffer: sendableRGB.buffer)
                                let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
                                let jpegQual = HiResLockStillExperiment.jpegQuality(
                                    isFlagEnabled: hiResLockExperimentEnabled,
                                    didLockMarkerOnThisStill: gateDecision.reason == "marker_hold"
                                )
                                if let jpegData = engine.detectionCIContext.jpegRepresentation(
                                    of: ciImg,
                                    colorSpace: cs,
                                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: jpegQual]
                                ) {
                                    let isMarkerTriggered = (gateDecision.reason == "marker_hold") || !eligibleSightings.isEmpty
                                    if isMarkerTriggered {
                                        let jpegConfirmed = ArUcoDetector.confirmSightingsInJPEGData(
                                            jpegData,
                                            printedSizesByMarker: sizesSnapshot,
                                            defaultPrintedSizeMM: 250.0,
                                            depthMapMM: dPixels,
                                            confidenceMap: cPixels,
                                            depthWidth: wD,
                                            depthHeight: hD,
                                            fx: fx,
                                            fy: fy,
                                            cx: cx,
                                            cy: cy,
                                            cameraToWorld: cMat
                                        )
                                        let eligibleIDSet = Set(eligibleSightings.map(\.markerID))
                                        confirmedSightings = jpegConfirmed.filter { eligibleIDSet.contains($0.markerID) }
                                        CaptureWriterEngine.appendDetectionDebug(captureDir: dir, line: String(
                                            format: "%.3f,%d,jpeg_confirm,%@,,,,,,,%.3f,%.2f,%@,eligible=%@ confirmed=%@ gate=%@,,,,",
                                            timestampS, sIdx, eligibleSightings.map { String($0.markerID) }.joined(separator: "|"),
                                            turnRateRadS, predictedBlurPx, trackingStr,
                                            eligibleSightings.map { String($0.markerID) }.joined(separator: "|"),
                                            confirmedSightings.map { String($0.markerID) }.joined(separator: "|"),
                                            gateDecision.reason ?? "none"))
                                    }

                                    // Write the still if it is a periodic still OR if at least one marker was confirmed on the JPEG (`YS-70`)
                                    if gateDecision.reason != "marker_hold" || !confirmedSightings.isEmpty {
                                        let meta = IncrementalByteDigest.compute(for: jpegData)
                                        let sLine = String(
                                            format: "%.6f,%d,%@,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.6f,%.1f\n",
                                            timestampS, sIdx, stillRelPath,
                                            camPosW.x, camPosW.y, camPosW.z,
                                            normQuat.imag.x, normQuat.imag.y, normQuat.imag.z, normQuat.real,
                                            fx, fy, cx, cy, rgbW, rgbH,
                                            exposureDurationS, isoVal
                                        )
                                        encodedStill = (stillRelPath, jpegData, meta, sLine)
                                    } else {
                                        // Roll back the hold still counter if OpenCV rejected the marker on the encoded JPEG (`YS-68`, `YS-70`)
                                        engine.restoreStillGate(gateDecision.snapshotBefore)
                                    }
                                } else {
                                    engine.restoreStillGate(gateDecision.snapshotBefore)
                                }
                                sp.endInterval("StillJPEGEncode", stillState)
                            }

                            var markerLines: [String] = []
                            for m in confirmedSightings {
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
                                markerLines.append(mLine)
                            }

                            let finalDetected = confirmedSightings
                            let finalRawDetected = rawDetected3D
                            let finalEncodedStill = encodedStill
                            let finalMarkerLines = markerLines
                            if let m0 = finalDetected.first(where: { $0.markerID == 0 }) {
                                let mq = simd_quatd(ix: m0.quaternionXYZW.x, iy: m0.quaternionXYZW.y, iz: m0.quaternionXYZW.z, r: m0.quaternionXYZW.w)
                                let m0XAxis = simd_act(mq, SIMD3<Double>(1.0, 0.0, 0.0))
                                engine.anchorMarker0Frame(originWorld: m0.positionWorld, xAxisWorld: m0XAxis)
                            }

                            if finalEncodedStill != nil || !finalMarkerLines.isEmpty {
                                self.writerQueue.async { [weak self] in
                                    guard let self, !engine.isClosed else { return }
                                    do {
                                        if let still = finalEncodedStill, !engine.savedStillsIndices.contains(sIdx) {
                                            let stillURL = dir.appendingPathComponent(still.relPath)
                                            try still.data.write(to: stillURL)
                                            engine.digestRegistry.record(still.meta, forRelativePath: still.relPath)
                                            engine.savedStillsIndices.insert(sIdx)
                                            try engine.appendLine(still.csvLine, to: engine.stillsFH)
                                        }
                                        for mLine in finalMarkerLines {
                                            try engine.appendLine(mLine, to: engine.markersFH)
                                        }
                                    } catch {
                                        let errMsg = error.localizedDescription
                                        Task { @MainActor in
                                            self.handleStreamFailure(stream: "stills", message: errMsg)
                                        }
                                    }
                                }
                            }

                            if shouldRunArUco || !finalDetected.isEmpty || !finalRawDetected.isEmpty || covResult != nil {
                                Task { @MainActor [weak self] in
                                    guard let self, self.phase == .recording else { return }
                                    if let cov = covResult {
                                        self.coveredGroundCellCount = cov.snapshot.coveredCellCount
                                        self.coveredGroundAreaM2 = cov.snapshot.coveredAreaM2
                                        self.coveredGroundCells = cov.snapshot.coveredCells
                                        self.latestElevatedPointsAbove2m += cov.elevatedPointsAbove2m
                                    }
                                    if !finalDetected.isEmpty {
                                        _ = self.phoneLoopState.ingestSightings(
                                            finalDetected,
                                            timestampS: timestampS,
                                            savedIndex: sIdx,
                                            cameraPositionWorld: camPosW
                                        )
                                        self.detectedMarkerIDs = self.phoneLoopState.detectedMarkerIDs
                                        self.isLoopClosed = self.phoneLoopState.unifiedLoopClosed
                                        self.loopClosureGapMeters = self.phoneLoopState.unifiedLoopClosureGapMeters
                                        self.updateProvisionalGrade()

                                        // YS-73: After 3 sightings of a marker, check if median measured size is > 3% off
                                        for m in finalDetected {
                                            let markerMeas = self.phoneLoopState.allSightings
                                                .filter { $0.sighting.markerID == m.markerID }
                                                .map(\.sighting.sizeMMMeasured)
                                            let sizeCheck = ExposureAndStillnessGate.evaluateLiveMarkerSizeDeviation(
                                                measuredSizesMM: markerMeas,
                                                printedSizeMM: m.sizeMMPrinted
                                            )
                                            if sizeCheck.flagged, let medMM = sizeCheck.medianMM {
                                                self.lastSizeMismatchMarkerID = m.markerID
                                                self.liveMarkerSizeWarning = String(
                                                    format: "Marker %d size %.1f mm differs from %.0f mm setting (> 3%%)",
                                                    m.markerID, medMM, m.sizeMMPrinted
                                                )
                                            }
                                        }
                                    }

                                    let wasClosingLocked = self.guidedEngine.isClosingMarker0Locked
                                    let monotonicT = max(timestampS, self.latestTimestampS)
                                    let detStep = self.guidedEngine.step(
                                        timestampS: monotonicT,
                                        savedIndex: sIdx,
                                        dt: 0.0,
                                        trackingState: trackingStr,
                                        trackingReason: reasonStr,
                                        pitchDeg: pitchDegrees,
                                        turnRateRadS: turnRateRadS,
                                        predictedBlurPx: predictedBlurPx,
                                        cameraPositionWorld: camPosW,
                                        confirmedSightings: finalDetected,
                                        rawSightings: finalRawDetected,
                                        unifiedLoopClosed: self.phoneLoopState.unifiedLoopClosed,
                                        thermalLevel: activeThermalLevel,
                                        lowDiskSpaceWarning: self.lowDiskSpaceWarning != nil,
                                        hasValidDepthForLookUp: self.latestValidDepthSamplesCount >= 8,
                                        elevatedPointsAbove2m: 0,
                                        didRunMarkerDetection: shouldRunArUco
                                    )

                                    // Prevent 5-still hold cap deadlock (`YS-68`, `YS-74`): if a `marker_hold` still was saved
                                    // but `GuidedCaptureEngine` did not consume it into a segment or lock, roll back the hold counter
                                    if gateDecision.shouldSave, gateDecision.reason == "marker_hold" {
                                        let didConsumeSegmentOrLock: Bool = {
                                            switch detStep.haptic {
                                            case .holdSegmentConfirmed, .markerLocked, .loopClosedPathComplete:
                                                return true
                                            default:
                                                return false
                                            }
                                        }()
                                        if !didConsumeSegmentOrLock && !forceArUco {
                                            engine.restoreStillGate(gateDecision.snapshotBefore)
                                        }
                                    }

                                    self.guidedWalkPhase = self.guidedEngine.phase
                                    self.activeRingState = self.guidedEngine.activeRingState
                                    self.lockedMarkerIDs = self.guidedEngine.lockedMarkerIDs
                                    self.lookUpProgressFraction = self.guidedEngine.lookUpTracker.progressFraction
                                    self.isLookUpDone = self.guidedEngine.lookUpTracker.isDone
                                    self.thermalSchedule = detStep.thermalSchedule
                                    if self.experimentFlags.arAnchorOnLocks {
                                        switch detStep.haptic {
                                        case .markerLocked(let mid):
                                            if let s = finalDetected.first(where: { $0.markerID == mid }) {
                                                let anchorMat = ARAnchorLockExperiment.makeAnchorTransform(
                                                    positionWorld: s.positionWorld,
                                                    quaternionXYZW: s.quaternionXYZW
                                                )
                                                #if canImport(ARKit)
                                                let anchor = ARAnchor(
                                                    name: ARAnchorLockExperiment.anchorName(forMarkerID: mid, isClosingHold: false),
                                                    transform: anchorMat
                                                )
                                                self.arSession.add(anchor: anchor)
                                                #endif
                                            }
                                        case .loopClosedPathComplete:
                                            if let s = finalDetected.first(where: { $0.markerID == 0 }) {
                                                let anchorMat = ARAnchorLockExperiment.makeAnchorTransform(
                                                    positionWorld: s.positionWorld,
                                                    quaternionXYZW: s.quaternionXYZW
                                                )
                                                #if canImport(ARKit)
                                                let anchor = ARAnchor(
                                                    name: ARAnchorLockExperiment.anchorName(forMarkerID: 0, isClosingHold: true),
                                                    transform: anchorMat
                                                )
                                                self.arSession.add(anchor: anchor)
                                                #endif
                                            }
                                        default:
                                            break
                                        }
                                    }
                                    self.dispatchHapticIfNeeded(detStep.haptic)
                                    let didSpeakDet = self.dispatchVoiceCoachIfNeeded(self.guidedEngine.lastSpokenPhraseThisStep)
                                    self.syncMarker0To1PreWalkTapeCheckIfNeeded()
                                    self.statusBanner = detStep.instruction

                                    // YS-67 / YS-71 / YS-74: Only a completed 3-still closing hold on Marker 0 writes `loop_closed`
                                    // so all 3 confirmed stills of the closing hold precede `loop_closed_t_s`!
                                    if !wasClosingLocked && self.guidedEngine.isClosingMarker0Locked {
                                        self.phoneLoopState.recordLoopClosedEvent(at: monotonicT)
                                        self.isLoopClosed = self.phoneLoopState.unifiedLoopClosed
                                        self.loopClosureGapMeters = self.phoneLoopState.unifiedLoopClosureGapMeters
                                        if self.phoneLoopState.unifiedLoopClosed, let gap = self.phoneLoopState.unifiedLoopClosureGapMeters {
                                            self.logEvent(kind: "loop_closed", detail: String(format: "gap_m=%.4f", gap))
                                        }
                                        self.updateProvisionalGrade()
                                    }
                                    self.isFinishAvailable = self.guidedEngine.isFinishAvailable(unifiedLoopClosed: self.phoneLoopState.unifiedLoopClosed)

                                    // Log explicit `marker_locked` / `loop_closed` / spoken guidance event rows to `guidance.csv` (`YS-67`, `YS-74`, `YS-75`)
                                    let guidanceEventName: String? = {
                                        switch detStep.haptic {
                                        case .markerLocked: return "marker_locked"
                                        case .loopClosedPathComplete: return "loop_closed"
                                        default: return didSpeakDet ? "voice_coach" : nil
                                        }
                                    }()
                                    if let evName = guidanceEventName {
                                        let phaseNow = self.guidedEngine.phase.rawValue
                                        let instrNow = detStep.instruction
                                        let covCellsNow = self.coveredGroundCellCount
                                        self.writerQueue.async { [weak self] in
                                            guard let self, !engine.isClosed else { return }
                                            do {
                                                try engine.appendMonotonicGuidanceRow(
                                                    timestampS: monotonicT,
                                                    savedIndex: sIdx,
                                                    event: evName,
                                                    phase: phaseNow,
                                                    instruction: instrNow,
                                                    spoken: didSpeakDet,
                                                    pitchDeg: pitchDegrees,
                                                    rollDeg: rollDegrees,
                                                    trackingState: trackingStr,
                                                    exposureDurationS: exposureDurationS,
                                                    iso: isoVal,
                                                    turnRateRadS: turnRateRadS,
                                                    predictedBlurPx: predictedBlurPx,
                                                    thermalState: effectiveThermalStr,
                                                    coverageCells: covCellsNow
                                                )
                                            } catch {
                                                Task { @MainActor in
                                                    self.handleStreamFailure(stream: "guidance", message: error.localizedDescription)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } catch {
                    let errMsg = error.localizedDescription
                    Task { @MainActor in
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
            _ = self.evaluatePreFlightChecklist()
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
