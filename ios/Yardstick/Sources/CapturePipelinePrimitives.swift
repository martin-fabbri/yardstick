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

import CryptoKit
import Foundation
import os

/// Precomputed SHA-256, IEEE 802.3 CRC-32, and byte size for a capture file (`YS-84`).
public struct FileDigestMeta: Sendable, Equatable {
    public let sha256: String
    public let crc32: UInt32
    public let sizeBytes: UInt64

    public init(sha256: String, crc32: UInt32, sizeBytes: UInt64) {
        self.sha256 = sha256
        self.crc32 = crc32
        self.sizeBytes = sizeBytes
    }
}

/// Incremental SHA-256 + IEEE 802.3 CRC-32 + byte-size accumulator so files are hashed as they are
/// written rather than re-read at the end of a walk (`YS-84`).
public struct IncrementalByteDigest: Sendable {
    public static let crc32Table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 {
            c = (c & 1 != 0) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1)
        }
        return c
    }

    private var hasher = SHA256()
    private var crc: UInt32 = 0xFFFFFFFF
    public private(set) var sizeBytes: UInt64 = 0

    public init() {}

    public mutating func update(with data: Data) {
        hasher.update(data: data)
        sizeBytes &+= UInt64(data.count)
        let table = Self.crc32Table
        var localCRC = crc
        data.withUnsafeBytes { rawBuf in
            for byte in rawBuf {
                let idx = Int((localCRC ^ UInt32(byte)) & 0xFF)
                localCRC = (localCRC >> 8) ^ table[idx]
            }
        }
        crc = localCRC
    }

    public func finalize() -> FileDigestMeta {
        let digest = hasher.finalize()
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return FileDigestMeta(sha256: hex, crc32: ~crc, sizeBytes: sizeBytes)
    }

    /// Compute SHA-256, CRC-32, and byte count for an in-memory `Data` buffer in a single pass (`YS-84`).
    public static func compute(for data: Data) -> FileDigestMeta {
        var acc = IncrementalByteDigest()
        acc.update(with: data)
        return acc.finalize()
    }

    /// Stream-compute SHA-256, IEEE 802.3 CRC-32, and byte size in 64 KB chunks (`YS-84`).
    public static func streamFile(at fileURL: URL) throws -> FileDigestMeta {
        let fh = try FileHandle(forReadingFrom: fileURL)
        defer { try? fh.close() }
        var acc = IncrementalByteDigest()
        while true {
            var shouldBreak = false
            var readErr: Error?
            autoreleasepool {
                do {
                    guard let chunk = try fh.read(upToCount: 65536), !chunk.isEmpty else {
                        shouldBreak = true
                        return
                    }
                    acc.update(with: chunk)
                } catch {
                    readErr = error
                    shouldBreak = true
                }
            }
            if let readErr { throw readErr }
            if shouldBreak { break }
        }
        return acc.finalize()
    }

    /// Compute the relative path of `fileURL` inside `baseDir`, resolving symlinks on both URLs (`YS-65`, `YS-84`).
    public static func relativePath(of fileURL: URL, inside baseDir: URL) -> String? {
        let baseResolved = baseDir.resolvingSymlinksInPath().standardizedFileURL.path
        let fileResolved = fileURL.resolvingSymlinksInPath().standardizedFileURL.path
        let prefix = baseResolved.hasSuffix("/") ? baseResolved : (baseResolved + "/")
        guard fileResolved.hasPrefix(prefix) else { return nil }
        let rel = String(fileResolved.dropFirst(prefix.count))
        return rel.isEmpty ? nil : rel
    }

    /// Enumerate capture files in `dir`, reusing `precomputedDigests` without re-reading pre-hashed files from disk (`YS-84`).
    public static func collectManifestFileDigests(
        in dir: URL,
        precomputedDigests: [String: FileDigestMeta] = [:]
    ) throws -> (filesArray: [[String: Any]], crcAndSizeByRelPath: [String: (crc32: UInt32, sizeBytes: UInt64)]) {
        let resolvedDir = dir.resolvingSymlinksInPath().standardizedFileURL
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: resolvedDir, includingPropertiesForKeys: [.isRegularFileKey])
        var filesArray: [[String: Any]] = []
        var crcByRel: [String: (crc32: UInt32, sizeBytes: UInt64)] = [:]

        while let fileURL = enumerator?.nextObject() as? URL {
            guard let rel = relativePath(of: fileURL, inside: resolvedDir) else { continue }
            if rel == "manifest.json" || rel.hasPrefix("standard/") || rel.hasPrefix(".") {
                continue
            }
            let vals = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard vals.isRegularFile == true else { continue }

            let meta: FileDigestMeta
            if let pre = precomputedDigests[rel] {
                meta = pre
            } else {
                meta = try streamFile(at: fileURL)
            }
            filesArray.append([
                "path": rel,
                "size_bytes": meta.sizeBytes,
                "sha256": meta.sha256,
            ])
            crcByRel[rel] = (meta.crc32, meta.sizeBytes)
        }
        filesArray.sort { ($0["path"] as? String ?? "") < ($1["path"] as? String ?? "") }
        return (filesArray, crcByRel)
    }
}

/// Thread-safe registry of file digests computed during capture writing (`YS-84`).
public final class PrecomputedDigestRegistry: Sendable {
    private struct State: Sendable {
        var byRelPath: [String: FileDigestMeta] = [:]
        var totalRecordedBytes: UInt64 = 0
    }

    private let lock = OSAllocatedUnfairLock(initialState: State())

    public init() {}

    public func record(_ meta: FileDigestMeta, forRelativePath relPath: String) {
        lock.withLock { state in
            if let old = state.byRelPath[relPath] {
                state.totalRecordedBytes = state.totalRecordedBytes >= old.sizeBytes
                    ? (state.totalRecordedBytes - old.sizeBytes)
                    : 0
            }
            state.byRelPath[relPath] = meta
            state.totalRecordedBytes &+= meta.sizeBytes
        }
    }

    public func addRecordedBytes(_ deltaBytes: UInt64) {
        lock.withLock { state in
            state.totalRecordedBytes &+= deltaBytes
        }
    }

    public var totalRecordedBytes: UInt64 {
        lock.withLock { $0.totalRecordedBytes }
    }

    public func snapshot() -> [String: FileDigestMeta] {
        lock.withLock { $0.byRelPath }
    }

    public func reset() {
        lock.withLock { state in
            state.byRelPath.removeAll()
            state.totalRecordedBytes = 0
        }
    }
}

/// Atomic writer-side backlog counter that is never updated through `@MainActor` (`YS-68`).
///
/// Prevents UI main-thread stalls from delaying job-completion decrements and causing false
/// `writer_queue_full` frame drops.
public final class WriterBacklogCounter: Sendable {
    private struct State: Sendable {
        var pendingJobs: Int = 0
        var peakJobs: Int = 0
        var overloadCount: Int = 0
    }

    public let maxBacklog: Int
    private let lock = OSAllocatedUnfairLock(initialState: State())

    public init(maxBacklog: Int = 8) {
        self.maxBacklog = maxBacklog
    }

    /// Increment the writer backlog before dispatching a frame to `writerQueue`.
    /// Returns `(overloaded: true, backlog)` if the backlog prior to this frame already exceeded `maxBacklog`.
    @discardableResult
    public func acquireSlot() -> (overloaded: Bool, currentBacklog: Int) {
        lock.withLock { state in
            let wasOverloaded = state.pendingJobs > maxBacklog
            state.pendingJobs += 1
            if state.pendingJobs > state.peakJobs {
                state.peakJobs = state.pendingJobs
            }
            if wasOverloaded {
                state.overloadCount += 1
            }
            return (wasOverloaded, state.pendingJobs)
        }
    }

    /// Decrement the writer backlog synchronously on `writerQueue` when a writer job finishes (`YS-68`).
    public func releaseSlot() {
        lock.withLock { state in
            state.pendingJobs = max(0, state.pendingJobs - 1)
        }
    }

    public var currentBacklog: Int {
        lock.withLock { $0.pendingJobs }
    }

    public var peakBacklog: Int {
        lock.withLock { $0.peakJobs }
    }

    public var overloadDropCount: Int {
        lock.withLock { $0.overloadCount }
    }

    public func reset() {
        lock.withLock { state in
            state = State()
        }
    }
}

/// Single-slot background queue for ArUco marker detection and live ground coverage (`YS-68`).
///
/// If a detection job is already running when a new frame arrives, incoming work is dropped immediately
/// rather than queuing up behind stale frames or blocking `writerQueue`.
public final class SingleSlotDetectionScheduler: Sendable {
    public struct Stats: Sendable, Equatable {
        public var isBusy: Bool = false
        public var submittedCount: Int = 0
        public var droppedBusyCount: Int = 0
        public var completedCount: Int = 0
    }

    private let queue: DispatchQueue
    private let lock = OSAllocatedUnfairLock(initialState: Stats())

    public init(label: String = "io.yardstick.capture.detection", qos: DispatchQoS = .userInitiated) {
        self.queue = DispatchQueue(label: label, qos: qos)
    }

    /// Submit `work` to the detection queue if idle; drops immediately and returns `false` if busy.
    @discardableResult
    public func submitIfIdle(_ work: @escaping @Sendable () -> Void) -> Bool {
        let accepted = lock.withLock { state -> Bool in
            if state.isBusy {
                state.droppedBusyCount += 1
                return false
            }
            state.isBusy = true
            state.submittedCount += 1
            return true
        }
        guard accepted else { return false }

        queue.async { [lock] in
            defer {
                lock.withLock { state in
                    state.isBusy = false
                    state.completedCount += 1
                }
            }
            work()
        }
        return true
    }

    /// Wait synchronously for any in-flight detection job to finish before closing capture streams.
    public func drainSynchronously() {
        queue.sync {}
    }

    /// Schedule a completion callback on the serial detection queue so it runs strictly after any
    /// in-flight detection job has completed and dispatched its output (`YS-68`).
    public func scheduleAfterPendingWork(_ block: @escaping @Sendable () -> Void) {
        queue.async(execute: block)
    }

    public var stats: Stats {
        lock.withLock { $0 }
    }

    public func resetStats() {
        drainSynchronously()
        lock.withLock { state in
            state = Stats()
        }
    }
}

/// Pure, deterministic rate-limiter for marker-triggered and periodic still JPEGs (`YS-68`).
///
/// Limits marker-triggered stills to at most 1 per `0.35 s` and at most `5` stills per contiguous
/// marker hold (resetting only when a marker has been out of view for `> 1.50 s`).
public struct StillCaptureGate: Sendable {
    public struct HoldRecord: Sendable, Equatable {
        public var lastSeenTimeS: Double
        public var stillsSavedInHold: Int
    }

    public let minMarkerStillIntervalS: Double
    public let maxStillsPerMarkerHold: Int
    public let holdResetGapS: Double
    public let periodicStillMinIntervalS: Double
    public let periodicStillMaxIntervalS: Double
    public let minPeriodicSharpness: Double

    public private(set) var lastMarkerStillTimeS: Double = -100.0
    public private(set) var lastAnyStillTimeS: Double = -100.0
    public private(set) var holdByMarkerID: [Int: HoldRecord] = [:]
    public private(set) var markerStillsSavedCount: Int = 0
    public private(set) var periodicStillsSavedCount: Int = 0

    public init(
        minMarkerStillIntervalS: Double = 0.35,
        maxStillsPerMarkerHold: Int = 5,
        holdResetGapS: Double = 1.50,
        periodicStillMinIntervalS: Double = 2.00,
        periodicStillMaxIntervalS: Double = 3.00,
        minPeriodicSharpness: Double = 12.0
    ) {
        self.minMarkerStillIntervalS = minMarkerStillIntervalS
        self.maxStillsPerMarkerHold = maxStillsPerMarkerHold
        self.holdResetGapS = holdResetGapS
        self.periodicStillMinIntervalS = periodicStillMinIntervalS
        self.periodicStillMaxIntervalS = periodicStillMaxIntervalS
        self.minPeriodicSharpness = minPeriodicSharpness
    }

    public var totalStillsSavedCount: Int {
        markerStillsSavedCount + periodicStillsSavedCount
    }

    /// Evaluate whether the frame at `timestampS` should save a JPEG still (`YS-68`).
    public mutating func evaluateFrame(
        timestampS: Double,
        detectedMarkerIDs: [Int],
        forceOperatorMarkerTap: Bool = false,
        sharpness: Double? = nil
    ) -> (shouldSave: Bool, reason: String?) {
        // Expire holds for markers that have not been seen for > holdResetGapS
        let expiredIDs = holdByMarkerID.compactMap { mid, rec -> Int? in
            (timestampS - rec.lastSeenTimeS) > holdResetGapS ? mid : nil
        }
        for mid in expiredIDs {
            holdByMarkerID.removeValue(forKey: mid)
        }

        let uniqueIDs = Array(Set(detectedMarkerIDs)).sorted()
        if !uniqueIDs.isEmpty {
            var eligibleIDs: [Int] = []
            for mid in uniqueIDs {
                var rec = holdByMarkerID[mid] ?? HoldRecord(lastSeenTimeS: timestampS, stillsSavedInHold: 0)
                rec.lastSeenTimeS = timestampS
                holdByMarkerID[mid] = rec
                if rec.stillsSavedInHold < maxStillsPerMarkerHold {
                    eligibleIDs.append(mid)
                }
            }

            let intervalOK = (timestampS - lastMarkerStillTimeS) >= (minMarkerStillIntervalS - 1e-6)
            if (!eligibleIDs.isEmpty && intervalOK) || (forceOperatorMarkerTap && intervalOK) {
                for mid in uniqueIDs {
                    if var rec = holdByMarkerID[mid] {
                        rec.stillsSavedInHold = min(maxStillsPerMarkerHold, rec.stillsSavedInHold + 1)
                        holdByMarkerID[mid] = rec
                    }
                }
                lastMarkerStillTimeS = timestampS
                lastAnyStillTimeS = timestampS
                markerStillsSavedCount += 1
                return (true, "marker_hold")
            }
            // While holding on a marker that already reached its 5-still cap (or within 0.35 s),
            // do not emit redundant periodic stills of the same stationary marker.
            return (false, nil)
        }

        let dtAny = timestampS - lastAnyStillTimeS
        if dtAny >= (periodicStillMaxIntervalS - 1e-6) {
            lastAnyStillTimeS = timestampS
            periodicStillsSavedCount += 1
            return (true, "periodic_max_interval")
        }
        if dtAny >= (periodicStillMinIntervalS - 1e-6),
           let s = sharpness,
           s >= minPeriodicSharpness {
            lastAnyStillTimeS = timestampS
            periodicStillsSavedCount += 1
            return (true, "periodic_sharp")
        }

        return (false, nil)
    }
}

/// Mid-walk free disk space monitor that warns before sealing can fail with `ENOSPC` (`YS-84`).
public enum DiskSpaceMonitor {
    /// Minimum free bytes floor (`2.5 GB`) during a walk before warning (`YS-84`).
    public static let minimumFreeBytesWarningFloor: Int64 = 2_500_000_000
    /// Additional headroom (`1.5 GB`) above the current raw capture size needed to pack the store-mode `.zip`.
    public static let zipHeadroomSafetyMarginBytes: Int64 = 1_500_000_000

    public static func requiredFreeBytesForSafeSeal(recordedCaptureBytes: Int64) -> Int64 {
        return max(minimumFreeBytesWarningFloor, recordedCaptureBytes + zipHeadroomSafetyMarginBytes)
    }

    public static func evaluateMidWalkStorage(
        availableBytes: Int64,
        recordedCaptureBytes: Int64
    ) -> (isLowSpace: Bool, warningMessage: String?) {
        let threshold = requiredFreeBytesForSafeSeal(recordedCaptureBytes: recordedCaptureBytes)
        if availableBytes < threshold {
            return (true, "Almost out of space. Finish soon.")
        }
        return (false, nil)
    }

    public static func querySystemAvailableBytes(at url: URL) -> Int64? {
        if let vals = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let cap = vals.volumeAvailableCapacityForImportantUsage {
            return Int64(cap)
        }
        if let vals = try? url.resourceValues(forKeys: [.volumeAvailableCapacityKey]),
           let cap = vals.volumeAvailableCapacity {
            return Int64(cap)
        }
        return nil
    }
}
