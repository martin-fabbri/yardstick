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

import SwiftUI
import UIKit

#if canImport(ARKit) && canImport(RealityKit)
import ARKit
import RealityKit

/// Live RealityKit `ARView` camera feed sharing `recorder.arSession` (`YS-40`, `YS-83`).
///
/// Replaces the deprecated SceneKit `ARSCNView` with `ARView(frame:cameraMode:automaticallyConfigureSession: false)`
/// so `CaptureSessionRecorder` remains the sole owner of the `ARSession` configuration and delegate.
private struct ARCameraPreviewView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(
            frame: .zero,
            cameraMode: .ar,
            automaticallyConfigureSession: false
        )
        view.session = session
        view.renderOptions.insert([
            .disablePersonOcclusion,
            .disableDepthOfField,
            .disableMotionBlur,
            .disableCameraGrain,
            .disableHDR,
        ])
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        if uiView.session !== session {
            uiView.session = session
        }
    }
}
#endif

/// Main SwiftUI operator interface for Yardstick (`YS-00`, `YS-05`, `YS-07`, `YS-12`, `YS-22`, `YS-40`, `YS-44`, `YS-83`, `YS-84`).
///
/// Screens:
/// 1. **Pre-walk Setup**: verifies LiDAR support (`YS-00`), configures printed marker sizes (`250 mm` Letter/A4 or `180 mm` clipboard) for Markers `0–5` (`YS-33`).
/// 2. **Live Recording HUD**: camera preview + center crosshair for 5×5 LiDAR patch tape checks (`YS-40`), live pitch/roll guide (`YS-05`, `YS-12`), marker & North calibration badges (`YS-14`), low-space warning (`YS-84`), determinate `"Saving your walk"` progress overlay (`YS-84`), and 5 action buttons (`Marker`, `Tape A`, `Tape B`, `Note`, `Done`, `YS-07`).
/// 3. **Post-walk Summary**: displays loop-closure gap (`YS-32`), provisional grade & reasons (`YS-42`, `YS-44`), and one-tap AirDrop / Files export (`YS-22`).
public struct ContentView: View {
    @StateObject private var recorder = CaptureSessionRecorder()
    @State private var showTapeBKeypad = false
    @State private var tapeMetersInput = "5.00"
    @State private var tapeRoleInput = "primary"
    @State private var showNoteDialog = false
    @State private var noteTextInput = ""
    @State private var showShareSheet = false
    @State private var showAboutSheet = false
    @State private var didTriggerAutoBench = false

    public init() {}

    public var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch recorder.phase {
            case .setup:
                setupView
            case .recording, .sealing:
                recordingHUDView
            case .completed, .failed:
                summaryView
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            guard !didTriggerAutoBench else { return }
            let args = ProcessInfo.processInfo.arguments
            if let idx = args.firstIndex(of: "--bench-seal-dir"),
               idx + 1 < args.count {
                didTriggerAutoBench = true
                let dirPath = args[idx + 1]
                let dirURL = URL(fileURLWithPath: dirPath, isDirectory: true)
                _ = try? CaptureSessionRecorder.sealManifestAndPackZip(in: dirURL)
            } else if let idx = args.firstIndex(of: "--auto-bench-capture"),
               idx + 1 < args.count,
               let seconds = Double(args[idx + 1]),
               seconds > 0,
               recorder.phase == .setup {
                didTriggerAutoBench = true
                recorder.requestPermissionsAndStart()
                DispatchQueue.main.asyncAfter(deadline: .now() + seconds) {
                    if recorder.phase == .recording {
                        recorder.stopAndSealCapture()
                    }
                }
            }
        }
    }

    private var setupView: some View {
        VStack(spacing: 20) {
            Text("Yardstick")
                .font(.largeTitle.bold())
                .foregroundStyle(.white)

            Text("Calibrated iPhone LiDAR Yard Capture (yardstick-capture/1)")
                .font(.subheadline)
                .foregroundStyle(.gray)
                .multilineTextAlignment(.center)

            if !CaptureSessionRecorder.isLiDARSupported {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text("LiDAR sensor required on physical iPhone Pro / Pro Max (YS-00)")
                        .font(.footnote)
                        .foregroundStyle(.yellow)
                }
                .padding(10)
                .background(Color.white.opacity(0.08))
                .clipShape(.rect(cornerRadius: 8))
            }

            VStack(alignment: .leading, spacing: 12) {
                Text("Printed Marker Edge Size (tap to toggle 250 mm / 180 mm)")
                    .font(.headline)
                    .foregroundStyle(.white)

                ForEach(0..<6, id: \.self) { mid in
                    let sizeMM = Int(recorder.markerSizesMM[mid] ?? 250.0)
                    Button {
                        recorder.toggleMarkerSize(for: mid)
                    } label: {
                        HStack {
                            Text(mid == 0 ? "Marker 0 (Origin & Loop Closure)" : "Marker \(mid)")
                                .font(.body.weight(.medium))
                            Spacer()
                            Text("\(sizeMM) mm")
                                .font(.body.monospacedDigit().bold())
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(sizeMM == 250 ? Color.blue.opacity(0.35) : Color.orange.opacity(0.35))
                                .clipShape(.rect(cornerRadius: 6))
                        }
                        .foregroundStyle(.white)
                        .padding(.vertical, 4)
                    }
                }
            }
            .padding()
            .background(Color.white.opacity(0.06))
            .clipShape(.rect(cornerRadius: 12))

            Button {
                recorder.requestPermissionsAndStart()
            } label: {
                Text(CaptureSessionRecorder.isLiDARSupported ? "Start Yard Walk" : "LiDAR Required (YS-00)")
                    .font(.headline)
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(CaptureSessionRecorder.isLiDARSupported ? Color.green : Color.gray)
                    .clipShape(.rect(cornerRadius: 12))
            }
            .disabled(!CaptureSessionRecorder.isLiDARSupported)

            Button {
                showAboutSheet = true
            } label: {
                Label("About & Open-Source Licences", systemImage: "info.circle")
                    .font(.footnote)
                    .foregroundStyle(.gray)
            }
        }
        .padding(24)
        .sheet(isPresented: $showAboutSheet) {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Yardstick")
                            .font(.title2.bold())
                        Text("Contract: yardstick-capture/1 · Apache License 2.0")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Divider()
                        Text("Open-Source Notices")
                            .font(.headline)
                        Text(ArUcoDetector.openCVApacheAttributionNotice)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                }
                .navigationTitle("About")
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { showAboutSheet = false }
                    }
                }
            }
        }
    }

    private var recordingHUDView: some View {
        ZStack {
            // Live ARKit Camera Preview via RealityKit ARView (YS-40, YS-83)
            #if canImport(ARKit) && canImport(RealityKit)
            ARCameraPreviewView(session: recorder.arSession)
                .ignoresSafeArea()
            #endif

            // 3-Segment Marker Lock Ring drawn on the marker in the camera view (YS-74)
            if let ringState = recorder.activeRingState {
                GeometryReader { proxy in
                    let normCenter = MarkerRingOverlayView.normalizedCenter(
                        for: ringState,
                        rgbWidth: Double(recorder.actualWidth),
                        rgbHeight: Double(recorder.actualHeight)
                    )
                    MarkerRingOverlayView(
                        ringState: ringState,
                        animationsStatic: recorder.thermalSchedule.animationsStatic
                    )
                    .position(
                        x: normCenter.x * proxy.size.width,
                        y: normCenter.y * proxy.size.height
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }

            // Center Crosshair for 5x5 LiDAR depth patch tape checks (YS-40)
            VStack {
                Spacer()
                ZStack {
                    Circle()
                        .stroke(Color.green.opacity(0.85), lineWidth: 2)
                        .frame(width: 36, height: 36)
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 14, height: 2)
                    Rectangle()
                        .fill(Color.green)
                        .frame(width: 2, height: 14)
                }
                if let d = recorder.crosshairDepthMeters {
                    Text(String(format: "%.2f m", d))
                        .font(.caption.monospacedDigit().bold())
                        .padding(4)
                        .background(Color.black.opacity(0.6))
                        .clipShape(.rect(cornerRadius: 4))
                }
                Spacer()
            }

            VStack {
                // Top telemetry & guidance bar (YS-05, YS-12, YS-14, YS-84)
                VStack(alignment: .leading, spacing: 6) {
                    Text(recorder.statusBanner)
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)

                    if let warning = recorder.lowDiskSpaceWarning {
                        HStack(spacing: 6) {
                            Image(systemName: "externaldrive.badge.exclamationmark")
                            Text(warning)
                                .font(.caption.bold())
                        }
                        .foregroundStyle(.black)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.yellow)
                        .clipShape(.rect(cornerRadius: 6))
                    }

                    if let sizeWarn = recorder.liveMarkerSizeWarning {
                        HStack(spacing: 6) {
                            Image(systemName: "ruler")
                            Text(sizeWarn)
                                .font(.caption.bold())
                        }
                        .foregroundStyle(.black)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.orange)
                        .clipShape(.rect(cornerRadius: 6))
                    }

                    HStack(spacing: 12) {
                        Label("Frames: \(recorder.savedFrameCount)", systemImage: "video.fill")
                        Label("Markers: \(recorder.lockedMarkerIDs.count)/4", systemImage: "qrcode.viewfinder")
                        Label(String(format: "Ground: %.1f m²", recorder.coveredGroundAreaM2), systemImage: "square.grid.3x3.fill")
                        Label("Tapes: \(recorder.tapeCheckCount)", systemImage: "ruler.fill")
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.green)

                    if recorder.guidedWalkPhase == .houseWallLookUp {
                        HStack(spacing: 12) {
                            ProgressView(value: recorder.lookUpProgressFraction, total: 1.0)
                                .tint(.green)
                            Button("Skip") {
                                recorder.skipLookUpStep()
                            }
                            .font(.caption.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.white.opacity(0.2))
                            .clipShape(.rect(cornerRadius: 6))
                        }
                    }

                    HStack(spacing: 16) {
                        Text(String(format: "Pitch: %.1f° (target -30°)", recorder.pitchDeg))
                        Text(String(format: "Roll: %.1f°", recorder.rollDeg))
                        Text("Track: \(recorder.trackingStateText)")
                    }
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.gray)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.75))

                if !recorder.coveredGroundCells.isEmpty {
                    HStack {
                        Spacer()
                        GroundCoverageMiniMapView(cells: recorder.coveredGroundCells)
                            .frame(width: 96, height: 96)
                            .padding(8)
                    }
                }

                Spacer()

                if showTapeBKeypad {
                    tapeKeypadOverlay
                }

                // Bottom 5-button operator bar: Marker, Tape A, Tape B, Note, Done (YS-07)
                HStack(spacing: 10) {
                    hudButton(title: "Marker", icon: "qrcode", color: .blue) {
                        recorder.logEvent(kind: "marker", detail: "Operator marker sighting")
                    }
                    hudButton(title: "Tape A", icon: "a.circle.fill", color: .orange) {
                        recorder.captureTapeEndpointA()
                    }
                    hudButton(title: "Tape B", icon: "b.circle.fill", color: .orange) {
                        // Lock Endpoint B 3D point immediately at tap time BEFORE opening keypad (YS-40)
                        if recorder.captureTapeEndpointB() {
                            showTapeBKeypad = true
                        }
                    }
                    hudButton(
                        title: recorder.isRecordingVoiceNote ? "Note On" : "Note",
                        icon: recorder.isRecordingVoiceNote ? "mic.badge.xmark" : "mic.fill",
                        color: .purple
                    ) {
                        let nextState = !recorder.isRecordingVoiceNote
                        recorder.setRecordingVoiceNote(nextState)
                        if nextState {
                            recorder.logEvent(kind: "note", detail: "Spoken note anchor")
                        }
                    }
                    hudButton(
                        title: recorder.isFinishAvailable ? "Finish" : "Done",
                        icon: "checkmark.seal.fill",
                        color: recorder.isFinishAvailable ? .green : .gray
                    ) {
                        recorder.stopAndSealCapture()
                    }
                }
                .padding(12)
                .background(Color.black.opacity(0.85))
                .disabled(recorder.phase == .sealing)
            }

            if recorder.phase == .sealing {
                sealingProgressOverlay
            }
        }
    }

    private var sealingProgressOverlay: some View {
        VStack(spacing: 16) {
            Text(recorder.sealingStatusText)
                .font(.title3.bold())
                .foregroundStyle(.white)

            ProgressView(value: recorder.sealingProgress, total: 1.0)
                .tint(.green)
                .frame(maxWidth: 260)

            Text(String(format: "%d%% — Keep screen open", Int((recorder.sealingProgress * 100.0).rounded())))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.gray)
        }
        .padding(24)
        .background(Color.black.opacity(0.92))
        .clipShape(.rect(cornerRadius: 16))
        .padding(24)
    }

    private var tapeKeypadOverlay: some View {
        VStack(spacing: 12) {
            Text("Enter Tape-Measure Distance (meters)")
                .font(.headline)
            Picker("Role", selection: $tapeRoleInput) {
                Text("Primary").tag("primary")
                Text("Held-Out").tag("held_out")
            }
            .pickerStyle(.segmented)

            TextField("5.00", text: $tapeMetersInput)
                .keyboardType(.decimalPad)
                .font(.title2.monospacedDigit())
                .multilineTextAlignment(.center)
                .padding(8)
                .background(Color.white.opacity(0.12))
                .clipShape(.rect(cornerRadius: 8))

            HStack(spacing: 12) {
                Button("Cancel") {
                    showTapeBKeypad = false
                }
                .foregroundStyle(.red)
                Spacer()
                Button("Commit Tape B") {
                    if let val = Double(tapeMetersInput), val > 0 {
                        recorder.commitTapeEndpointB(tapeMeters: val, role: tapeRoleInput)
                    }
                    showTapeBKeypad = false
                }
                .font(.headline)
                .foregroundStyle(.green)
            }
        }
        .padding(16)
        .background(Color.black.opacity(0.92))
        .clipShape(.rect(cornerRadius: 12))
        .padding()
    }

    private var summaryView: some View {
        VStack(spacing: 18) {
            Text("Capture Summary")
                .font(.largeTitle.bold())

            Text("Provisional Grade: \(recorder.provisionalGrade)")
                .font(.title.bold())
                .foregroundStyle(recorder.provisionalGrade == "A" || recorder.provisionalGrade == "B" ? .green : .yellow)

            VStack(alignment: .leading, spacing: 8) {
                Text("Capture ID: \(recorder.captureID)")
                Text("Saved Depth Frames: \(recorder.savedFrameCount)")
                Text(String(format: "Ground Coverage: %.1f m² (%d cells)", recorder.coveredGroundAreaM2, recorder.coveredGroundCellCount))
                if !recorder.coveredGroundCells.isEmpty {
                    GroundCoverageMiniMapView(cells: recorder.coveredGroundCells)
                        .frame(height: 120)
                }
                Text("Tape Checks Logged: \(recorder.tapeCheckCount)")
                ForEach(recorder.tapeCheckSummaries, id: \.self) { summary in
                    Text("  • \(summary)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white)
                }
                if let gap = recorder.loopClosureGapMeters {
                    Text(String(format: "Marker 0 Loop-Closure Gap: %.1f cm", gap * 100.0))
                } else {
                    Text("Marker 0 Loop-Closure Gap: not revisited")
                }
                ForEach(recorder.provisionalReasons, id: \.self) { reason in
                    Text("• \(reason)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.gray)
                }
            }
            .font(.subheadline)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.08))
            .clipShape(.rect(cornerRadius: 12))

            if let zipURL = recorder.archiveZipURL {
                ShareLink(item: zipURL) {
                    Label("Export Capture Archive (.zip)", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .clipShape(.rect(cornerRadius: 12))
                }
            }

            Button {
                recorder.requestPermissionsAndStart()
            } label: {
                Text("New Capture")
                    .font(.headline)
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.white)
                    .clipShape(.rect(cornerRadius: 12))
            }
        }
        .padding(24)
    }

    private func hudButton(title: String, icon: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                Text(title)
                    .font(.caption2.bold())
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(color.opacity(0.8))
            .clipShape(.rect(cornerRadius: 8))
        }
    }
}

/// 3-segment Marker Lock Ring drawn on the marker in the camera view (`YS-74`, `YS-85`).
///
/// States (`DESIGN-SPEC-NATIVE.md` §3a):
/// - **Seen, not counting**: thin outline
/// - **Counting**: each OpenCV-confirmed still fills one of the 3 segments
/// - **Paused**: filled segments stay, nothing resets, empty segments dim; when too far (`showStepCloserChevron`), shows a chevron pointing toward the operator
/// - **Done**: solid disc with a checkmark
public struct MarkerRingOverlayView: View {
    public let ringState: MarkerRingVisualState
    public let animationsStatic: Bool

    public init(ringState: MarkerRingVisualState, animationsStatic: Bool = false) {
        self.ringState = ringState
        self.animationsStatic = animationsStatic
    }

    public static func normalizedCenter(for ringState: MarkerRingVisualState, rgbWidth: Double = 1920.0, rgbHeight: Double = 1440.0) -> CGPoint {
        let pt = ringState.normalizedPortraitCenter(rgbWidth: rgbWidth, rgbHeight: rgbHeight)
        return CGPoint(x: pt.x, y: pt.y)
    }

    public var body: some View {
        ZStack {
            if ringState.isLocked {
                Circle()
                    .fill(Color.green.opacity(0.88))
                    .frame(width: 78, height: 78)
                Image(systemName: "checkmark")
                    .font(.title.bold())
                    .foregroundStyle(.white)
            } else {
                ForEach(0..<3, id: \.self) { segIndex in
                    let isFilled = segIndex < ringState.confirmedSegments
                    let startDeg = -90.0 + Double(segIndex) * 120.0 + 8.0
                    let endDeg = startDeg + 104.0
                    Circle()
                        .trim(
                            from: CGFloat((startDeg + 90.0) / 360.0),
                            to: CGFloat((endDeg + 90.0) / 360.0)
                        )
                        .stroke(
                            isFilled
                                ? Color.green
                                : Color.white.opacity(ringState.isPaused ? 0.25 : 0.55),
                            style: StrokeStyle(
                                lineWidth: isFilled ? 7.0 : (ringState.confirmedSegments == 0 ? 3.5 : 5.5),
                                lineCap: .round
                            )
                        )
                        .rotationEffect(.degrees(-90))
                        .frame(width: 84, height: 84)
                }

                VStack(spacing: 2) {
                    Text("\(ringState.markerID)")
                        .font(.headline.monospacedDigit().bold())
                        .foregroundStyle(.white)
                    if ringState.showStepCloserChevron {
                        Image(systemName: "chevron.compact.down")
                            .font(.caption.bold())
                            .foregroundStyle(.orange)
                    }
                }
            }
        }
        .animation(animationsStatic ? nil : .easeInOut(duration: 0.2), value: ringState)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            ringState.isLocked
                ? "Marker \(ringState.markerID) captured"
                : "Marker \(ringState.markerID), \(ringState.confirmedSegments) of 3 photos confirmed"
        )
    }
}

/// Top-down `0.25 m` ground coverage grid view (`YS-78`).
public struct GroundCoverageMiniMapView: View {
    public let cells: Set<GroundGridCell>

    public init(cells: Set<GroundGridCell>) {
        self.cells = cells
    }

    public var body: some View {
        Canvas { context, size in
            guard !cells.isEmpty else { return }
            let xs = cells.map(\.ix)
            let zs = cells.map(\.iz)
            guard let minX = xs.min(), let maxX = xs.max(),
                  let minZ = zs.min(), let maxZ = zs.max()
            else { return }

            let spanX = max(1, maxX - minX + 1)
            let spanZ = max(1, maxZ - minZ + 1)
            let maxSpan = CGFloat(max(spanX, spanZ))
            let padding: CGFloat = 8.0
            let usableW = max(1.0, size.width - 2.0 * padding)
            let usableH = max(1.0, size.height - 2.0 * padding)
            let cellPx = min(usableW / maxSpan, usableH / maxSpan)
            let offsetX = padding + (usableW - CGFloat(spanX) * cellPx) * 0.5
            let offsetY = padding + (usableH - CGFloat(spanZ) * cellPx) * 0.5

            for cell in cells {
                let col = CGFloat(cell.ix - minX)
                let row = CGFloat(cell.iz - minZ)
                let rect = CGRect(
                    x: offsetX + col * cellPx,
                    y: offsetY + row * cellPx,
                    width: max(1.5, cellPx - 0.5),
                    height: max(1.5, cellPx - 0.5)
                )
                context.fill(Path(rect), with: .color(.green.opacity(0.75)))
            }
        }
        .background(.regularMaterial)
        .clipShape(.rect(cornerRadius: 10))
        .accessibilityHidden(true)
    }
}
