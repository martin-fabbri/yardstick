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

/// Main SwiftUI operator interface for Yardstick (`YS-00`, `YS-05`, `YS-07`, `YS-12`, `YS-22`, `YS-40`, `YS-44`).
///
/// Screens:
/// 1. **Pre-walk Setup**: verifies LiDAR support (`YS-00`), configures printed marker sizes (`250 mm` Letter/A4 or `180 mm` clipboard) for Markers `0–5` (`YS-33`).
/// 2. **Live Recording HUD**: camera preview + center crosshair for 5×5 LiDAR patch tape checks (`YS-40`), live pitch/roll guide (`YS-05`, `YS-12`), marker & North calibration badges (`YS-14`), and 5 action buttons (`Marker`, `Tape A`, `Tape B`, `Note`, `Done`, `YS-07`).
/// 3. **Post-walk Summary**: displays loop-closure gap (`YS-32`), provisional grade & reasons (`YS-42`, `YS-44`), and one-tap AirDrop / Files export (`YS-22`).
public struct ContentView: View {
    @StateObject private var recorder = CaptureSessionRecorder()
    @State private var showTapeBKeypad = false
    @State private var tapeMetersInput = "5.00"
    @State private var tapeRoleInput = "primary"
    @State private var showNoteDialog = false
    @State private var noteTextInput = ""
    @State private var showShareSheet = false

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
    }

    private var setupView: some View {
        VStack(spacing: 20) {
            Text("Yardstick")
                .font(.largeTitle.bold())
                .foregroundColor(.white)

            Text("Calibrated iPhone LiDAR Yard Capture (yardstick-capture/1)")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)

            if !CaptureSessionRecorder.isLiDARSupported {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.yellow)
                    Text("LiDAR sensor required on physical iPhone Pro / Pro Max (YS-00)")
                        .font(.footnote)
                        .foregroundColor(.yellow)
                }
                .padding(10)
                .background(Color.white.opacity(0.08))
                .cornerRadius(8)
            }

            VStack(alignment: .leading, spacing: 12) {
                Text("Printed Marker Edge Size (tap to toggle 250 mm / 180 mm)")
                    .font(.headline)
                    .foregroundColor(.white)

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
                                .cornerRadius(6)
                        }
                        .foregroundColor(.white)
                        .padding(.vertical, 4)
                    }
                }
            }
            .padding()
            .background(Color.white.opacity(0.06))
            .cornerRadius(12)

            Button {
                recorder.requestPermissionsAndStart()
            } label: {
                Text("Start Yard Walk")
                    .font(.headline)
                    .foregroundColor(.black)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .cornerRadius(12)
            }
        }
        .padding(24)
    }

    private var recordingHUDView: some View {
        ZStack {
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
                        .cornerRadius(4)
                }
                Spacer()
            }

            VStack {
                // Top telemetry & guidance bar (YS-05, YS-12, YS-14)
                VStack(alignment: .leading, spacing: 6) {
                    Text(recorder.statusBanner)
                        .font(.subheadline.bold())
                        .foregroundColor(.white)

                    HStack(spacing: 12) {
                        Label("Frames: \(recorder.savedFrameCount)", systemImage: "video.fill")
                        Label("Markers: \(recorder.detectedMarkerIDs.count)", systemImage: "qrcode.viewfinder")
                        Label("North: \(recorder.northSampleCount)/10", systemImage: "location.north.line.fill")
                        Label("Tapes: \(recorder.tapeCheckCount)", systemImage: "ruler.fill")
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.green)

                    HStack(spacing: 16) {
                        Text(String(format: "Pitch: %.1f° (target -30°)", recorder.pitchDeg))
                        Text(String(format: "Roll: %.1f°", recorder.rollDeg))
                        Text("Track: \(recorder.trackingStateText)")
                    }
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(.gray)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.75))

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
                        showTapeBKeypad = true
                    }
                    hudButton(title: "Note", icon: "mic.fill", color: .purple) {
                        recorder.logEvent(kind: "note", detail: "Spoken note anchor")
                    }
                    hudButton(title: "Done", icon: "checkmark.seal.fill", color: .green) {
                        recorder.stopAndSealCapture()
                    }
                }
                .padding(12)
                .background(Color.black.opacity(0.85))
            }
        }
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
                .cornerRadius(8)

            HStack(spacing: 12) {
                Button("Cancel") {
                    showTapeBKeypad = false
                }
                .foregroundColor(.red)
                Spacer()
                Button("Commit Tape B") {
                    if let val = Double(tapeMetersInput), val > 0 {
                        recorder.commitTapeEndpointB(tapeMeters: val, role: tapeRoleInput)
                    }
                    showTapeBKeypad = false
                }
                .font(.headline)
                .foregroundColor(.green)
            }
        }
        .padding(16)
        .background(Color.black.opacity(0.92))
        .cornerRadius(12)
        .padding()
    }

    private var summaryView: some View {
        VStack(spacing: 18) {
            Text("Capture Summary")
                .font(.largeTitle.bold())

            Text("Provisional Grade: \(recorder.provisionalGrade)")
                .font(.title.bold())
                .foregroundColor(recorder.provisionalGrade == "A" || recorder.provisionalGrade == "B" ? .green : .yellow)

            VStack(alignment: .leading, spacing: 8) {
                Text("Capture ID: \(recorder.captureID)")
                Text("Saved Depth Frames: \(recorder.savedFrameCount)")
                Text("Tape Checks Logged: \(recorder.tapeCheckCount)")
                if let gap = recorder.loopClosureGapMeters {
                    Text(String(format: "Marker 0 Loop-Closure Gap: %.1f cm", gap * 100.0))
                } else {
                    Text("Marker 0 Loop-Closure Gap: not revisited")
                }
                ForEach(recorder.provisionalReasons, id: \.self) { reason in
                    Text("• \(reason)")
                        .font(.caption.monospaced())
                        .foregroundColor(.gray)
                }
            }
            .font(.subheadline)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.white.opacity(0.08))
            .cornerRadius(12)

            if let zipURL = recorder.archiveZipURL {
                ShareLink(item: zipURL) {
                    Label("Export Capture Archive (.zip)", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(12)
                }
            }

            Button {
                recorder.requestPermissionsAndStart()
            } label: {
                Text("New Capture")
                    .font(.headline)
                    .foregroundColor(.black)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.white)
                    .cornerRadius(12)
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
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(color.opacity(0.8))
            .cornerRadius(8)
        }
    }
}
