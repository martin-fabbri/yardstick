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
        let coachingOverlay = ARCoachingOverlayView()
        coachingOverlay.session = session
        coachingOverlay.goal = .horizontalPlane
        coachingOverlay.activatesAutomatically = true
        coachingOverlay.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(coachingOverlay)
        NSLayoutConstraint.activate([
            coachingOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            coachingOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            coachingOverlay.topAnchor.constraint(equalTo: view.topAnchor),
            coachingOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
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

/// Root SwiftUI navigation and guided capture experience (`YS-72`, `YS-80`, `YS-81`, `YS-83`, `YS-84`, `DESIGN-SPEC-NATIVE.md`).
///
/// Architecture (`DESIGN-SPEC-NATIVE.md` §2a):
/// - **Captures** (`NavigationStack`, root): follows the user's system Light/Dark color scheme (`P7`), shows `ContentUnavailableView`
///   when empty or an `.insetGrouped` `List` of completed walks, with a toolbar Settings button and a bottom `.glassProminent` **New Capture** button.
/// - **Welcome / Get Your Markers / Lay Your Markers**: 3-step onboarding `.sheet`, re-openable from Settings and the empty state.
/// - **Before You Start** (`.sheet`, `.large` detent): `Form` of pre-flight checks (`YS-76`), print check card, centre-to-centre
///   `Marker 0 -> Marker 1` tape entry (`YS-77`), and **Start Walk**.
/// - **Walk** (`.fullScreenCover`, forced `.dark` scheme): full-screen RealityKit `ARView`, top-leading `✕` (`End walk`),
///   single-line regular Liquid Glass instruction capsule with outdoor contrast backing (`YS-81`), trailing tilt gauge,
///   3-segment marker lock ring (`YS-74`), and bottom `GlassEffectContainer` (`Note` / `Skip`, `Marker progress`, and conditional `Finish`).
/// - **Review Walk** (`.sheet` over the still-running walk, `.medium` / `.large` detents): top-down yard map on `.regularMaterial`,
///   verdict headline, at most 3 fix items (`YS-72`, `YS-80` #1), **Save Capture** (swapping to determinate `"Saving your walk"` progress),
///   and **Keep Walking**.
/// - **Capture Detail**: pushed onto `Captures` with plain-language verdict, accuracy sentence, `CAPTURED` rows, `DETAILS` disclosure group,
///   and `ShareLink`.
public struct ContentView: View {
    @StateObject private var recorder = CaptureSessionRecorder()
    @State private var navigationPath: [SavedCaptureRecord.ID] = []
    @State private var showOnboardingSheet = false
    @State private var showBeforeYouStartSheet = false
    @State private var showSettingsSheet = false
    @State private var showWalkFullScreenCover = false
    @State private var showReviewWalkSheet = false
    @State private var renamingRecordID: String? = nil
    @State private var renamingTitleInput: String = ""
    @State private var didTriggerLaunchArgs = false

    public init() {}

    public var body: some View {
        NavigationStack(path: $navigationPath) {
            capturesRootView
                .navigationTitle("Captures")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            showSettingsSheet = true
                        } label: {
                            Image(systemName: "gearshape")
                        }
                        .accessibilityLabel("Settings")
                    }
                }
                .safeAreaInset(edge: .bottom) {
                    Button {
                        recorder.resetToSetupForNewCapture()
                        showBeforeYouStartSheet = true
                    } label: {
                        Label("New Capture", systemImage: "plus")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent)
                    .controlSize(.large)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .accessibilityShowsLargeContentViewer()
                }
                .navigationDestination(for: SavedCaptureRecord.ID.self) { recordID in
                    if let record = recorder.savedCaptureRecords.first(where: { $0.id == recordID }) {
                        CaptureDetailView(record: record)
                    }
                }
        }
        .sheet(isPresented: $showOnboardingSheet) {
            WelcomeOnboardingSheetView(isPresented: $showOnboardingSheet)
        }
        .sheet(isPresented: $showBeforeYouStartSheet) {
            BeforeYouStartSheetView(
                recorder: recorder,
                isPresented: $showBeforeYouStartSheet,
                onStartWalk: {
                    showBeforeYouStartSheet = false
                    showWalkFullScreenCover = true
                    recorder.requestPermissionsAndStart()
                }
            )
            .presentationDetents([.large])
        }
        .sheet(isPresented: $showSettingsSheet) {
            SettingsSheetView(
                recorder: recorder,
                isPresented: $showSettingsSheet,
                onOpenOnboarding: {
                    showSettingsSheet = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                        showOnboardingSheet = true
                    }
                }
            )
        }
        .fullScreenCover(isPresented: $showWalkFullScreenCover) {
            WalkFullScreenCoverView(
                recorder: recorder,
                showReviewWalkSheet: $showReviewWalkSheet,
                onDismissAfterSave: { savedRecordID in
                    showReviewWalkSheet = false
                    showWalkFullScreenCover = false
                    if let id = savedRecordID {
                        navigationPath = [id]
                    }
                }
            )
        }
        .onChange(of: recorder.phase) { _, newPhase in
            if newPhase == .completed, showWalkFullScreenCover {
                let latestID = recorder.savedCaptureRecords.first?.id
                showReviewWalkSheet = false
                showWalkFullScreenCover = false
                if let id = latestID {
                    navigationPath = [id]
                }
            }
        }
        .onAppear {
            handleLaunchArgumentsIfNeeded()
        }
    }

    @ViewBuilder
    private var capturesRootView: some View {
        if recorder.savedCaptureRecords.isEmpty {
            ContentUnavailableView {
                Label("No Captures Yet", systemImage: "map")
            } description: {
                Text("Lay your four markers in the yard, then walk it with your iPhone.")
            } actions: {
                Button("How to Lay Markers") {
                    showOnboardingSheet = true
                }
                .buttonStyle(.glass)
            }
        } else {
            List {
                ForEach(recorder.savedCaptureRecords) { record in
                    NavigationLink(value: record.id) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(record.title)
                                .font(.headline)
                                .foregroundStyle(.primary)
                            Text(record.dateText)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            HStack(spacing: 6) {
                                Image(systemName: record.detailSummary.verdictSymbol)
                                    .foregroundStyle(record.detailSummary.isVerdictPassing ? .green : .orange)
                                Text(record.detailSummary.verdictTitle)
                                    .font(.footnote.weight(.semibold))
                                    .foregroundStyle(.primary)
                            }
                        }
                        .padding(.vertical, 4)
                        .accessibilityElement(children: .combine)
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) {
                            recorder.savedCaptureRecords.removeAll { $0.id == record.id }
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                        Button {
                            renamingRecordID = record.id
                            renamingTitleInput = record.title
                        } label: {
                            Label("Rename", systemImage: "pencil")
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .alert(
                "Rename Capture",
                isPresented: Binding(
                    get: { renamingRecordID != nil },
                    set: { if !$0 { renamingRecordID = nil } }
                )
            ) {
                TextField("Back yard", text: $renamingTitleInput)
                Button("Save") {
                    if let targetID = renamingRecordID,
                       let idx = recorder.savedCaptureRecords.firstIndex(where: { $0.id == targetID }) {
                        let trimmed = renamingTitleInput.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.isEmpty {
                            recorder.savedCaptureRecords[idx].title = trimmed
                        }
                    }
                    renamingRecordID = nil
                }
                Button("Cancel", role: .cancel) {
                    renamingRecordID = nil
                }
            }
        }
    }

    private func handleLaunchArgumentsIfNeeded() {
        guard !didTriggerLaunchArgs else { return }
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--bench-seal-dir"), idx + 1 < args.count {
            didTriggerLaunchArgs = true
            let dirURL = URL(fileURLWithPath: args[idx + 1], isDirectory: true)
            _ = try? CaptureSessionRecorder.sealManifestAndPackZip(in: dirURL)
            return
        }
        if let idx = args.firstIndex(of: "--auto-bench-capture"),
           idx + 1 < args.count,
           let seconds = Double(args[idx + 1]),
           seconds > 0,
           recorder.phase == .setup {
            didTriggerLaunchArgs = true
            showWalkFullScreenCover = true
            recorder.requestPermissionsAndStart()
            DispatchQueue.main.asyncAfter(deadline: .now() + seconds) {
                if recorder.phase == .recording {
                    recorder.stopAndSealCapture()
                }
            }
            return
        }
        if let screenArg = args.first(where: { $0.hasPrefix("--ui-test-screen=") }) {
            didTriggerLaunchArgs = true
            let target = String(screenArg.dropFirst("--ui-test-screen=".count))
            seedSampleCaptureRecordForPreview()
            switch target {
            case "onboarding":
                showOnboardingSheet = true
            case "preflight":
                showBeforeYouStartSheet = true
            case "settings":
                showSettingsSheet = true
            case "walk":
                showWalkFullScreenCover = true
            case "review":
                showWalkFullScreenCover = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    _ = recorder.evaluateWalkReview()
                    showReviewWalkSheet = true
                }
            case "detail":
                if let firstID = recorder.savedCaptureRecords.first?.id {
                    navigationPath = [firstID]
                }
            default:
                break
            }
        }
    }

    private func seedSampleCaptureRecordForPreview() {
        guard recorder.savedCaptureRecords.isEmpty else { return }
        var sampleCells: Set<GroundGridCell> = []
        for ix in -6...6 {
            for iz in -5...5 where abs(ix) + abs(iz) <= 9 {
                sampleCells.insert(GroundGridCell(ix: ix, iz: iz))
            }
        }
        let summary = CaptureDetailSummary.build(
            provisionalGrade: "A",
            status: "stopped",
            distanceTolerance10mM: 0.028,
            lockedMarkerIDs: [0, 1, 2, 3],
            isLoopClosed: true,
            isLookUpDone: true,
            wasLookUpSkipped: false,
            primaryTapeMeters: 6.00
        )
        recorder.savedCaptureRecords = [
            SavedCaptureRecord(
                id: "20261101T170000Z_yardstick",
                title: "Back yard",
                dateText: "Sun, Nov 1 · 8 min",
                durationSeconds: 480,
                savedFrameCount: 2865,
                coveredGroundCells: sampleCells,
                coveredGroundAreaM2: Double(sampleCells.count) * 0.0625,
                detailSummary: summary,
                archiveZipURL: nil
            ),
        ]
    }
}

// MARK: - 1. First-Run Onboarding Sheet (`DESIGN-SPEC-NATIVE.md` §2a, §2c)

private struct WelcomeOnboardingSheetView: View {
    @Binding var isPresented: Bool
    @State private var stepIndex: Int = 0

    var body: some View {
        NavigationStack {
            Group {
                switch stepIndex {
                case 0:
                    welcomeStep
                case 1:
                    getMarkersStep
                default:
                    layMarkersStep
                }
            }
            .navigationTitle(stepTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                if stepIndex > 0 {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("Back") {
                            stepIndex = max(0, stepIndex - 1)
                        }
                    }
                } else {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Skip") {
                            isPresented = false
                        }
                    }
                }
            }
        }
        .interactiveDismissDisabled(stepIndex == 0)
    }

    private var stepTitle: String {
        switch stepIndex {
        case 0: return "Measure Your Yard"
        case 1: return "Get Your Markers"
        default: return "Lay Your Markers"
        }
    }

    private var welcomeStep: some View {
        List {
            Section {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Print four markers")
                            .font(.headline)
                        Text("Plain paper is fine.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: "printer.fill")
                        .font(.title2)
                }
                .padding(.vertical, 4)

                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Lay them in the yard")
                            .font(.headline)
                        Text("One by the door, three near the corners.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: "square.grid.2x2.fill")
                        .font(.title2)
                }
                .padding(.vertical, 4)

                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Walk the yard")
                            .font(.headline)
                        Text("Start and finish at marker 0.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                } icon: {
                    Image(systemName: "figure.walk")
                        .font(.title2)
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.insetGrouped)
        .safeAreaInset(edge: .bottom) {
            Button {
                stepIndex = 1
            } label: {
                Text("Continue")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glassProminent)
            .controlSize(.large)
            .padding(20)
        }
    }

    private var getMarkersStep: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 12) {
                        ForEach(0..<4, id: \.self) { id in
                            Image(systemName: "\(id).square.fill")
                                .font(.largeTitle)
                                .symbolRenderingMode(.hierarchical)
                        }
                    }
                    .padding(.vertical, 6)
                    Text("Print at Actual Size, not \"Fit to Page\".")
                        .font(.body)
                        .foregroundStyle(.primary)
                }
                .padding(.vertical, 6)
            }
        }
        .listStyle(.insetGrouped)
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 10) {
                Button {
                    stepIndex = 2
                } label: {
                    Text("I Already Have Markers")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
            }
            .padding(20)
        }
    }

    private var layMarkersStep: some View {
        List {
            Section {
                Label("Marker 0 is your start. Put it by the back door.", systemImage: "0.circle.fill")
                Label("Put 1, 2 and 3 near the corners and along the fence.", systemImage: "square.dashed")
                Label("Lay them flat, out of shade.", systemImage: "sun.max.fill")
            }
        }
        .listStyle(.insetGrouped)
        .safeAreaInset(edge: .bottom) {
            Button {
                isPresented = false
            } label: {
                Text("Done")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glassProminent)
            .controlSize(.large)
            .padding(20)
        }
    }
}

// MARK: - 2. Before You Start Pre-Flight Sheet (`YS-76`, `YS-77`, `YS-80` #3 & #4, `YS-81`)

private struct BeforeYouStartSheetView: View {
    @ObservedObject var recorder: CaptureSessionRecorder
    @Binding var isPresented: Bool
    let onStartWalk: () -> Void

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor
    @State private var preWalkTapeInput: String = ""
    @State private var customPrintSizeCMInput: String = "18.0"
    @State private var showPrintCustomSizeField: Bool = false
    @FocusState private var isTapeFieldFocused: Bool

    private var isSimulatorPreviewAllowed: Bool {
        ProcessInfo.processInfo.arguments.contains { $0.hasPrefix("--ui-test-screen=") }
    }

    var body: some View {
        NavigationStack {
            Group {
                if !CaptureSessionRecorder.isLiDARSupported && !isSimulatorPreviewAllowed {
                    ContentUnavailableView(
                        "This iPhone Can't Measure Yards",
                        systemImage: "camera.metering.unknown",
                        description: Text("Yardstick requires an iPhone Pro or Pro Max with a LiDAR Scanner.")
                    )
                } else {
                    preFlightForm
                }
            }
            .navigationTitle("Before You Start")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                let canStart = (CaptureSessionRecorder.isLiDARSupported || isSimulatorPreviewAllowed)
                    && recorder.preFlightReport.canStartWalk
                Button {
                    onStartWalk()
                } label: {
                    Text("Start Walk")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
                .disabled(!canStart)
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .accessibilityShowsLargeContentViewer()
            }
        }
        .onAppear {
            if let existingTape = recorder.preWalkTapeMeters0To1, existingTape > 0 {
                preWalkTapeInput = String(format: "%.2f", existingTape)
            }
            recorder.startPreFlightCameraProbeIfAuthorized()
            _ = recorder.evaluatePreFlightChecklist()
        }
    }

    private var preFlightForm: some View {
        Form {
            Section {
                ForEach(recorder.preFlightReport.visibleRows, id: \.id) { row in
                    if row.id != "print_check" && row.id != "tape_measurement" {
                        preFlightStatusRow(row)
                    }
                }
            }

            Section {

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: recorder.isPrintSizeConfirmed ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(recorder.isPrintSizeConfirmed ? .green : .secondary)
                        Text("Check your print")
                            .font(.body)
                        Spacer()
                        if differentiateWithoutColor {
                            Text(recorder.isPrintSizeConfirmed ? "Passed" : "Check")
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                    }
                    Text(recorder.preFlightReport.printCheckRow.detail)
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    if !recorder.isPrintSizeConfirmed {
                        HStack(spacing: 10) {
                            Button("Yes") {
                                recorder.isPrintSizeConfirmed = true
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)

                            Button("No, It Measures…") {
                                showPrintCustomSizeField.toggle()
                            }
                            .buttonStyle(.glass)
                            .controlSize(.small)
                        }
                    }

                    if showPrintCustomSizeField {
                        HStack(spacing: 8) {
                            TextField("18.0", text: $customPrintSizeCMInput)
                                .keyboardType(.decimalPad)
                            Text("cm")
                                .foregroundStyle(.secondary)
                            Button("Save") {
                                if let cm = Double(customPrintSizeCMInput), cm >= 8.0 && cm <= 40.0 {
                                    recorder.setAllMarkersPrintedSizeMM(cm * 10.0)
                                    recorder.isPrintSizeConfirmed = true
                                    showPrintCustomSizeField = false
                                }
                            }
                            .buttonStyle(.glassProminent)
                            .controlSize(.small)
                        }
                    }
                }
                .padding(.vertical, 4)

                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: (recorder.preWalkTapeMeters0To1 ?? 0) > 0 ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle((recorder.preWalkTapeMeters0To1 ?? 0) > 0 ? .green : .secondary)
                        Text("Tape measurement")
                    }
                    Text("Measure from the centre of marker 0 to the centre of marker 1.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        TextField("Metres, for example 3.00", text: $preWalkTapeInput)
                            .keyboardType(.decimalPad)
                            .focused($isTapeFieldFocused)
                            .textFieldStyle(.roundedBorder)
                            .font(.title3)
                            .accessibilityLabel("Tape measurement in metres")
                            .onChange(of: preWalkTapeInput) { _, newValue in
                                let normalized = newValue
                                    .trimmingCharacters(in: .whitespaces)
                                    .replacingOccurrences(of: ",", with: ".")
                                if let val = Double(normalized), val > 0 {
                                    recorder.setPreWalkTapeMeters0To1(val)
                                } else if normalized.isEmpty {
                                    recorder.setPreWalkTapeMeters0To1(nil)
                                }
                            }
                        Text("m")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    Picker("Tape lay", selection: $recorder.tapeSurface) {
                        ForEach(CaptureSessionRecorder.TapeSurface.allCases, id: \.self) { surface in
                            Text(surface.label).tag(surface)
                        }
                    }
                    .pickerStyle(.segmented)
                    .accessibilityLabel("Where the tape lay")
                }
                .padding(.vertical, 4)
                .contentShape(Rectangle())
                .onTapGesture { isTapeFieldFocused = true }
            } footer: {
                Text("Stops you: too dark, no space, too hot.")
            }
        }
        .scrollDismissesKeyboard(.interactively)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { isTapeFieldFocused = false }
            }
        }
    }

    @ViewBuilder
    private func preFlightStatusRow(_ row: PreFlightCheckRow) -> some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: preFlightSymbol(for: row.state))
                .foregroundStyle(preFlightColor(for: row.state))
                .symbolRenderingMode(.hierarchical)
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title)
                    .font(.body)
                    .foregroundStyle(.primary)
                Text(row.detail)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if differentiateWithoutColor {
                Text(preFlightStateWord(for: row.state))
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            if row.id == "lens", row.actionTitle != nil {
                Button("Check Again") {
                    _ = recorder.resetLensSmudgeProbeAndCheckAgain()
                }
                .buttonStyle(.glassProminent)
                .controlSize(.small)
            } else if row.id == "storage" && row.state == .block,
                      let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                Link("Open Settings", destination: settingsURL)
                    .font(.footnote.weight(.semibold))
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func preFlightSymbol(for state: PreFlightCheckState) -> String {
        switch state {
        case .pass: return "checkmark.circle.fill"
        case .warn: return "exclamationmark.triangle.fill"
        case .block: return "xmark.octagon.fill"
        case .hidden: return "circle"
        }
    }

    private func preFlightColor(for state: PreFlightCheckState) -> Color {
        switch state {
        case .pass: return .green
        case .warn: return .orange
        case .block: return .red
        case .hidden: return .secondary
        }
    }

    private func preFlightStateWord(for state: PreFlightCheckState) -> String {
        switch state {
        case .pass: return "Passed"
        case .warn: return "Check"
        case .block: return "Fix"
        case .hidden: return ""
        }
    }
}

// MARK: - 3. Live Walk Full-Screen Cover (`YS-72`, `YS-74`, `YS-80`, `YS-81`)

private struct WalkFullScreenCoverView: View {
    @ObservedObject var recorder: CaptureSessionRecorder
    @Binding var showReviewWalkSheet: Bool
    let onDismissAfterSave: (SavedCaptureRecord.ID?) -> Void

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var colorSchemeContrast
    @Namespace private var bottomGlassNamespace
    @State private var showEndEarlyConfirmation = false

    var body: some View {
        ZStack {
            // Live ARKit Camera Preview via RealityKit ARView (`YS-83`, `YS-81`: camera feed is hidden from VoiceOver)
            #if canImport(ARKit) && canImport(RealityKit)
            ARCameraPreviewView(session: recorder.arSession)
                .ignoresSafeArea()
                .accessibilityHidden(true)
            #endif

            // ② 3-Segment Marker Lock Ring drawn on the marker in the camera view (`YS-74`, `YS-85`)
            if let ringState = recorder.activeRingState {
                GeometryReader { proxy in
                    let normCenter = MarkerRingOverlayView.normalizedCenter(
                        for: ringState,
                        rgbWidth: Double(recorder.actualWidth),
                        rgbHeight: Double(recorder.actualHeight)
                    )
                    MarkerRingOverlayView(
                        ringState: ringState,
                        animationsStatic: recorder.thermalSchedule.animationsStatic || reduceMotion
                    )
                    .position(
                        x: normCenter.x * proxy.size.width,
                        y: normCenter.y * proxy.size.height
                    )
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
            }

            // ④ In-view Marker Beacons (`0.40 m` above each seen marker) & ⑤ Off-screen Edge Arrows (`DESIGN-SPEC-NATIVE.md` §3a ④ & ⑤)
            if !recorder.beaconProjections.isEmpty {
                MarkerBeaconsAndEdgeArrowsOverlayView(projections: recorder.beaconProjections)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }

            // ③ Trailing-edge Tilt Gauge (`DESIGN-SPEC-NATIVE.md` §3a ③: hidden during a hold, silent, accessibilityHidden)
            if recorder.activeRingState == nil && recorder.guidedWalkPhase != .houseWallLookUp {
                HStack {
                    Spacer()
                    TiltGaugeOverlayView(pitchDeg: recorder.pitchDeg)
                        .padding(.trailing, 14)
                }
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            }

            // Top controls: ⑦ ✕ End walk (top-leading) + ① Instruction capsule (top-centre on regular Liquid Glass)
            VStack(spacing: 12) {
                HStack {
                    Button {
                        showEndEarlyConfirmation = true
                    } label: {
                        Image(systemName: "xmark")
                            .font(.headline.weight(.semibold))
                            .frame(minWidth: 44, minHeight: 44)
                    }
                    .buttonStyle(.glass)
                    .buttonBorderShape(.circle)
                    .accessibilityLabel("End walk")
                    .accessibilityShowsLargeContentViewer()

                    Spacer()

                    if !recorder.coveredGroundCells.isEmpty {
                        GroundCoverageMiniMapView(cells: recorder.coveredGroundCells)
                            .frame(width: 76, height: 76)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                // ① Single-line instruction capsule on regular Liquid Glass with outdoor sun dimming layer (`YS-80`, `YS-81`)
                let dimOpacity = (colorSchemeContrast == .increased) ? 0.55 : 0.35
                Text(recorder.statusBanner)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.center)
                    .lineLimit(dynamicTypeSize.isAccessibilitySize ? 3 : 2)
                    .contentTransition(.opacity)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(
                        Capsule()
                            .fill(Color.black.opacity(dimOpacity))
                    )
                    .glassEffect(.regular, in: Capsule())
                    .padding(.horizontal, 20)
                    .accessibilitySortPriority(100)
                    .accessibilityAddTraits(.updatesFrequently)

                // Developer-only live diagnostics overlay (`Settings ▸ Developer ▸ Live Diagnostics`, off by default per `YS-80` §3c)
                if recorder.showDeveloperDiagnosticsOverlay {
                    DeveloperDiagnosticsOverlayView(recorder: recorder)
                        .padding(.horizontal, 16)
                }

                Spacer()
            }
        }
        .safeAreaInset(edge: .bottom) {
            bottomControlGroup
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
        }
        .preferredColorScheme(.dark)
        .modifier(HideStatusBarToolbarModifier())
        .persistentSystemOverlays(.hidden)
        .interactiveDismissDisabled()
        // YS-81: VoiceOver Magic Tap (two-finger double tap) starts and stops a voice note
        .accessibilityAction(.magicTap) {
            recorder.setRecordingVoiceNote(!recorder.isRecordingVoiceNote)
        }
        .confirmationDialog(
            "End walk early?",
            isPresented: $showEndEarlyConfirmation,
            titleVisibility: .visible
        ) {
            Button("End Walk", role: .destructive) {
                recorder.markEndEarlyChosen()
                _ = recorder.evaluateWalkReview()
                showReviewWalkSheet = true
            }
            Button("Keep Walking", role: .cancel) {}
        } message: {
            Text("You haven't returned to marker 0, so this walk probably won't be accepted.")
        }
        .sheet(isPresented: $showReviewWalkSheet) {
            ReviewWalkSheetView(
                recorder: recorder,
                isPresented: $showReviewWalkSheet,
                onCompletedSave: {
                    let latestID = recorder.savedCaptureRecords.first?.id
                    onDismissAfterSave(latestID)
                }
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
            .interactiveDismissDisabled()
            .presentationBackgroundInteraction(.disabled)
        }
    }

    @ViewBuilder
    private var bottomControlGroup: some View {
        GlassEffectContainer(spacing: 14) {
            if dynamicTypeSize.isAccessibilitySize {
                // YS-81 / DESIGN-SPEC-NATIVE.md §6 line 541: At AX sizes, stack Note & Finish above the marker capsule
                VStack(spacing: 10) {
                    HStack(spacing: 14) {
                        leadingActionControl
                        if recorder.isFinishAvailable {
                            finishButton
                        }
                    }
                    MarkerProgressCapsuleView(
                        lockedMarkerIDs: recorder.lockedMarkerIDs,
                        seenMarkerIDs: recorder.detectedMarkerIDs,
                        phase: recorder.guidedWalkPhase,
                        isFinishAvailable: recorder.isFinishAvailable,
                        isAccessibilitySize: true,
                        reduceMotion: reduceMotion || recorder.thermalSchedule.animationsStatic
                    )
                    .glassEffectID("markerProgress", in: bottomGlassNamespace)
                }
            } else {
                HStack(spacing: 14) {
                    leadingActionControl
                    Spacer(minLength: 4)
                    MarkerProgressCapsuleView(
                        lockedMarkerIDs: recorder.lockedMarkerIDs,
                        seenMarkerIDs: recorder.detectedMarkerIDs,
                        phase: recorder.guidedWalkPhase,
                        isFinishAvailable: recorder.isFinishAvailable,
                        isAccessibilitySize: false,
                        reduceMotion: reduceMotion || recorder.thermalSchedule.animationsStatic
                    )
                    .glassEffectID("markerProgress", in: bottomGlassNamespace)
                    Spacer(minLength: 4)
                    // ⑥c Finish does NOT exist until YS-71 + closing hold at Marker 0 pass (`YS-72`)
                    if recorder.isFinishAvailable {
                        finishButton
                    }
                }
            }
        }
        .disabled(recorder.phase == .sealing)
    }

    @ViewBuilder
    private var leadingActionControl: some View {
        if recorder.guidedWalkPhase == .houseWallLookUp {
            // DESIGN-SPEC-NATIVE.md §3b: A Skip button replaces Note during the look-up step
            Button("Skip") {
                recorder.skipLookUpStep()
            }
            .buttonStyle(.glass)
            .controlSize(.extraLarge)
            .glassEffectID("leadingAction", in: bottomGlassNamespace)
            .accessibilityShowsLargeContentViewer()
        } else {
            // ⑥a Note button: circular glass button with `mic.fill` -> `waveform` variable-color effect while recording
            Button {
                recorder.setRecordingVoiceNote(!recorder.isRecordingVoiceNote)
            } label: {
                Image(systemName: recorder.isRecordingVoiceNote ? "waveform" : "mic.fill")
                    .font(.title3.weight(.semibold))
                    .symbolEffect(
                        .variableColor.iterative,
                        isActive: recorder.isRecordingVoiceNote && !reduceMotion && !recorder.thermalSchedule.animationsStatic
                    )
                    .frame(minWidth: 48, minHeight: 48)
            }
            .buttonStyle(.glass)
            .buttonBorderShape(.circle)
            .controlSize(.extraLarge)
            .glassEffectID("leadingAction", in: bottomGlassNamespace)
            .accessibilityLabel(recorder.isRecordingVoiceNote ? "Stop note" : "Add voice note")
            .accessibilityInputLabels(["Stop note", "Note", "Voice note"])
            .accessibilityShowsLargeContentViewer()
        }
    }

    private var finishButton: some View {
        Button {
            _ = recorder.evaluateWalkReview()
            showReviewWalkSheet = true
        } label: {
            Label("Finish", systemImage: "checkmark")
                .font(.headline)
        }
        .buttonStyle(.glassProminent)
        .controlSize(.extraLarge)
        .glassEffectID("finishButton", in: bottomGlassNamespace)
        .accessibilityLabel("Finish")
        .accessibilityInputLabels(["Finish", "Finish walk"])
        .accessibilityShowsLargeContentViewer()
    }
}

// MARK: - 4. Marker Progress Capsule & Tilt Gauge (`DESIGN-SPEC-NATIVE.md` §3a ③, ⑥b, §5, §6)

private struct HideStatusBarToolbarModifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 27.0, *) {
            content.toolbarVisibility(.hidden, for: .statusBar)
        } else {
            content
        }
    }
}

private struct MarkerProgressCapsuleView: View {
    let lockedMarkerIDs: Set<Int>
    let seenMarkerIDs: Set<Int>
    let phase: GuidedWalkPhase
    let isFinishAvailable: Bool
    let isAccessibilitySize: Bool
    let reduceMotion: Bool

    var body: some View {
        Group {
            if isAccessibilitySize {
                Text(MarkerProgressAccessibility.accessibilitySizeText(
                    lockedMarkerIDs: lockedMarkerIDs,
                    isFinishAvailable: isFinishAvailable
                ))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.primary)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
            } else if isFinishAvailable {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("All markers")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
            } else {
                HStack(spacing: 10) {
                    ForEach(0..<4, id: \.self) { mid in
                        markerSymbolView(for: mid)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
            }
        }
        .glassEffect(.regular, in: Capsule())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Markers")
        .accessibilityValue(
            MarkerProgressAccessibility.voiceOverValue(
                lockedMarkerIDs: lockedMarkerIDs,
                isFinishAvailable: isFinishAvailable
            )
        )
    }

    @ViewBuilder
    private func markerSymbolView(for mid: Int) -> some View {
        let isDone = lockedMarkerIDs.contains(mid)
        let isSeen = seenMarkerIDs.contains(mid)
        let isWaitingForReturn = (mid == 0 && phase == .closingMarker0 && !isFinishAvailable)
        let symbolName = (isDone || isWaitingForReturn) ? "\(mid).circle.fill" : "\(mid).circle"

        HStack(spacing: 2) {
            Image(systemName: symbolName)
                .font(.title3.weight(.semibold))
                .foregroundStyle((isDone || isSeen || isWaitingForReturn) ? .primary : .secondary)
                .contentTransition(.symbolEffect(.replace))
                .symbolEffect(.breathe, isActive: isWaitingForReturn && !reduceMotion)
            if isWaitingForReturn && reduceMotion {
                // DESIGN-SPEC-NATIVE.md §5 line 512: under Reduce Motion, show static arrow.uturn.backward badge
                Image(systemName: "arrow.uturn.backward")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.primary)
            }
        }
    }
}

private struct TiltGaugeOverlayView: View {
    let pitchDeg: Double

    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height
            // Target band: -30° ± 12° (-42° ... -18°) mapped onto [-70° ... +20°]
            func yForPitch(_ deg: Double) -> CGFloat {
                let clamped = min(20.0, max(-70.0, deg))
                let norm = (20.0 - clamped) / 90.0
                return CGFloat(norm) * h
            }
            let topBandY = yForPitch(-18.0)
            let botBandY = yForPitch(-42.0)
            let bandRect = CGRect(x: 0, y: topBandY, width: w, height: max(4, botBandY - topBandY))
            context.fill(Path(roundedRect: bandRect, cornerRadius: w * 0.5), with: .color(.green.opacity(0.45)))

            let dotY = yForPitch(pitchDeg)
            let inBand = (pitchDeg >= -42.0 && pitchDeg <= -18.0)
            let dotRect = CGRect(x: 0, y: dotY - w * 0.5, width: w, height: w)
            context.fill(Path(ellipseIn: dotRect), with: .color(inBand ? .white : .orange))
        }
        .frame(width: 6, height: 120)
        .background(.regularMaterial)
        .clipShape(Capsule())
    }
}

private struct MarkerBeaconsAndEdgeArrowsOverlayView: View {
    let projections: [MarkerBeaconProjection]

    var body: some View {
        GeometryReader { proxy in
            ForEach(projections) { proj in
                if proj.isOnScreen {
                    // ④ In-view marker beacon (`0.40 m` above seen marker): `n.circle` when waiting, `n.circle.fill` when done
                    Image(systemName: proj.isDone ? "\(proj.markerID).circle.fill" : "\(proj.markerID).circle")
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(proj.isDone ? .green : .white)
                        .padding(6)
                        .background(.ultraThinMaterial, in: Circle())
                        .position(
                            x: proj.normalizedPortraitXY.x * proxy.size.width,
                            y: proj.normalizedPortraitXY.y * proxy.size.height
                        )
                } else if let chevron = proj.edgeChevronSymbol {
                    // ⑤ Off-screen edge arrow: `chevron.left` / `chevron.right` + marker digit pinned to nearest screen edge
                    HStack(spacing: 4) {
                        if chevron == "chevron.left" {
                            Image(systemName: chevron)
                                .font(.caption.weight(.bold))
                            Text("\(proj.markerID)")
                                .font(.subheadline.monospacedDigit().weight(.bold))
                        } else {
                            Text("\(proj.markerID)")
                                .font(.subheadline.monospacedDigit().weight(.bold))
                            Image(systemName: chevron)
                                .font(.caption.weight(.bold))
                        }
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.regularMaterial, in: Capsule())
                    .position(
                        x: proj.normalizedPortraitXY.x * proxy.size.width,
                        y: proj.normalizedPortraitXY.y * proxy.size.height
                    )
                }
            }
        }
    }
}

// MARK: - 5. Live Review Walk Sheet (`YS-72`, `YS-80`, `YS-81`, `YS-84`, `DESIGN-SPEC-NATIVE.md` §3e, §4e)

private struct ReviewWalkSheetView: View {
    @ObservedObject var recorder: CaptureSessionRecorder
    @Binding var isPresented: Bool
    let onCompletedSave: () -> Void

    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showSaveAnywayConfirmation = false

    var body: some View {
        NavigationStack {
            Group {
                if recorder.phase == .sealing {
                    savingWalkProgressView
                } else {
                    reviewContentView
                }
            }
            .navigationTitle(recorder.phase == .sealing ? "Saving your walk" : "Review Walk")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            _ = recorder.evaluateWalkReview()
        }
        .confirmationDialog(
            "Save anyway?",
            isPresented: $showSaveAnywayConfirmation,
            titleVisibility: .visible
        ) {
            Button("Save Anyway", role: .destructive) {
                recorder.stopAndSealCapture()
            }
            Button("Head Back", role: .cancel) {
                recorder.performWalkReviewAction(.returnToMarker0)
                isPresented = false
            }
        } message: {
            Text("This walk probably won't be accepted.")
        }
    }

    private var savingWalkProgressView: some View {
        VStack(spacing: 18) {
            Spacer()
            ProgressView(value: recorder.sealingProgress, total: 1.0) {
                Text("Saving your walk")
                    .font(.headline)
            } currentValueLabel: {
                Text("\(Int((recorder.sealingProgress * 100.0).rounded()))%")
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .progressViewStyle(.linear)
            .padding(.horizontal, 32)
            Spacer()
        }
        .padding(24)
    }

    private var reviewContentView: some View {
        let report = recorder.latestWalkReviewReport
        return ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Content-layer yard map on `.regularMaterial` (NOT Liquid Glass per P2)
                GroundCoverageMiniMapView(
                    cells: recorder.coveredGroundCells,
                    accessibilitySummary: report.mapAccessibilitySummary
                )
                .frame(height: 150)

                // Verdict headline + symbol
                HStack(spacing: 10) {
                    Image(systemName: report.symbolName)
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(verdictColor(for: report.verdictKind))
                        .symbolEffect(.wiggle, value: report.verdictKind == .almostThere && !reduceMotion)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(report.headline)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(.primary)
                        Text(report.subheadline)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                // <= 3 review items, worst first (`YS-72`, `DESIGN-SPEC-NATIVE.md` §4e)
                if !report.items.isEmpty {
                    VStack(spacing: 10) {
                        ForEach(report.items) { item in
                            HStack(spacing: 12) {
                                Image(systemName: item.symbolName)
                                    .foregroundStyle(item.isWarning ? .orange : .secondary)
                                Text(item.sentence)
                                    .font(.subheadline)
                                    .foregroundStyle(.primary)
                                Spacer()
                                if differentiateWithoutColor {
                                    Text(item.differentiateWithoutColorBadge)
                                        .font(.footnote.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                }
                                if let btnTitle = item.buttonTitle, let action = item.action {
                                    Button(btnTitle) {
                                        recorder.performWalkReviewAction(action)
                                        isPresented = false
                                    }
                                    .buttonStyle(.glassProminent)
                                    .controlSize(.small)
                                }
                            }
                            .padding(12)
                            .background(.regularMaterial, in: ConcentricRectangle())
                            .accessibilityElement(children: .combine)
                        }
                    }
                }
            }
            .padding(20)
        }
        .safeAreaInset(edge: .bottom) {
            VStack(spacing: 10) {
                Button {
                    if report.requiresSaveAnywayConfirmation {
                        showSaveAnywayConfirmation = true
                    } else {
                        recorder.stopAndSealCapture()
                    }
                } label: {
                    Text("Save Capture")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .controlSize(.large)

                Button {
                    isPresented = false
                } label: {
                    Text("Keep Walking")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glass)
                .controlSize(.regular)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
    }

    private func verdictColor(for kind: WalkReviewVerdictKind) -> Color {
        switch kind {
        case .looksGood: return .green
        case .almostThere: return .orange
        case .probablyWontBeAccepted: return .red
        }
    }
}

// MARK: - 6. Capture Detail View (`YS-80`, `YS-81`, `DESIGN-SPEC-NATIVE.md` §3f)

public struct CaptureDetailView: View {
    public let record: SavedCaptureRecord
    @Environment(\.accessibilityDifferentiateWithoutColor) private var differentiateWithoutColor
    @State private var showTechnicalDetails = false

    public init(record: SavedCaptureRecord) {
        self.record = record
    }

    public var body: some View {
        List {
            Section {
                GroundCoverageMiniMapView(
                    cells: record.coveredGroundCells,
                    accessibilitySummary: "Map of your walk. \(record.detailSummary.markersValue)."
                )
                .frame(height: 160)
                .listRowInsets(EdgeInsets())

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Image(systemName: record.detailSummary.verdictSymbol)
                            .foregroundStyle(record.detailSummary.isVerdictPassing ? .green : .orange)
                        Text(record.detailSummary.verdictTitle)
                            .font(.title3.weight(.semibold))
                    }
                    Text(record.detailSummary.accuracySentence)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .accessibilityElement(children: .combine)
            }

            Section("Captured") {
                detailRow(
                    title: "Markers",
                    value: record.detailSummary.markersValue,
                    passed: record.detailSummary.markersPassed
                )
                detailRow(
                    title: "Back at the start",
                    value: record.detailSummary.backAtStartValue,
                    passed: record.detailSummary.backAtStartPassed
                )
                detailRow(
                    title: "Looked up",
                    value: record.detailSummary.lookedUpValue,
                    passed: record.detailSummary.lookedUpPassed
                )
                detailRow(
                    title: "Tape measurement",
                    value: record.detailSummary.tapeMeasurementValue,
                    passed: record.detailSummary.tapeMeasurementPassed
                )
            }

            Section {
                DisclosureGroup("Details", isExpanded: $showTechnicalDetails) {
                    LabeledContent("Grade", value: record.detailSummary.gradeLetter)
                    LabeledContent("Duration", value: "\(record.durationSeconds) s")
                    LabeledContent("Frames", value: "\(record.savedFrameCount)")
                    LabeledContent("Coverage", value: String(format: "%.1f m²", record.coveredGroundAreaM2))
                    LabeledContent("Format", value: "yardstick-capture/1")
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle(record.title)
        .toolbar {
            if let zipURL = record.archiveZipURL {
                ToolbarItem(placement: .topBarTrailing) {
                    ShareLink(item: zipURL) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Share Capture")
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if let zipURL = record.archiveZipURL {
                ShareLink(item: zipURL) {
                    Label("Share Capture", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
        }
    }

    private func detailRow(title: String, value: String, passed: Bool) -> some View {
        LabeledContent(title) {
            HStack(spacing: 6) {
                Text(value)
                    .foregroundStyle(passed ? .primary : .secondary)
                if differentiateWithoutColor {
                    Text(passed ? "Passed" : "Check")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - 7. Settings Sheet & Developer Live Diagnostics (`YS-70`, `YS-75`, `YS-80`, `DESIGN-SPEC-NATIVE.md` §3c, §3g)

private struct SettingsSheetView: View {
    @ObservedObject var recorder: CaptureSessionRecorder
    @Binding var isPresented: Bool
    let onOpenOnboarding: () -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Markers") {
                    LabeledContent("Marker Size", value: String(format: "%.1f cm", (recorder.markerSizesMM[0] ?? 180.0) / 10.0))

                    Button("How to Lay Markers") {
                        onOpenOnboarding()
                    }
                }

                Section {
                    Toggle("Voice Coach", isOn: $recorder.isVoiceCoachEnabled)
                    Toggle("Haptics", isOn: $recorder.isHapticsEnabled)
                } header: {
                    Text("Coaching")
                } footer: {
                    Text("Speaks each step aloud.")
                }

                Section("About") {
                    LabeledContent("Version", value: "1.0")
                    LabeledContent("Capture format", value: "yardstick-capture/1")
                    NavigationLink("Open-Source Licences") {
                        ScrollView {
                            Text(ArUcoDetector.openCVApacheAttributionNotice)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .padding()
                        }
                        .navigationTitle("Licences")
                    }
                }

                Section("Developer") {
                    Toggle("Live Diagnostics", isOn: $recorder.showDeveloperDiagnosticsOverlay)
                    NavigationLink("Experiments") {
                        Form {
                            Section {
                                Toggle(
                                    "Exposure Cap During Holds (≤ 1/250 s)",
                                    isOn: $recorder.experimentFlags.exposureCapDuringHolds
                                )
                                Toggle(
                                    "Hi-Res Still at Lock",
                                    isOn: $recorder.experimentFlags.hiResStillAtLock
                                )
                                Toggle(
                                    "Hardware Button as Note / Finish",
                                    isOn: $recorder.experimentFlags.hardwareButtonControl
                                )
                                Toggle(
                                    "ARAnchor on Locks",
                                    isOn: $recorder.experimentFlags.arAnchorOnLocks
                                )
                                Toggle(
                                    "Pause & Resume (Padded Audio)",
                                    isOn: $recorder.experimentFlags.pauseAndResumeWithPaddedAudio
                                )
                            } footer: {
                                Text("Each experiment is off by default and logged in guidance.csv.")
                            }
                        }
                        .navigationTitle("Experiments")
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        isPresented = false
                    }
                }
            }
        }
    }
}

private struct DeveloperDiagnosticsOverlayView: View {
    @ObservedObject var recorder: CaptureSessionRecorder

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 12) {
                Text("Frames: \(recorder.savedFrameCount)")
                Text("Drops: \(recorder.droppedFrameCount)")
                Text("Markers: \(recorder.lockedMarkerIDs.count)/4")
                Text(String(format: "Area: %.1f m²", recorder.coveredGroundAreaM2))
            }
            HStack(spacing: 12) {
                Text(String(format: "Pitch: %.1f°", recorder.pitchDeg))
                Text(String(format: "Roll: %.1f°", recorder.rollDeg))
                Text("Track: \(recorder.trackingStateText)")
                Text(String(format: "Blur: %.2f px", recorder.latestPredictedBlurPx))
            }
        }
        .font(.footnote.monospacedDigit())
        .foregroundStyle(.primary)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: ConcentricRectangle())
        .accessibilityHidden(true)
    }
}

// MARK: - 8. 3-Segment Marker Lock Ring & Top-Down Ground Coverage Map (`YS-74`, `YS-78`, `YS-81`, `YS-85`)

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

/// Top-down `0.25 m` ground coverage grid view on `.regularMaterial` (`YS-78`, `YS-81`).
public struct GroundCoverageMiniMapView: View {
    public let cells: Set<GroundGridCell>
    public let accessibilitySummary: String?

    public init(cells: Set<GroundGridCell>, accessibilitySummary: String? = nil) {
        self.cells = cells
        self.accessibilitySummary = accessibilitySummary
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
        .clipShape(ConcentricRectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityHidden(accessibilitySummary == nil)
        .accessibilityLabel(accessibilitySummary ?? "Ground coverage map")
    }
}
