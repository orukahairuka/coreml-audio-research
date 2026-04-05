import AVFoundation
import SwiftUI

/// 合成・再生・録音のオーケストレーションを担う ViewModel
@MainActor
@Observable
final class SynthesisViewModel {

    // MARK: - UI State

    var status: String = "待機中"
    var isProcessing: Bool = false
    var progress: Double = 0
    var errorMessage: String?
    var selectedPrecision: ModelPrecision = .float16
    var selectedComputeUnit: ComputeUnitOption = .cpuAndGPU

    var canPlay: Bool { synthesisResult != nil && !isProcessing }
    private(set) var isPlaying: Bool = false
    var hasResult: Bool { synthesisResult != nil }

    // MARK: - Audio Source

    var audioSource: AudioSource = .bundledSample
    var recordings: [URL] = []

    // MARK: - Recording State

    private(set) var isRecording: Bool = false
    private(set) var recordingTime: TimeInterval = 0
    var maxRecordingDuration: TimeInterval { AudioRecorder.maxDuration }

    // MARK: - Private

    private let synthesizer = AudioSynthesizer()
    private let audioPlayer = AudioPlayer()
    private let audioRecorder = AudioRecorder()
    private(set) var synthesisResult: SynthesisResult?

    // MARK: - Init

    init() {
        audioPlayer.onPlaybackFinished = { [weak self] in
            Task { @MainActor in self?.isPlaying = false }
        }
        audioRecorder.onRecordingFinished = { [weak self] url in
            Task { @MainActor in
                self?.isRecording = false
                self?.recordingTime = 0
                if let url {
                    self?.loadRecordings()
                    self?.audioSource = .recording(url)
                } else {
                    self?.errorMessage = "録音に失敗しました"
                }
            }
        }
        audioRecorder.onTimeUpdate = { [weak self] time in
            Task { @MainActor in self?.recordingTime = time }
        }
        loadRecordings()
    }

    // MARK: - Recording Actions

    func startRecording() {
        errorMessage = nil
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else {
                    self.errorMessage = "マイクの使用が許可されていません。設定アプリから許可してください。"
                    return
                }
                do {
                    try self.audioRecorder.startRecording()
                    self.isRecording = true
                    self.recordingTime = 0
                } catch {
                    self.errorMessage = "録音開始エラー: \(error.localizedDescription)"
                }
            }
        }
    }

    func stopRecording() {
        audioRecorder.stopRecording()
    }

    func loadRecordings() {
        recordings = AudioRecorder.recordings()
    }

    func deleteRecording(at url: URL) {
        try? AudioRecorder.deleteRecording(at: url)
        if case .recording(let selected) = audioSource, selected == url {
            audioSource = .bundledSample
        }
        loadRecordings()
    }

    // MARK: - Synthesis Actions

    func runSynthesis() async {
        errorMessage = nil
        synthesisResult = nil
        isProcessing = true
        progress = 0
        defer { isProcessing = false }

        do {
            try synthesizer.loadModels(
                precision: selectedPrecision,
                computeUnits: selectedComputeUnit.mlComputeUnits
            )
            status = "モデルロード完了 (\(selectedPrecision.rawValue), \(selectedComputeUnit.displayName))"

            let inputURL: URL
            switch audioSource {
            case .bundledSample:
                guard let url = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
                    errorMessage = "input_sample.wav がバンドルに見つかりません"
                    return
                }
                inputURL = url
            case .recording(let url):
                guard FileManager.default.fileExists(atPath: url.path) else {
                    errorMessage = "録音ファイルが見つかりません: \(url.lastPathComponent)"
                    return
                }
                inputURL = url
            }

            let result = try await synthesizer.synthesize(
                inputURL: inputURL,
                precision: selectedPrecision,
                computeUnit: selectedComputeUnit,
                onProgress: { [weak self] (statusText: String, progressValue: Double) in
                    self?.status = statusText
                    self?.progress = progressValue
                }
            )

            synthesisResult = result
            status = "合成完了"
            progress = 1.0
            playOutput()
        } catch {
            errorMessage = error.localizedDescription
            status = "エラー"
        }
    }

    func playOutput() {
        guard let waveform = synthesisResult?.outputWaveform else { return }
        do {
            try audioPlayer.play(waveform: waveform, sampleRate: AudioFeatureExtractor.sampleRate)
            isPlaying = true
        } catch {
            errorMessage = "再生エラー: \(error.localizedDescription)"
        }
    }

}
