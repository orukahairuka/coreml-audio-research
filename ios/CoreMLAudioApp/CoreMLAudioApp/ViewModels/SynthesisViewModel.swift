import SwiftUI

/// 合成・再生のオーケストレーションを担う ViewModel
@MainActor
@Observable
final class SynthesisViewModel {

    // MARK: - UI State

    var status: String = "待機中"
    var isProcessing: Bool = false
    var progress: Double = 0
    var errorMessage: String?

    var canPlay: Bool { synthesisResult != nil && !isProcessing }
    private(set) var isPlaying: Bool = false
    var hasResult: Bool { synthesisResult != nil }

    // MARK: - Private

    private let synthesizer = AudioSynthesizer()
    private let audioPlayer = AudioPlayer()
    private(set) var synthesisResult: SynthesisResult?

    // MARK: - Init

    init() {
        audioPlayer.onPlaybackFinished = { [weak self] in
            Task { @MainActor in self?.isPlaying = false }
        }
    }

    // MARK: - Actions

    func runSynthesis() async {
        errorMessage = nil
        synthesisResult = nil
        isProcessing = true
        progress = 0
        defer { isProcessing = false }

        do {
            try synthesizer.loadModels()
            status = "モデルロード完了"

            guard let inputURL = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
                errorMessage = "input_sample.wav がバンドルに見つかりません"
                return
            }

            let result = try await synthesizer.synthesize(
                inputURL: inputURL,
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
