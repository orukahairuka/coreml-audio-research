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

    var canPlay: Bool { outputWaveform != nil && !isProcessing }
    var isPlaying: Bool { audioPlayer.isPlaying }

    // MARK: - Private

    private let synthesizer = AudioSynthesizer()
    private let audioPlayer = AudioPlayer()
    private var outputWaveform: [Float]?

    // MARK: - Actions

    func runSynthesis() async {
        errorMessage = nil
        outputWaveform = nil
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

            let waveform = try await synthesizer.synthesize(inputURL: inputURL) {
                [weak self] statusText, progressValue in
                self?.status = statusText
                self?.progress = progressValue
            }

            outputWaveform = waveform
            status = "合成完了"
            progress = 1.0
            playOutput()
        } catch {
            errorMessage = error.localizedDescription
            status = "エラー"
        }
    }

    func playOutput() {
        guard let waveform = outputWaveform else { return }
        do {
            try audioPlayer.play(waveform: waveform, sampleRate: AudioFeatureExtractor.sampleRate)
        } catch {
            errorMessage = "再生エラー: \(error.localizedDescription)"
        }
    }

    func stopPlayback() {
        audioPlayer.stop()
    }
}
