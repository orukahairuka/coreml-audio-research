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
    /// HiFi-GAN 入力 shape バリアント。本番デフォルトは `.fixed262` (実機テストで全 computeUnits 安定)。
    /// 他の RangeDim 系は研究用に残してあり、UI から切り替えられる。
    var selectedShapeMode: ShapeModeOption = .fixed262

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
    private let stabilityTester = VocoderStabilityTester()
    private(set) var synthesisResult: SynthesisResult?
    private var playbackContinuation: CheckedContinuation<Void, Never>?

    // 安定性テストのサマリ（テスト直後に表示する）
    private(set) var stabilityCsvURL: URL?
    private(set) var stabilitySummary: String?

    // 差分テストのサマリ
    private(set) var diffCsvURL: URL?
    private(set) var diffSummary: String?

    // MARK: - Init

    init() {
        audioPlayer.onPlaybackFinished = { [weak self] in
            Task { @MainActor in
                guard let self = self else { return }
                self.isPlaying = false
                if let cont = self.playbackContinuation {
                    self.playbackContinuation = nil
                    cont.resume()
                }
            }
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

        // 未生成バリアントはここで弾く (ロード失敗より早く UI に伝える)
        if !selectedShapeMode.isAvailable(for: selectedPrecision) {
            errorMessage = "\(selectedPrecision.rawValue) + \(selectedShapeMode.displayName) のモデルは生成されていません"
            status = "エラー"
            return
        }

        do {
            try synthesizer.loadModels(
                precision: selectedPrecision,
                computeUnits: selectedComputeUnit.mlComputeUnits,
                shapeMode: selectedShapeMode
            )
            status = "モデルロード完了 (\(selectedPrecision.rawValue), \(selectedShapeMode.displayName), \(selectedComputeUnit.displayName))"

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
            saveMelArtifacts(result: result)
            saveTimingArtifact(result: result)
            status = "合成完了"
            progress = 1.0
            await playOutputAndAwaitCompletion()
        } catch {
            errorMessage = error.localizedDescription
            status = "エラー"
        }
    }

    private func saveMelArtifacts(result: SynthesisResult) {
        // 入力メル: 全組み合わせで同じなので固定名で上書き
        do {
            try MelArtifactWriter.write(
                melData: result.inputMelSpectrogram,
                frameCount: result.inputFrameCount,
                nMels: result.nMels,
                baseName: "input_mel"
            )
        } catch {
            print("[MelArtifactWriter] 入力メル保存失敗: \(error.localizedDescription)")
        }

        // 出力メル: 精度・デバイスでファイル名を分ける
        let baseName = "output_mel_\(result.precision.rawValue)_\(result.computeUnit.rawValue)"
        do {
            try MelArtifactWriter.write(
                melData: result.outputMelSpectrogram,
                frameCount: result.outputFrameCount,
                nMels: result.nMels,
                baseName: baseName
            )
        } catch {
            print("[MelArtifactWriter] 出力メル保存失敗 (\(baseName)): \(error.localizedDescription)")
        }
    }

    private func saveTimingArtifact(result: SynthesisResult) {
        do {
            try TimingJsonWriter.write(
                timing: result.timing,
                precision: result.precision,
                computeUnit: result.computeUnit
            )
        } catch {
            print("[TimingJsonWriter] 保存失敗: \(error.localizedDescription)")
        }
    }

    /// Float16 fixed262 を `.cpuAndGPU` と `.all` で実行して出力波形の差分を比較する。
    /// 結果は `Documents/Result/stability/vocoder_diff_f16_fixed262_<timestamp>.csv` と
    /// 同名 prefix の `.npy` 3 本 (cpuAndGPU / all / diff) に保存される。
    func runVocoderDiffTest() async {
        errorMessage = nil
        diffCsvURL = nil
        diffSummary = nil
        isProcessing = true
        progress = 0
        status = "差分テスト開始..."
        defer { isProcessing = false }

        let csvURL = await stabilityTester.runFloat16Fixed262DiffTest(onProgress: { [weak self] message in
            self?.status = message
        })
        diffCsvURL = csvURL
        diffSummary = status   // status の最終メッセージ ("差分テスト完了 — diff: ...") を表示用に保持
        progress = 1.0
    }

    /// HiFi-GAN の shape バリアント × 3 computeUnits を網羅的に試して結果を CSV に書き出す。
    /// 結果は `Documents/Result/stability/vocoder_stability_<timestamp>.csv` に保存される。
    func runStabilityTest() async {
        errorMessage = nil
        stabilityCsvURL = nil
        stabilitySummary = nil
        isProcessing = true
        progress = 0
        status = "安定性テスト開始..."
        defer { isProcessing = false }

        let (results, csvURL) = await stabilityTester.runAll(onProgress: { [weak self] message in
            self?.status = message
        })

        stabilityCsvURL = csvURL

        // サマリ: status 別件数 + 飽和(>=0.99)した組み合わせ数
        let total = results.count
        let success = results.filter { $0.status == "success" }.count
        let saturated = results.filter { ($0.stats?.ratioOver099 ?? 0) > 0.5 || ($0.stats?.ratioUnderMinus099 ?? 0) > 0.5 }.count
        let nans = results.filter { ($0.stats?.nanCount ?? 0) > 0 }.count
        let failed = total - success
        stabilitySummary = "実行: \(total) / 成功: \(success) / 失敗: \(failed) / 飽和: \(saturated) / NaN: \(nans)"
        status = "安定性テスト完了"
        progress = 1.0
    }

    func playOutput() {
        guard let result = synthesisResult else { return }
        let baseName = "\(result.precision.rawValue)_\(result.computeUnit.rawValue)"
        do {
            try audioPlayer.play(
                waveform: result.outputWaveform,
                sampleRate: AudioFeatureExtractor.sampleRate,
                baseName: baseName
            )
            isPlaying = true
        } catch {
            errorMessage = "再生エラー: \(error.localizedDescription)"
        }
    }

    private func playOutputAndAwaitCompletion() async {
        guard let result = synthesisResult else { return }
        let baseName = "\(result.precision.rawValue)_\(result.computeUnit.rawValue)"
        do {
            try audioPlayer.play(
                waveform: result.outputWaveform,
                sampleRate: AudioFeatureExtractor.sampleRate,
                baseName: baseName
            )
            isPlaying = true
        } catch {
            errorMessage = "再生エラー: \(error.localizedDescription)"
            return
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            playbackContinuation = cont
        }
    }

}
