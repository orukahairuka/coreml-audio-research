import CoreML
import Accelerate

/// CoreML モデル 3 つ (Encoder, Decoder, HiFi-GAN) を使って音声合成を行う
final class AudioSynthesizer {

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var hifigan: MLModel?
    private var loadedPrecision: ModelPrecision?
    private var loadedComputeUnits: MLComputeUnits?

    /// 指定精度・計算デバイスで CoreML モデルをロードする（同じ設定でロード済みならスキップ）
    func loadModels(precision: ModelPrecision, computeUnits: MLComputeUnits) throws {
        if loadedPrecision == precision && loadedComputeUnits == computeUnits
            && encoder != nil && decoder != nil && hifigan != nil {
            return
        }

        let encoderName = "Transformer_Encoder_\(precision.suffix)"
        let decoderName = "Transformer_Decoder_\(precision.suffix)"
        let hifiganName = "HiFiGAN_Generator_\(precision.suffix)"

        guard let encoderURL = Bundle.main.url(forResource: encoderName, withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: decoderName, withExtension: "mlmodelc"),
              let hifiganURL = Bundle.main.url(forResource: hifiganName, withExtension: "mlmodelc") else {
            throw SynthesisError.modelNotFound(precision: precision.rawValue)
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let enc = try MLModel(contentsOf: encoderURL, configuration: config)
        let dec = try MLModel(contentsOf: decoderURL, configuration: config)
        let hifi = try MLModel(contentsOf: hifiganURL, configuration: config)

        encoder = enc
        decoder = dec
        hifigan = hifi
        loadedPrecision = precision
        loadedComputeUnits = computeUnits
    }

    /// 入力音声 URL から合成を実行し、SynthesisResult を返す
    /// - Parameter onProgress: (statusMessage, progressFraction) を各ステップで呼ぶ
    func synthesize(
        inputURL: URL,
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        onProgress: @MainActor (String, Double) -> Void
    ) async throws -> SynthesisResult {
        guard let encoder, let decoder, let hifigan else {
            throw SynthesisError.modelNotLoaded
        }

        // 1. メルスペクトログラム抽出
        await MainActor.run { onProgress("特徴量抽出中...", 0.0) }
        let inputWaveform = try AudioFeatureExtractor.loadAudio(from: inputURL)
        let melResult = try AudioFeatureExtractor.extractMelSpectrogram(from: inputURL)
        let melData = melResult.mel
        let frameCount = melResult.frameCount
        let inputDisplayMel = try AudioFeatureExtractor.melSpectrogramForDisplay(from: inputURL)
        let nMels = AudioFeatureExtractor.nMels

        // 2. Encoder
        await MainActor.run { onProgress("Encoder 実行中...", 0.05) }
        let encoderRunner = EncoderRunner(model: encoder)
        let (memory, encoderMs) = try await encoderRunner.run(mel: melData, frameCount: frameCount, nMels: nMels)
        let encoderStats = ArrayStats.compute(from: memory)

        // 3. Decoder (自己回帰ループ)
        await MainActor.run { onProgress("Decoder 実行中... (0/\(frameCount))", 0.1) }
        let decoderRunner = DecoderRunner(model: decoder)
        let (postnetOut, decoderStepStats, decoderTotalMs) = try await decoderRunner.run(
            memory: memory,
            frameCount: frameCount,
            nMels: nMels,
            onStep: { completed in
                let progress = 0.1 + 0.8 * Double(completed) / Double(frameCount)
                onProgress("Decoder 実行中... (\(completed)/\(frameCount))", progress)
            }
        )

        // 4. HiFi-GAN
        await MainActor.run { onProgress("HiFi-GAN 実行中...", 0.9) }
        let totalFrames = frameCount

        // 入力統計は転置前の postnetOut で計算（min/max/mean/NaN/Inf は要素順に依存しないため転置後と同値）
        let hifiganInputStats = ArrayStats.compute(from: postnetOut)

        let vocoderRunner = VocoderRunner(model: hifigan)
        let vocoderResult = try await vocoderRunner.run(postnetOut: postnetOut, totalFrames: totalFrames, nMels: nMels)
        var waveform = vocoderResult.waveform
        let hifiganMs = vocoderResult.predictMs
        let hifiganOutputStats = ArrayStats.compute(from: waveform)

        // 5. デエンファシスフィルタ: y[n] = x[n] + coeff * y[n-1]
        await MainActor.run { onProgress("後処理中...", 0.95) }
        waveform = AudioFeatureExtractor.applyDeemphasis(waveform)
        let waveformAfterDeemphasis = ArrayStats.compute(from: waveform)

        // デバッグ情報をまとめる
        let debugInfo = PipelineDebugInfo(
            encoderOutput: encoderStats,
            decoderSteps: decoderStepStats,
            hifiganInput: hifiganInputStats,
            hifiganOutput: hifiganOutputStats,
            waveformAfterDeemphasis: waveformAfterDeemphasis
        )

        // 6. 出力メルスペクトログラム (postnet_out を可視化用 dB に変換)
        var outputMelNormalized = [Float](repeating: 0, count: totalFrames * nMels)
        for t in 0..<totalFrames {
            for m in 0..<nMels {
                outputMelNormalized[t * nMels + m] = postnetOut[[0, t as NSNumber, m as NSNumber]].floatValue
            }
        }
        let outputDisplayMel = AudioFeatureExtractor.denormalizeToDisplayDb(outputMelNormalized)

        let outputDurationMs = Double(waveform.count) / AudioFeatureExtractor.sampleRate * 1000.0
        let modelSizeBytes = Self.computeModelSizeBytes(precision: precision)
        let timing = TimingInfo(
            encoderMs: encoderMs,
            decoderTotalMs: decoderTotalMs,
            decoderStepCount: frameCount,
            hifiganMs: hifiganMs,
            outputDurationMs: outputDurationMs,
            modelSizeBytes: modelSizeBytes
        )

        return SynthesisResult(
            precision: precision,
            computeUnit: computeUnit,
            inputWaveform: inputWaveform,
            outputWaveform: waveform,
            inputMelSpectrogram: inputDisplayMel.mel,
            outputMelSpectrogram: outputDisplayMel,
            inputFrameCount: inputDisplayMel.frameCount,
            outputFrameCount: totalFrames,
            nMels: nMels,
            sampleRate: AudioFeatureExtractor.sampleRate,
            debugInfo: debugInfo,
            timing: timing
        )
    }

    /// 指定 precision の 3 モデル (Encoder + Decoder + HiFi-GAN) の .mlmodelc 合計バイト数を返す
    private static func computeModelSizeBytes(precision: ModelPrecision) -> Int64 {
        let names = [
            "Transformer_Encoder_\(precision.suffix)",
            "Transformer_Decoder_\(precision.suffix)",
            "HiFiGAN_Generator_\(precision.suffix)",
        ]
        var total: Int64 = 0
        for name in names {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { continue }
            total += directorySizeBytes(at: url)
        }
        return total
    }

    /// ディレクトリ配下の全ファイル合計サイズ (バイト)
    private static func directorySizeBytes(at url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]
        ) else { return 0 }
        var size: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true, let s = values?.fileSize {
                size += Int64(s)
            }
        }
        return size
    }

    enum SynthesisError: LocalizedError {
        case modelNotFound(precision: String)
        case modelNotLoaded
        case decoderFailed

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let precision):
                return "\(precision) の CoreML モデルが見つかりません。.mlpackage をプロジェクトに追加してください。"
            case .modelNotLoaded: return "モデルがロードされていません。"
            case .decoderFailed: return "Decoder の出力を取得できませんでした。"
            }
        }
    }
}
