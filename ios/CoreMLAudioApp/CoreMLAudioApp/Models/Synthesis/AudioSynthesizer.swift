import CoreML
import Accelerate

/// CoreML モデル 3 つ (Encoder, Decoder, HiFi-GAN) を使って音声合成を行う
final class AudioSynthesizer {

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var hifigan: MLModel?
    private var loadedPrecision: ModelPrecision?
    private var loadedComputeUnits: MLComputeUnits?
    private var loadedShapeMode: ShapeModeOption?
    private var loadedEncoderName: String?
    private var loadedDecoderName: String?
    private var loadedHifiganName: String?
    private var loadedHifiganShapeLabel: String?
    private var loadedHifiganPolicy: VocoderInputPolicy?
    private var loadedTransformerPolicy: TransformerInputPolicy?

    /// 指定精度・計算デバイス・shape mode で CoreML モデルをロードする
    /// （同じ設定でロード済みならスキップ）
    ///
    /// - Parameter shapeMode: HiFi-GAN の入力 shape バリアント。本番は `.fixed262` を推奨。
    ///                        shape 付きモデルを優先し、`range1` は legacy 命名にも対応する。
    func loadModels(precision: ModelPrecision, computeUnits: MLComputeUnits, shapeMode: ShapeModeOption) throws {
        if loadedPrecision == precision && loadedComputeUnits == computeUnits
            && loadedShapeMode == shapeMode
            && encoder != nil && decoder != nil && hifigan != nil {
            return
        }

        let encoderName = shapeMode.transformerEncoderResourceName(for: precision)
        let decoderName = shapeMode.transformerDecoderResourceName(for: precision)
        guard let hifiganName = shapeMode.hifiganResourceName(for: precision) else {
            throw SynthesisError.modelNotFound(
                precision: "\(precision.rawValue) + \(shapeMode.displayName)"
            )
        }
        let hifiganShapeLabel = shapeMode.resolvedShapeLabel(for: hifiganName, precision: precision)
        let hifiganPolicy = shapeMode.inputPolicy
        let transformerPolicy = shapeMode.transformerInputPolicy

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
        loadedShapeMode = shapeMode
        loadedEncoderName = encoderName
        loadedDecoderName = decoderName
        loadedHifiganName = hifiganName
        loadedHifiganShapeLabel = hifiganShapeLabel
        loadedHifiganPolicy = hifiganPolicy
        loadedTransformerPolicy = transformerPolicy
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

        // 調査用 snapshot (本番無効。CMLA_DEBUG_SNAPSHOT=1 のときだけ作られる)
        let debug = DebugRunSnapshot.makeIfEnabled(
            precision: precision.rawValue,
            computeUnit: computeUnit.rawValue
        )
        debug?.writeContext(
            precision: precision.rawValue,
            computeUnit: computeUnit.rawValue,
            shapeMode: loadedShapeMode?.rawValue ?? "?",
            inputURL: inputURL
        )
        debug?.writeFloat2D(
            name: "mel_normalized",
            data: melData,
            rows: frameCount,
            cols: nMels
        )

        // 2. Encoder
        await MainActor.run { onProgress("Encoder 実行中...", 0.05) }
        let transformerPolicy = loadedTransformerPolicy ?? .dynamic(maxT: 1000)
        // policy 上限を超える入力はここで先にクロップしておく。Encoder/Decoder それぞれが
        // 内部で同じ min(frameCount, maxT/targetT) を持つが、外で先に確定しておかないと
        // 後段（postnetOut のループや TimingInfo）が元の frameCount を読みに行って範囲外アクセスする。
        let effectiveFrameCount: Int
        switch transformerPolicy {
        case .fixed(let targetT):
            effectiveFrameCount = min(frameCount, targetT)
        case .dynamic(let maxT):
            effectiveFrameCount = min(frameCount, maxT)
        }
        if effectiveFrameCount < frameCount {
            print("[AudioSynthesizer] WARN: frameCount=\(frameCount) > policy max, cropped to \(effectiveFrameCount)")
        }
        let encoderRunner = EncoderRunner(model: encoder, policy: transformerPolicy)
        let (memory, encoderMs) = try await encoderRunner.run(mel: melData, frameCount: effectiveFrameCount, nMels: nMels)
        let encoderStats = ArrayStats.compute(from: memory)

        // 調査用: Encoder 出力 (memory) を [T_src × hidden] で書き出す。
        // shape は MLMultiArray.shape を見て決める。
        if let debug = debug {
            let shape = memory.shape.map { $0.intValue }
            if shape.count == 3 {
                // [1, T, hidden] を 2D 化
                debug.writeMlArray(
                    name: "encoder_output",
                    array: memory,
                    outerDim: shape[1],
                    innerDim: shape[2]
                )
            } else {
                debug.appendSummaryLine("encoder_output: unexpected shape=\(shape) — skipped npy")
            }
            debug.appendSummaryLine(String(format: "encoder_stats: \(encoderStats.summary) predict_ms=%.3f", encoderMs))
        }

        // 3. Decoder (自己回帰ループ)
        await MainActor.run { onProgress("Decoder 実行中... (0/\(effectiveFrameCount))", 0.1) }
        let decoderRunner = DecoderRunner(model: decoder, policy: transformerPolicy)
        let (postnetOut, decoderStepStats, decoderTotalMs) = try await decoderRunner.run(
            memory: memory,
            frameCount: effectiveFrameCount,
            nMels: nMels,
            onStep: { completed in
                let progress = 0.1 + 0.8 * Double(completed) / Double(effectiveFrameCount)
                onProgress("Decoder 実行中... (\(completed)/\(effectiveFrameCount))", progress)
            },
            debug: debug
        )

        // 4. HiFi-GAN
        await MainActor.run { onProgress("HiFi-GAN 実行中...", 0.9) }
        let totalFrames = effectiveFrameCount

        // 入力統計は転置前の postnetOut で計算（min/max/mean/NaN/Inf は要素順に依存しないため転置後と同値）
        let hifiganInputStats = ArrayStats.compute(from: postnetOut)

        // shapeMode と policy はロード時に確定。ロード前なら fallback (.fixed(262)) を使う。
        let vocoderPolicy = loadedHifiganPolicy ?? .fixed(targetT: 262)
        // 調査用: Decoder postnet 出力 (HiFi-GAN への入力) を書き出す。
        // postnetOut shape: [1, T, nMels]
        if let debug = debug {
            let shape = postnetOut.shape.map { $0.intValue }
            if shape.count == 3 {
                debug.writeMlArray(
                    name: "postnet_output",
                    array: postnetOut,
                    outerDim: shape[1],
                    innerDim: shape[2]
                )
            } else {
                debug.appendSummaryLine("postnet_output: unexpected shape=\(shape) — skipped npy")
            }
            debug.appendSummaryLine("postnet_stats: \(hifiganInputStats.summary)")
        }

        let vocoderRunner = VocoderRunner(model: hifigan, policy: vocoderPolicy)
        let vocoderResult = try await vocoderRunner.run(postnetOut: postnetOut, totalFrames: totalFrames, nMels: nMels)
        var waveform = vocoderResult.waveform
        let hifiganMs = vocoderResult.predictMs
        let hifiganInputT = vocoderResult.inputT
        let hifiganOutputStats = ArrayStats.compute(from: waveform)

        // 調査用: HiFi-GAN 出力波形 (デエンファシス前) を書き出す。
        if let debug = debug {
            debug.writeFloat1D(name: "waveform_predeemph", data: waveform)
            debug.appendSummaryLine(String(format: "hifigan_predict_ms=%.3f inputT=%d", hifiganMs, hifiganInputT))
        }

        // 本番ロギング: 実機で「どのモデル × どの設定で生成したか」と
        // 「出力波形の数値プロファイル」を 1 行で残す。grep しやすいよう接頭辞を統一。
        // ArrayStats は rms を持っていないため、ここで waveform をもう一度走査して計算する。
        var sumSq: Double = 0
        for v in waveform where v.isFinite { sumSq += Double(v) * Double(v) }
        let rms = waveform.isEmpty ? Float.nan : Float((sumSq / Double(waveform.count)).squareRoot())
        let nanFlag = hifiganOutputStats.hasNaN ? "yes" : "no"
        let infFlag = hifiganOutputStats.hasInf ? "yes" : "no"
        print(String(
            format: "[HiFiGAN production] model=%@ precision=%@ shape_mode=%@ computeUnits=%@ "
                + "input_shape=[1,%d,%d] output_count=%d "
                + "min=%.6f max=%.6f mean=%.6e rms=%.6f nan=%@ inf=%@ predict_ms=%.3f",
            loadedHifiganName ?? "?",
            precision.rawValue,
            loadedHifiganShapeLabel ?? "?",
            computeUnit.rawValue,
            nMels, hifiganInputT,
            waveform.count,
            hifiganOutputStats.min, hifiganOutputStats.max, hifiganOutputStats.mean, rms,
            nanFlag, infFlag, hifiganMs
        ))

        // 5. デエンファシスフィルタ: y[n] = x[n] + coeff * y[n-1]
        await MainActor.run { onProgress("後処理中...", 0.95) }
        waveform = AudioFeatureExtractor.applyDeemphasis(waveform)
        let waveformAfterDeemphasis = ArrayStats.compute(from: waveform)

        // 調査用: deemph 後の最終波形 (wav 書き出し直前の値)
        if let debug = debug {
            debug.writeFloat1D(name: "waveform_postdeemph", data: waveform)
        }

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
        let modelSizeBytes = Self.computeModelSizeBytes(
            precision: precision,
            encoderName: loadedEncoderName,
            decoderName: loadedDecoderName,
            hifiganName: loadedHifiganName
        )
        let timing = TimingInfo(
            encoderMs: encoderMs,
            decoderTotalMs: decoderTotalMs,
            decoderStepCount: effectiveFrameCount,
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

    /// 指定 precision の Encoder / Decoder と、実際にロードした HiFi-GAN の .mlmodelc 合計バイト数を返す
    private static func computeModelSizeBytes(
        precision: ModelPrecision,
        encoderName: String?,
        decoderName: String?,
        hifiganName: String?
    ) -> Int64 {
        let names = [
            encoderName ?? "Transformer_Encoder_\(precision.suffix)",
            decoderName ?? "Transformer_Decoder_\(precision.suffix)",
            hifiganName ?? "HiFiGAN_Generator_\(precision.suffix)_fixed262",
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
