import CoreML
import Accelerate

/// CoreML モデル 3 つ (Encoder, Decoder, HiFi-GAN) を使って音声合成を行う
final class AudioSynthesizer {

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var hifigan: MLModel?
    private var loadedPrecision: ModelPrecision?
    private var loadedComputeUnits: MLComputeUnits?

    // MARK: - Debug Helpers

    /// MLMultiArray の統計情報を計算する
    private func computeStats(of array: MLMultiArray) -> ArrayStats {
        let count = array.count
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        var sum: Float = 0
        var hasNaN = false
        var hasInf = false
        for i in 0..<count {
            let v = array[i].floatValue
            if v.isNaN { hasNaN = true }
            if v.isInfinite { hasInf = true }
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
            sum += v
        }
        return ArrayStats(
            min: minVal, max: maxVal, mean: sum / Float(count),
            hasNaN: hasNaN, hasInf: hasInf
        )
    }

    /// Float 配列の統計情報を計算する
    private func computeStats(of array: [Float]) -> ArrayStats {
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        var sum: Float = 0
        var hasNaN = false
        var hasInf = false
        for v in array {
            if v.isNaN { hasNaN = true }
            if v.isInfinite { hasInf = true }
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
            sum += v
        }
        return ArrayStats(
            min: minVal, max: maxVal, mean: sum / Float(array.count),
            hasNaN: hasNaN, hasInf: hasInf
        )
    }

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
        let melArray = try MLMultiArray(shape: [1, frameCount as NSNumber, nMels as NSNumber], dataType: .float32)
        for i in 0..<(frameCount * nMels) {
            melArray[i] = NSNumber(value: melData[i])
        }

        let posArray = try MLMultiArray(shape: [1, frameCount as NSNumber], dataType: .int32)
        for i in 0..<frameCount {
            posArray[i] = NSNumber(value: Int32(i + 1))
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "pos": MLFeatureValue(multiArray: posArray)
        ])
        let encoderOutput = try await encoder.prediction(from: encoderInput)
        guard let memoryFeature = encoderOutput.featureValue(for: encoderOutput.featureNames.first ?? ""),
              let memory = memoryFeature.multiArrayValue else {
            throw SynthesisError.decoderFailed
        }
        let encoderStats = computeStats(of: memory)

        // 3. Decoder (自己回帰ループ)
        await MainActor.run { onProgress("Decoder 実行中... (0/\(frameCount))", 0.1) }

        // 初期入力: ゼロベクトル [1, 1, 256]
        var decoderInputData = [Float](repeating: 0, count: nMels)
        var currentLength = 1
        var lastMelOut: MLMultiArray?
        var lastPostnetOut: MLMultiArray?
        var decoderStepStats = [DecoderStepStats]()

        for step in 0..<frameCount {
            // decoder_input: [1, currentLength, 256]
            let decInput = try MLMultiArray(shape: [1, currentLength as NSNumber, nMels as NSNumber], dataType: .float32)
            for i in 0..<(currentLength * nMels) {
                decInput[i] = NSNumber(value: decoderInputData[i])
            }

            // pos: [1, currentLength]
            let decPos = try MLMultiArray(shape: [1, currentLength as NSNumber], dataType: .int32)
            for i in 0..<currentLength {
                decPos[i] = NSNumber(value: Int32(i + 1))
            }

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "memory": MLFeatureValue(multiArray: memory),
                "decoder_input": MLFeatureValue(multiArray: decInput),
                "pos": MLFeatureValue(multiArray: decPos)
            ])
            let decoderOutput = try await decoder.prediction(from: decoderInput)

            // mel_out は変換時に明示命名済み。postnet 出力は残りのキーから取得
            let postnetKey = decoderOutput.featureNames.first(where: { $0 != "mel_out" }) ?? ""
            lastMelOut = decoderOutput.featureValue(for: "mel_out")?.multiArrayValue
            lastPostnetOut = decoderOutput.featureValue(for: postnetKey)?.multiArrayValue

            // 最後のフレームを取得して入力に追加
            guard let melOut = lastMelOut else { throw SynthesisError.decoderFailed }

            // デバッグ: 先頭・中間・末尾の5ステップずつ + NaN/Inf 検出時は全ステップ記録
            let melStats = computeStats(of: melOut)
            let postStats = lastPostnetOut.map { computeStats(of: $0) }
                ?? ArrayStats(min: 0, max: 0, mean: 0, hasNaN: false, hasInf: false)
            let shouldRecord = step < 5
                || step >= frameCount - 5
                || (frameCount > 10 && step >= frameCount / 2 - 2 && step <= frameCount / 2 + 2)
                || melStats.hasNaN || melStats.hasInf
                || postStats.hasNaN || postStats.hasInf
            if shouldRecord {
                decoderStepStats.append(DecoderStepStats(
                    step: step, melOut: melStats, postnetOut: postStats
                ))
            }

            for i in 0..<nMels {
                decoderInputData.append(melOut[[0, (currentLength - 1) as NSNumber, i as NSNumber]].floatValue)
            }
            currentLength += 1

            let progressValue = 0.1 + 0.8 * Double(step + 1) / Double(frameCount)
            await MainActor.run { onProgress("Decoder 実行中... (\(step + 1)/\(frameCount))", progressValue) }

            // UI 更新のために yield
            if step % 10 == 0 {
                await Task.yield()
            }
        }

        // 4. HiFi-GAN
        await MainActor.run { onProgress("HiFi-GAN 実行中...", 0.9) }
        guard let postnetOut = lastPostnetOut else { throw SynthesisError.decoderFailed }

        // postnet_out: [1, T, 256] → [1, 256, T] に転置
        let totalFrames = frameCount
        let vocoderInput = try MLMultiArray(shape: [1, nMels as NSNumber, totalFrames as NSNumber], dataType: .float32)
        for t in 0..<totalFrames {
            for m in 0..<nMels {
                vocoderInput[[0, m as NSNumber, t as NSNumber]] = postnetOut[[0, t as NSNumber, m as NSNumber]]
            }
        }

        let hifiganInputStats = computeStats(of: vocoderInput)

        let hifiganInputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: vocoderInput)
        ])
        let hifiganOutput = try await hifigan.prediction(from: hifiganInputProvider)
        guard let waveformFeature = hifiganOutput.featureValue(for: hifiganOutput.featureNames.first ?? ""),
              let waveformArray = waveformFeature.multiArrayValue else {
            throw SynthesisError.decoderFailed
        }
        let hifiganOutputStats = computeStats(of: waveformArray)

        // 波形を Float 配列に変換
        let sampleCount = waveformArray.count
        var waveform = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            waveform[i] = waveformArray[i].floatValue
        }
        let waveformBeforeDeemphasis = computeStats(of: waveform)

        // 5. デエンファシスフィルタ: y[n] = x[n] + coeff * y[n-1]
        await MainActor.run { onProgress("後処理中...", 0.95) }
        waveform = AudioFeatureExtractor.applyDeemphasis(waveform)
        let waveformAfterDeemphasis = computeStats(of: waveform)

        // デバッグ情報をまとめる
        let debugInfo = PipelineDebugInfo(
            encoderOutput: encoderStats,
            decoderSteps: decoderStepStats,
            hifiganInput: hifiganInputStats,
            hifiganOutput: hifiganOutputStats,
            waveformBeforeDeemphasis: waveformBeforeDeemphasis,
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
            debugInfo: debugInfo
        )
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
