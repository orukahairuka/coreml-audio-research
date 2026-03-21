import CoreML
import Accelerate
import Observation

/// CoreML モデル 3 つ (Encoder, Decoder, HiFi-GAN) を使って音声合成を行う
@MainActor
@Observable
final class AudioSynthesizer {

    var status: String = "待機中"
    var isProcessing: Bool = false
    var progress: Double = 0

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var hifigan: MLModel?

    /// CoreML モデルをロードする（ロード済みの場合はスキップ）
    func loadModels() throws {
        if encoder != nil { return }

        status = "モデルをロード中..."

        guard let encoderURL = Bundle.main.url(forResource: "Transformer_Encoder", withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: "Transformer_Decoder", withExtension: "mlmodelc"),
              let hifiganURL = Bundle.main.url(forResource: "HiFiGAN_Generator", withExtension: "mlmodelc") else {
            throw SynthesisError.modelNotFound
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        encoder = try MLModel(contentsOf: encoderURL, configuration: config)
        decoder = try MLModel(contentsOf: decoderURL, configuration: config)
        hifigan = try MLModel(contentsOf: hifiganURL, configuration: config)

        status = "モデルロード完了"
    }

    /// 入力音声 URL から合成を実行し、出力波形を返す
    func synthesize(inputURL: URL) async throws -> [Float] {
        guard let encoder, let decoder, let hifigan else {
            throw SynthesisError.modelNotLoaded
        }

        isProcessing = true
        progress = 0
        defer { isProcessing = false }

        // 1. メルスペクトログラム抽出
        status = "特徴量抽出中..."
        let (melData, frameCount) = try AudioFeatureExtractor.extractMelSpectrogram(from: inputURL)
        let nMels = AudioFeatureExtractor.nMels

        // 2. Encoder
        status = "Encoder 実行中..."
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
        let memory = encoderOutput.featureValue(for: encoderOutput.featureNames.first!)!.multiArrayValue!

        // 3. Decoder (自己回帰ループ)
        status = "Decoder 実行中... (0/\(frameCount))"

        // 初期入力: ゼロベクトル [1, 1, 256]
        var decoderInputData = [Float](repeating: 0, count: nMels)
        var currentLength = 1
        var lastMelOut: MLMultiArray?
        var lastPostnetOut: MLMultiArray?

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

            // 出力キーを取得
            let outputNames = decoderOutput.featureNames.sorted()
            lastMelOut = decoderOutput.featureValue(for: outputNames[0])?.multiArrayValue
            lastPostnetOut = decoderOutput.featureValue(for: outputNames[1])?.multiArrayValue

            // 最後のフレームを取得して入力に追加
            guard let melOut = lastMelOut else { throw SynthesisError.decoderFailed }
            for i in 0..<nMels {
                decoderInputData.append(melOut[[0, (currentLength - 1) as NSNumber, i as NSNumber]].floatValue)
            }
            currentLength += 1

            progress = Double(step + 1) / Double(frameCount)
            status = "Decoder 実行中... (\(step + 1)/\(frameCount))"

            // UI 更新のために yield
            if step % 10 == 0 {
                await Task.yield()
            }
        }

        // 4. HiFi-GAN
        status = "HiFi-GAN 実行中..."
        guard let postnetOut = lastPostnetOut else { throw SynthesisError.decoderFailed }

        // postnet_out: [1, T, 256] → [1, 256, T] に転置
        let totalFrames = frameCount
        let vocoderInput = try MLMultiArray(shape: [1, nMels as NSNumber, totalFrames as NSNumber], dataType: .float32)
        for t in 0..<totalFrames {
            for m in 0..<nMels {
                vocoderInput[[0, m as NSNumber, t as NSNumber]] = postnetOut[[0, t as NSNumber, m as NSNumber]]
            }
        }

        let hifiganInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: vocoderInput)
        ])
        let hifiganOutput = try await hifigan.prediction(from: hifiganInput)
        let waveformArray = hifiganOutput.featureValue(for: hifiganOutput.featureNames.first!)!.multiArrayValue!

        // 波形を Float 配列に変換
        let sampleCount = waveformArray.count
        var waveform = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            waveform[i] = waveformArray[i].floatValue
        }

        // 5. デエンファシスフィルタ: y[n] = x[n] + coeff * y[n-1]
        status = "後処理中..."
        let coeff = AudioFeatureExtractor.preemphasisCoeff
        for i in 1..<waveform.count {
            waveform[i] = waveform[i] + coeff * waveform[i - 1]
        }

        status = "合成完了"
        progress = 1.0
        return waveform
    }

    enum SynthesisError: LocalizedError {
        case modelNotFound
        case modelNotLoaded
        case decoderFailed

        var errorDescription: String? {
            switch self {
            case .modelNotFound: return "CoreML モデルが見つかりません。.mlpackage をプロジェクトに追加してください。"
            case .modelNotLoaded: return "モデルがロードされていません。"
            case .decoderFailed: return "Decoder の出力を取得できませんでした。"
            }
        }
    }
}
