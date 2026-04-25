import CoreML

/// HiFi-GAN ボコーダーの推論を 1 回実行する
///
/// Decoder の postnet 出力（メルスペクトログラム）を波形 (PCM) に変換する。
/// 入力テンソルの転置と出力波形の Swift 配列化を内部で完結させる。
final class VocoderRunner {
    private let model: MLModel

    init(model: MLModel) {
        self.model = model
    }

    /// HiFi-GAN を 1 回実行する
    /// - Parameters:
    ///   - postnetOut: Decoder の postnet 出力 [1, T, nMels]
    ///   - totalFrames: T (時間方向のフレーム数)
    ///   - nMels: メルビン数 (256)
    /// - Returns: 合成波形 (デエンファシス前の生 PCM, 22050 Hz) と `predict()` の所要時間 (ms)
    func run(postnetOut: MLMultiArray, totalFrames: Int, nMels: Int) async throws -> (waveform: [Float], predictMs: Double) {
        // 1. 転置: [1, T, nMels] → [1, nMels, T]
        //    HiFi-GAN は時間軸を最後に持つ形式を要求する
        let input = try MLMultiArray(shape: [1, nMels as NSNumber, totalFrames as NSNumber], dataType: .float32)
        for t in 0..<totalFrames {
            for m in 0..<nMels {
                input[[0, m as NSNumber, t as NSNumber]] = postnetOut[[0, t as NSNumber, m as NSNumber]]
            }
        }

        // 2. CoreML predict (時間計測は predict 呼び出しのみ)
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: input)
        ])
        let t0 = CFAbsoluteTimeGetCurrent()
        let output = try await model.prediction(from: inputProvider)
        let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        guard let waveformFeature = output.featureValue(for: output.featureNames.first ?? ""),
              let waveformArray = waveformFeature.multiArrayValue else {
            throw AudioSynthesizer.SynthesisError.decoderFailed
        }

        // 3. MLMultiArray → [Float] に変換
        let sampleCount = waveformArray.count
        var waveform = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            waveform[i] = waveformArray[i].floatValue
        }
        return (waveform, predictMs)
    }
}
