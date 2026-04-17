import CoreML

/// Transformer Encoder の推論を 1 回実行する
///
/// 入力メルスペクトログラムを Encoder に通して memory（中間表現）を得る。
/// memory は後段の Decoder で全ステップを通じて繰り返し参照される。
final class EncoderRunner {
    private let model: MLModel

    init(model: MLModel) {
        self.model = model
    }

    /// Encoder を 1 回実行する
    /// - Parameters:
    ///   - mel: 入力メル [T × nMels] を行優先で並べた配列
    ///   - frameCount: T (時間方向のフレーム数)
    ///   - nMels: メルビン数 (256)
    /// - Returns: Encoder 出力の memory (MLMultiArray)
    func run(mel: [Float], frameCount: Int, nMels: Int) async throws -> MLMultiArray {
        // 入力 mel: [1, T, nMels] の MLMultiArray に値を詰める
        let melArray = try MLMultiArray(shape: [1, frameCount as NSNumber, nMels as NSNumber], dataType: .float32)
        for i in 0..<(frameCount * nMels) {
            melArray[i] = NSNumber(value: mel[i])
        }

        // 入力 pos: [1, T] に 1, 2, ..., T を詰める（位置エンコーディング用）
        let posArray = try MLMultiArray(shape: [1, frameCount as NSNumber], dataType: .int32)
        for i in 0..<frameCount {
            posArray[i] = NSNumber(value: Int32(i + 1))
        }

        // CoreML の入力辞書を作って predict
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "pos": MLFeatureValue(multiArray: posArray)
        ])
        let output = try await model.prediction(from: input)

        // 出力から memory を取り出す（出力キーは 1 つだけの前提）
        guard let memoryFeature = output.featureValue(for: output.featureNames.first ?? ""),
              let memory = memoryFeature.multiArrayValue else {
            throw AudioSynthesizer.SynthesisError.decoderFailed
        }
        return memory
    }
}
