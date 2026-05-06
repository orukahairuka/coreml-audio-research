import CoreML

/// Transformer Encoder の推論を 1 回実行する
///
/// 入力メルスペクトログラムを Encoder に通して memory（中間表現）を得る。
/// memory は後段の Decoder で全ステップを通じて繰り返し参照される。
final class EncoderRunner {
    private let model: MLModel
    private let policy: TransformerInputPolicy

    init(model: MLModel, policy: TransformerInputPolicy = .dynamic(maxT: 1000)) {
        self.model = model
        self.policy = policy
    }

    /// Encoder を 1 回実行する
    /// - Parameters:
    ///   - mel: 入力メル [T × nMels] を行優先で並べた配列
    ///   - frameCount: T (時間方向のフレーム数)
    ///   - nMels: メルビン数 (256)
    /// - Returns: Encoder 出力の memory と、`predict()` の所要時間 (ms)
    func run(mel: [Float], frameCount: Int, nMels: Int) async throws -> (memory: MLMultiArray, predictMs: Double) {
        let inputT: Int
        let contentT: Int
        switch policy {
        case .fixed(let targetT):
            inputT = targetT
            contentT = min(frameCount, targetT)
            if frameCount > targetT {
                print("[EncoderRunner] WARN: frameCount=\(frameCount) > targetT=\(targetT), cropped to \(contentT)")
            }
        case .dynamic(let maxT):
            inputT = min(frameCount, maxT)
            contentT = inputT
            if frameCount > maxT {
                print("[EncoderRunner] WARN: frameCount=\(frameCount) > maxT=\(maxT), cropped to \(inputT)")
            }
        }

        // 入力 mel: [1, T, nMels] の MLMultiArray に値を詰める
        let melArray = try MLMultiArray(shape: [1, inputT as NSNumber, nMels as NSNumber], dataType: .float32)
        for i in 0..<(contentT * nMels) {
            melArray[i] = NSNumber(value: mel[i])
        }

        // 入力 pos: [1, T] に 1, 2, ..., T を詰める（位置エンコーディング用）
        let posArray = try MLMultiArray(shape: [1, inputT as NSNumber], dataType: .int32)
        for i in 0..<inputT {
            posArray[i] = NSNumber(value: Int32(i + 1))
        }

        // CoreML の入力辞書を作って predict (時間計測は predict 呼び出しのみ)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "pos": MLFeatureValue(multiArray: posArray)
        ])
        let t0 = CFAbsoluteTimeGetCurrent()
        let output = try await model.prediction(from: input)
        let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // 出力から memory を取り出す（出力キーは 1 つだけの前提）
        guard let memoryFeature = output.featureValue(for: output.featureNames.first ?? ""),
              let memory = memoryFeature.multiArrayValue else {
            throw AudioSynthesizer.SynthesisError.decoderFailed
        }
        return (memory, predictMs)
    }
}
