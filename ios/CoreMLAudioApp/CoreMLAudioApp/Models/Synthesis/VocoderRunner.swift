import CoreML

/// HiFi-GAN ボコーダーの推論を 1 回実行する
///
/// 入力 T 処理は `VocoderInputPolicy` で切り替える:
/// - `.fixed(targetT)`: 固定 shape モデル (例: `_fixed262`)。`totalFrames` を targetT に zero-pad / crop
/// - `.dynamic(maxT)`: RangeDim モデル。`totalFrames` をそのまま渡し、上限超過時のみ crop
final class VocoderRunner {
    private let model: MLModel
    private let policy: VocoderInputPolicy
    private let hopSize: Int

    /// - Parameters:
    ///   - model: HiFi-GAN MLModel
    ///   - policy: 入力 T の処理ポリシー (本番デフォルトは `.fixed(targetT: 262)`)
    ///   - hopSize: 1 frame あたりの出力サンプル数 (HiFi-GAN config の hop_size = 256)
    init(model: MLModel, policy: VocoderInputPolicy = .fixed(targetT: 262), hopSize: Int = 256) {
        self.model = model
        self.policy = policy
        self.hopSize = hopSize
    }

    /// HiFi-GAN を 1 回実行する
    /// - Parameters:
    ///   - postnetOut: Decoder の postnet 出力 [1, T, nMels] (T は可変)
    ///   - totalFrames: T (時間方向のフレーム数)
    ///   - nMels: メルビン数 (256)
    /// - Returns: 合成波形 (デエンファシス前の生 PCM, 22050 Hz, 長さ = `contentT * hopSize`),
    ///            `predict()` の所要時間 (ms), モデルに渡した実効 T (`inputT`)
    func run(postnetOut: MLMultiArray, totalFrames: Int, nMels: Int) async throws -> (waveform: [Float], predictMs: Double, inputT: Int) {
        // policy に応じてモデルへ渡す T (inputT) と「中身のあるフレーム数」(contentT) を決める。
        // contentT 以降は zero-pad なので、出力波形もここまでで切り出す。
        let inputT: Int
        let contentT: Int
        switch policy {
        case .fixed(let targetT):
            inputT = targetT
            contentT = min(totalFrames, targetT)
            if totalFrames > targetT {
                print("[VocoderRunner] WARN: totalFrames=\(totalFrames) > targetT=\(targetT), cropped to \(contentT)")
            } else if totalFrames < targetT {
                print("[VocoderRunner] padded \(targetT - totalFrames) frames with zeros (T=\(totalFrames) → \(targetT))")
            }
        case .dynamic(let maxT):
            inputT = min(totalFrames, maxT)
            contentT = inputT
            if totalFrames > maxT {
                print("[VocoderRunner] WARN: totalFrames=\(totalFrames) > maxT=\(maxT), cropped to \(inputT)")
            }
        }

        // 1. 転置 + pad/crop: [1, T, nMels] → [1, nMels, inputT]
        //    MLMultiArray は初期値 0 なので、contentT より後ろは zero のまま残る (fixed 用)
        let input = try MLMultiArray(
            shape: [1, nMels as NSNumber, inputT as NSNumber],
            dataType: .float32
        )
        for t in 0..<contentT {
            for m in 0..<nMels {
                input[[0, m as NSNumber, t as NSNumber]] = postnetOut[[0, t as NSNumber, m as NSNumber]]
            }
        }
        print("[VocoderRunner] mel shape: \(input.shape) (T=\(totalFrames), inputT=\(inputT), contentT=\(contentT), nMels=\(nMels))")

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

        // 3. MLMultiArray → [Float] に変換。
        //    contentT 分のサンプルだけ返すので、padding 区間 (zeros) は捨てる。
        let validSamples = contentT * hopSize
        let sampleCount = min(waveformArray.count, validSamples)
        var waveform = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            waveform[i] = waveformArray[i].floatValue
        }
        return (waveform, predictMs, inputT)
    }
}
