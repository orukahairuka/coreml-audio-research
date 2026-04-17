import CoreML

/// Transformer Decoder を自己回帰的に実行する
///
/// 入力 memory (Encoder 出力) を参照しながら、メルスペクトログラムを 1 フレームずつ生成する。
/// frameCount 回ループして decoder_input を 1 フレームずつ伸ばしていく。
///
/// 各ステップで前回までの出力全体を入力として渡す素朴な自己回帰実装になっており、
/// CoreML の動的入力形状を使う。
final class DecoderRunner {
    private let model: MLModel

    init(model: MLModel) {
        self.model = model
    }

    /// Decoder を frameCount 回まわして自己回帰的に出力を生成する
    /// - Parameters:
    ///   - memory: Encoder の出力 [1, T_src, hidden]
    ///   - frameCount: 生成するフレーム数 T (入力メルのフレーム数と同じ)
    ///   - nMels: メルビン数 (256)
    ///   - onStep: 各ステップ完了時に呼ばれる (引数は完了済みステップ数, 1-indexed)
    /// - Returns: 最終ステップの postnet 出力と、選別された各ステップの統計
    func run(
        memory: MLMultiArray,
        frameCount: Int,
        nMels: Int,
        onStep: @MainActor (Int) -> Void
    ) async throws -> (postnetOut: MLMultiArray, stepStats: [DecoderStepStats]) {
        // 初期入力: ゼロベクトル [1, 1, 256]（最初の 1 フレームぶん）
        var decoderInputData = [Float](repeating: 0, count: nMels)
        var currentLength = 1
        var lastMelOut: MLMultiArray?
        var lastPostnetOut: MLMultiArray?
        var stepStats = [DecoderStepStats]()

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

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "memory": MLFeatureValue(multiArray: memory),
                "decoder_input": MLFeatureValue(multiArray: decInput),
                "pos": MLFeatureValue(multiArray: decPos)
            ])
            let output = try await model.prediction(from: input)

            // mel_out は変換時に明示命名済み。postnet 出力は残りのキーから取得
            let postnetKey = output.featureNames.first(where: { $0 != "mel_out" }) ?? ""
            lastMelOut = output.featureValue(for: "mel_out")?.multiArrayValue
            lastPostnetOut = output.featureValue(for: postnetKey)?.multiArrayValue

            guard let melOut = lastMelOut else {
                throw AudioSynthesizer.SynthesisError.decoderFailed
            }

            // ステップ統計の選別記録: 先頭 5 + 中間 5 + 末尾 5 + NaN/Inf 検出時は全ステップ
            let melStats = ArrayStats.compute(from: melOut)
            let postStats = lastPostnetOut.map { ArrayStats.compute(from: $0) }
                ?? ArrayStats(min: 0, max: 0, mean: 0, hasNaN: false, hasInf: false)
            let shouldRecord = step < 5
                || step >= frameCount - 5
                || (frameCount > 10 && step >= frameCount / 2 - 2 && step <= frameCount / 2 + 2)
                || melStats.hasNaN || melStats.hasInf
                || postStats.hasNaN || postStats.hasInf
            if shouldRecord {
                stepStats.append(DecoderStepStats(
                    step: step, melOut: melStats, postnetOut: postStats
                ))
            }

            // 最後のフレームを次の入力に追加（自己回帰）
            for i in 0..<nMels {
                decoderInputData.append(melOut[[0, (currentLength - 1) as NSNumber, i as NSNumber]].floatValue)
            }
            currentLength += 1

            // 進捗通知（パーセント計算は呼び出し側に任せる）
            await MainActor.run { onStep(step + 1) }

            // UI 更新のため定期的に yield
            if step % 10 == 0 {
                await Task.yield()
            }
        }

        guard let postnetOut = lastPostnetOut else {
            throw AudioSynthesizer.SynthesisError.decoderFailed
        }
        return (postnetOut, stepStats)
    }
}
