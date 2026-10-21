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
    private let policy: TransformerInputPolicy

    init(model: MLModel, policy: TransformerInputPolicy = .dynamic(maxT: 1000)) {
        self.model = model
        self.policy = policy
    }

    /// Decoder を frameCount 回まわして自己回帰的に出力を生成する
    /// - Parameters:
    ///   - memory: Encoder の出力 [1, T_src, hidden]
    ///   - frameCount: 生成するフレーム数 T (入力メルのフレーム数と同じ)
    ///   - nMels: メルビン数 (256)
    ///   - onStep: 各ステップ完了時に呼ばれる (引数は完了済みステップ数, 1-indexed)
    ///   - debug: 調査用 snapshot (nil なら何もしない)。各 step の mel_out / postnet_out の
    ///            sha256 + stats を `decoder_steps.csv` に追記する。
    /// - Returns: 最終ステップの postnet 出力、選別された各ステップの統計、`predict()` 全ステップ合計の所要時間 (ms)、
    ///            各ステップの `predict()` 所要時間 (ms)、入力 decoder_input に含まれた NaN/Inf 個数の全ステップ最大値
    func run(
        memory: MLMultiArray,
        frameCount: Int,
        nMels: Int,
        onStep: @MainActor (Int) -> Void,
        debug: DebugRunSnapshot? = nil
    ) async throws -> (
        postnetOut: MLMultiArray, stepStats: [DecoderStepStats], totalPredictMs: Double,
        stepPredictMs: [Double], maxInputNonFiniteCount: Int
    ) {
        // 初期入力: ゼロベクトル [1, 1, 256]（最初の 1 フレームぶん）
        var decoderInputData = [Float](repeating: 0, count: nMels)
        var currentLength = 1
        var lastMelOut: MLMultiArray?
        var lastPostnetOut: MLMultiArray?
        var stepStats = [DecoderStepStats]()
        var totalPredictMs: Double = 0
        var stepPredictMs = [Double]()
        var maxInputNonFiniteCount = 0
        let maxLength: Int
        let fixedLength: Int?
        switch policy {
        case .fixed(let targetT):
            maxLength = targetT
            fixedLength = targetT
        case .dynamic(let maxT):
            maxLength = maxT
            fixedLength = nil
        }
        let effectiveFrameCount = min(frameCount, maxLength)
        if frameCount > maxLength {
            print("[DecoderRunner] WARN: frameCount=\(frameCount) > maxLength=\(maxLength), cropped to \(effectiveFrameCount)")
        }

        for step in 0..<effectiveFrameCount {
            // decoder_input: [1, currentLength, 256]
            let inputLength = fixedLength ?? currentLength
            // 固定長では先頭 currentLength フレームだけ書き込むので、残り（未生成の部分）は 0 で埋めておく
            let decInput = try ModelInputArray.zeros(shape: [1, inputLength, nMels], dataType: .float32)
            for i in 0..<(currentLength * nMels) {
                decInput[i] = NSNumber(value: decoderInputData[i])
            }
            // 前ステップの出力が NaN/Inf でも止めない（不安定な条件の観測を後段まで残すため）
            let nonFiniteCount = try ModelInputArray.validateFloat32(
                decInput, paddingRanges: [(currentLength * nMels)..<(inputLength * nMels)], name: "Decoder input"
            )
            maxInputNonFiniteCount = max(maxInputNonFiniteCount, nonFiniteCount)
            if nonFiniteCount > 0 {
                print("[DecoderRunner] WARN: step=\(step) decoder_input contains \(nonFiniteCount) NaN/Inf values")
            }

            // pos: [1, currentLength]
            let decPos = try MLMultiArray(shape: [1, inputLength as NSNumber], dataType: .int32)
            for i in 0..<inputLength {
                decPos[i] = NSNumber(value: Int32(i + 1))
            }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "memory": MLFeatureValue(multiArray: memory),
                "decoder_input": MLFeatureValue(multiArray: decInput),
                "pos": MLFeatureValue(multiArray: decPos)
            ])
            let t0 = CFAbsoluteTimeGetCurrent()
            let output = try await model.prediction(from: input)
            let stepMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            totalPredictMs += stepMs
            stepPredictMs.append(stepMs)

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

            // 調査用: 全 step の sha + stats を CSV に append (debug enabled 時のみ)
            debug?.appendDecoderStep(step: step, melOut: melOut, postnetOut: lastPostnetOut)

            let shouldRecord = step < 5
                || step >= effectiveFrameCount - 5
                || (effectiveFrameCount > 10 && step >= effectiveFrameCount / 2 - 2 && step <= effectiveFrameCount / 2 + 2)
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
        return (postnetOut, stepStats, totalPredictMs, stepPredictMs, maxInputNonFiniteCount)
    }
}
