import CoreML

/// Transformer Encoder の推論を 1 回実行する
///
/// 入力メルスペクトログラムを Encoder に通して memory（中間表現）を得る。
/// memory は後段の Decoder で全ステップを通じて繰り返し参照される。
nonisolated final class EncoderRunner {
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
    ///   - profiler: 処理別計測（卒論第4章の追加計測用）。nil なら計測しない
    ///   - optimizations: 高速化の指定（結果は変わらない）
    /// - Returns: Encoder 出力の memory、`predict()` の所要時間 (ms)、入力メルに含まれた NaN/Inf の個数
    nonisolated(nonsending) func run(
        mel: [Float], frameCount: Int, nMels: Int, profiler: StageProfiler? = nil,
        optimizations: SynthesisOptimizations = .adopted
    ) async throws -> (
        memory: MLMultiArray, predictMs: Double, inputNonFiniteCount: Int
    ) {
        if optimizations.contains(.offMainInference) {
            return try await performOffMain(
                mel: mel, frameCount: frameCount, nMels: nMels, profiler: profiler, optimizations: optimizations
            )
        }
        return try await perform(
            mel: mel, frameCount: frameCount, nMels: nMels, profiler: profiler, optimizations: optimizations
        )
    }

    /// `perform` をメインアクターの外で実行する
    @concurrent private func performOffMain(
        mel: [Float], frameCount: Int, nMels: Int, profiler: StageProfiler?, optimizations: SynthesisOptimizations
    ) async throws -> (memory: MLMultiArray, predictMs: Double, inputNonFiniteCount: Int) {
        try await perform(mel: mel, frameCount: frameCount, nMels: nMels, profiler: profiler, optimizations: optimizations)
    }

    private nonisolated(nonsending) func perform(
        mel: [Float], frameCount: Int, nMels: Int, profiler: StageProfiler?, optimizations: SynthesisOptimizations
    ) async throws -> (memory: MLMultiArray, predictMs: Double, inputNonFiniteCount: Int) {
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

        // 入力 mel: [1, T, nMels] の MLMultiArray に値を詰める。
        // 確保しただけでは中身が不定なので 0 で埋めてから、先頭 contentT フレームに書き込む
        let melArray = try ModelInputArray.zeros(shape: [1, inputT, nMels], dataType: .float32)
        if optimizations.contains(.encoderInputCopy) {
            try ModelInputArray.copyFloat32(mel, count: contentT * nMels, into: melArray, name: "Encoder mel")
        } else {
            for i in 0..<(contentT * nMels) {
                melArray[i] = NSNumber(value: mel[i])
            }
        }
        profiler?.lap(ProfileStage.encInputFill)
        let nonFiniteCount = try ModelInputArray.validateFloat32(
            melArray, paddingRanges: [(contentT * nMels)..<(inputT * nMels)], name: "Encoder mel"
        )
        if nonFiniteCount > 0 {
            print("[EncoderRunner] WARN: input mel contains \(nonFiniteCount) NaN/Inf values")
        }
        profiler?.lap(ProfileStage.encInputValidate)

        // 入力 pos: [1, T] に 1, 2, ..., T を詰める（位置エンコーディング用）
        let posArray = try MLMultiArray(shape: [1, inputT as NSNumber], dataType: .int32)
        for i in 0..<inputT {
            posArray[i] = NSNumber(value: Int32(i + 1))
        }
        profiler?.lap(ProfileStage.encPosFill)

        // CoreML の入力辞書を作って predict (時間計測は predict 呼び出しのみ)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "pos": MLFeatureValue(multiArray: posArray)
        ])
        profiler?.lap(ProfileStage.encFeatureProvider)
        let t0 = CFAbsoluteTimeGetCurrent()
        let output = optimizations.contains(.syncPrediction)
            ? try model.predictSynchronously(from: input)
            : try await model.prediction(from: input)
        let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        profiler?.lap(ProfileStage.encPredict)

        // 出力から memory を取り出す（出力キーは 1 つだけの前提）
        guard let memoryFeature = output.featureValue(for: output.featureNames.first ?? ""),
              let memory = memoryFeature.multiArrayValue else {
            throw AudioSynthesizer.SynthesisError.decoderFailed
        }
        profiler?.lap(ProfileStage.encOutputExtract)
        return (memory, predictMs, nonFiniteCount)
    }
}
