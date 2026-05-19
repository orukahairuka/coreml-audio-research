import Testing
@testable import CoreMLAudioApp
import AVFoundation
import CoreML
import Foundation

/// Phase 2 — 救済実験テスト。
///
/// 計画書: [`docs/2026-05-19/all-engine-precision-stability-plan.md`](../../../../docs/2026-05-19/all-engine-precision-stability-plan.md) §3
///
/// Phase 1 で `normal_loud` 以外（quiet / clipped / nan_inf）が出た組合せに対し、
/// 以下の救済戦略で `normal_loud` に持っていけるかを計測する。
///
/// | ID | 戦略 |
/// |---|---|
/// | A | dummy warm-up 1 回 → 本番 |
/// | B | dummy warm-up 2 回 → 本番 |
/// | C | warm-up → 検査 → retry（最大 N 回） |
/// | F | cpuOnly で warm-up → cpuAndGPU で本番（loadModels 2 回） |
/// | G | 毎 iter で AudioSynthesizer 再生成 |
/// | H | fixed262 vs range16_384 vs range16 比較 |
/// | I | 別 precision を先に 1 回走らせてから対象を走らせる |
///
/// D（Decoder のみ warm-up）と E（段別 warm-up）は AudioSynthesizer の private 段モデルへの
/// アクセスを必要とするため、別途 `Models/Debug/AudioSynthesizerWarmup.swift` を整備した後で追加する。
///
/// 各テストの run ラベルは `phase2_<strategy>_<precision>_<computeUnit>_<iter>` 形式で
/// aggregate スクリプトが拾える形に揃える。
@Suite(.serialized)
struct Phase2RescueTests {

    // MARK: - Strategy A — dummy warm-up 1 回 → 本番 3 回

    @Test
    func strategyA_f32CpuAndGPU() async throws {
        try await runWarmupNThenIters(
            precision: .float32, computeUnit: .cpuAndGPU,
            warmupCount: 1, productionIters: 3,
            strategyId: "A"
        )
    }

    @Test
    func strategyA_f16CpuAndGPU() async throws {
        try await runWarmupNThenIters(
            precision: .float16, computeUnit: .cpuAndGPU,
            warmupCount: 1, productionIters: 3,
            strategyId: "A"
        )
    }

    @Test
    func strategyA_int8CpuAndGPU() async throws {
        try await runWarmupNThenIters(
            precision: .int8, computeUnit: .cpuAndGPU,
            warmupCount: 1, productionIters: 3,
            strategyId: "A"
        )
    }

    // MARK: - Strategy B — dummy warm-up 2 回 → 本番 3 回

    @Test
    func strategyB_f32CpuAndGPU() async throws {
        try await runWarmupNThenIters(
            precision: .float32, computeUnit: .cpuAndGPU,
            warmupCount: 2, productionIters: 3,
            strategyId: "B"
        )
    }

    // MARK: - Strategy C — warm-up → 検査 → retry

    @Test
    func strategyC_f32CpuAndGPU() async throws {
        try await runWithRetry(
            precision: .float32, computeUnit: .cpuAndGPU,
            maxAttempts: 4,
            strategyId: "C"
        )
    }

    // MARK: - Strategy F — cpuOnly で warm-up → cpuAndGPU で本番

    @Test
    func strategyF_f32_cpuOnlyToGpu() async throws {
        try await runCrossComputeUnitsWarmup(
            warmupComputeUnit: .cpuOnly,
            productionComputeUnit: .cpuAndGPU,
            precision: .float32,
            productionIters: 3,
            strategyId: "F"
        )
    }

    // MARK: - Strategy G — 毎 iter で AudioSynthesizer 再生成

    @Test
    func strategyG_f32CpuAndGPU() async throws {
        try await runPerIterReload(
            precision: .float32, computeUnit: .cpuAndGPU,
            iters: 3,
            strategyId: "G"
        )
    }

    // MARK: - Strategy H — shape mode 比較

    @Test
    func strategyH_f32CpuAndGPU_range16_384() async throws {
        try await runWithShapeMode(
            precision: .float32, computeUnit: .cpuAndGPU,
            shapeMode: .range16_384,
            iters: 3,
            strategyId: "H"
        )
    }

    // MARK: - Strategy I — 別 precision を先行実行してから本番

    @Test
    func strategyI_f16BeforeF32CpuAndGPU() async throws {
        try await runWithPrecursorRun(
            precursorPrecision: .float16, precursorComputeUnit: .cpuAndGPU,
            targetPrecision: .float32, targetComputeUnit: .cpuAndGPU,
            targetIters: 3,
            strategyId: "I"
        )
    }

    // MARK: - Strategy J — clipped 波形に対する出力正規化（Phase 2 mini）
    //
    // F16/Int8 × cpuAndNE で出る clipped 波形 (rms 13000+, peak 70000+) に対し、
    // 3 通りの後処理で audible level に落とせるかと、聴感上 audio quality が救えるかを観測する。
    //
    // - peakNormalize: max(abs) を 0.95 に揃える
    // - rmsNormalize : rms を manual baseline (0.1534 ≒ 5029/32767) に揃える
    // - fixedGain025 : 一律 ×0.25
    //
    // 各 strategy ごとに別 runId の DebugRunSnapshot を作って `waveform_postdeemph.npy` を書く。
    // 既存 `scripts/npy_to_wav.py` でそのまま wav 化できるよう命名を合わせる。

    @Test
    func strategyJ_f16CpuAndNE_normalize() async throws {
        try await runWithOutputNormalize(
            precision: .float16, computeUnit: .cpuAndNE, strategyId: "J"
        )
    }

    @Test
    func strategyJ_int8CpuAndNE_normalize() async throws {
        try await runWithOutputNormalize(
            precision: .int8, computeUnit: .cpuAndNE, strategyId: "J"
        )
    }

    // MARK: - Strategy K — HiFi-GAN だけ cpuAndGPU に逃がす（Phase 2 mini）
    //
    // Decoder は cpuAndNE のまま、HiFi-GAN だけ cpuAndGPU に載せ直して合成する。
    // Phase 4 で「HiFi-GAN の 81 op が NE 行きで、HiFi-GAN 内部で振幅差が出ている」観測があったため、
    // HiFi-GAN を NE 経路から外せば clipping が緩むかを確認する。
    //
    // Decoder の dispatch は変えないので、Decoder 出力（postnet）は cpuAndNE 単体ケースと一致するはず。
    // → HiFi-GAN を入れ替えただけで波形が変わるかどうかが論点。

    @Test
    func strategyK_f16_decoderAneHifiganGpu() async throws {
        try await runHifiganFallback(
            precision: .float16,
            decoderComputeUnit: .cpuAndNE,
            hifiganComputeUnit: .cpuAndGPU,
            iters: 3,
            strategyId: "K"
        )
    }

    @Test
    func strategyK_int8_decoderAneHifiganGpu() async throws {
        try await runHifiganFallback(
            precision: .int8,
            decoderComputeUnit: .cpuAndNE,
            hifiganComputeUnit: .cpuAndGPU,
            iters: 3,
            strategyId: "K"
        )
    }

    // MARK: - 共通 helper

    private func runWarmupNThenIters(
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        warmupCount: Int,
        productionIters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(computeUnit.rawValue)"
        for w in 1...max(1, warmupCount) {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_warmup\(w)", 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: computeUnit, tag: "\(labelBase)_warmup\(w)")
        }
        for i in 1...productionIters {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_prod\(i)", 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: computeUnit, tag: "\(labelBase)_prod\(i)")
        }
    }

    private func runWithRetry(
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        maxAttempts: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(computeUnit.rawValue)"
        var attempts = 0
        var lastClass = "predict_failed"
        while attempts < maxAttempts {
            attempts += 1
            let tag = "\(labelBase)_attempt\(attempts)"
            setenv("CMLA_DEBUG_RUN_LABEL", tag, 1)
            let result = try await synthesize(synth, precision: precision, computeUnit: computeUnit)
            lastClass = classify(result: result)
            logResult(result: result, tag: tag, classification: lastClass)
            if lastClass == "normal_loud" {
                print("[Phase2RescueTests] strategy=\(strategyId) succeeded after \(attempts) attempt(s)")
                break
            }
        }
        // 結果がどうあれ assertion せず、log と aggregate に判定を委ねる。
        print("[Phase2RescueTests] strategy=\(strategyId) final class=\(lastClass) attempts=\(attempts)")
    }

    private func runCrossComputeUnitsWarmup(
        warmupComputeUnit: ComputeUnitOption,
        productionComputeUnit: ComputeUnitOption,
        precision: ModelPrecision,
        productionIters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(warmupComputeUnit.rawValue)to\(productionComputeUnit.rawValue)"

        try synth.loadModels(
            precision: precision,
            computeUnits: warmupComputeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_warmup", 1)
        try await synthesizeOnce(synth, precision: precision, computeUnit: warmupComputeUnit, tag: "\(labelBase)_warmup")

        try synth.loadModels(
            precision: precision,
            computeUnits: productionComputeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        for i in 1...productionIters {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_prod\(i)", 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: productionComputeUnit, tag: "\(labelBase)_prod\(i)")
        }
    }

    private func runPerIterReload(
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        iters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(computeUnit.rawValue)"
        let shapeMode = pickShapeMode(for: precision)
        for i in 1...iters {
            let synth = AudioSynthesizer()
            try synth.loadModels(
                precision: precision,
                computeUnits: computeUnit.mlComputeUnits,
                shapeMode: shapeMode
            )
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_iter\(i)", 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: computeUnit, tag: "\(labelBase)_iter\(i)")
        }
    }

    private func runWithShapeMode(
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        shapeMode: ShapeModeOption,
        iters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(computeUnit.rawValue)_\(shapeMode.rawValue)"
        for i in 1...iters {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_iter\(i)", 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: computeUnit, tag: "\(labelBase)_iter\(i)")
        }
    }

    private func runWithOutputNormalize(
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_\(computeUnit.rawValue)"

        // 元波形を 1 回だけ取る（同一 instance なので毎回 bit-identical な前提）
        setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_raw", 1)
        let raw = try await synthesize(synth, precision: precision, computeUnit: computeUnit)
        let rawClass = classify(result: raw)
        logResult(result: raw, tag: "\(labelBase)_raw", classification: rawClass)

        // 3 種類の正規化を順に適用して、それぞれ別 runId の snapshot を書く。
        let strategies: [(String, [Float])] = [
            ("peakNorm",   normalizePeak(raw.outputWaveform, target: 0.95)),
            ("rmsNorm",    normalizeRms(raw.outputWaveform, target: 5029.0 / 32767.0)),
            ("fixedGain025", scaleWaveform(raw.outputWaveform, factor: 0.25)),
        ]
        for (name, normalized) in strategies {
            let tag = "\(labelBase)_\(name)"
            writeNormalizedSnapshot(
                waveform: normalized,
                precision: precision,
                computeUnit: computeUnit,
                tag: tag,
                strategyDescription: "post-process(\(name)) of \(labelBase)_raw"
            )
            let cls = classify(waveform: normalized)
            logWaveform(waveform: normalized, tag: tag, classification: cls)
        }
    }

    private func runHifiganFallback(
        precision: ModelPrecision,
        decoderComputeUnit: ComputeUnitOption,
        hifiganComputeUnit: ComputeUnitOption,
        iters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: decoderComputeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        try synth.reloadHifigan(
            precision: precision,
            computeUnits: hifiganComputeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        let labelBase = "phase2_\(strategyId)_\(precision.rawValue)_dec\(decoderComputeUnit.rawValue)_hifi\(hifiganComputeUnit.rawValue)"
        // synthesize の computeUnit 引数はログ用なので、HiFi-GAN 側に揃える。
        for i in 1...iters {
            let tag = "\(labelBase)_iter\(i)"
            setenv("CMLA_DEBUG_RUN_LABEL", tag, 1)
            try await synthesizeOnce(synth, precision: precision, computeUnit: hifiganComputeUnit, tag: tag)
        }
    }

    private func runWithPrecursorRun(
        precursorPrecision: ModelPrecision,
        precursorComputeUnit: ComputeUnitOption,
        targetPrecision: ModelPrecision,
        targetComputeUnit: ComputeUnitOption,
        targetIters: Int,
        strategyId: String
    ) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let labelBase = "phase2_\(strategyId)_\(precursorPrecision.rawValue)\(precursorComputeUnit.rawValue)Before\(targetPrecision.rawValue)\(targetComputeUnit.rawValue)"

        // 先行: 別 precision/engine
        let precursor = AudioSynthesizer()
        try precursor.loadModels(
            precision: precursorPrecision,
            computeUnits: precursorComputeUnit.mlComputeUnits,
            shapeMode: pickShapeMode(for: precursorPrecision)
        )
        setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_precursor", 1)
        try await synthesizeOnce(precursor, precision: precursorPrecision, computeUnit: precursorComputeUnit, tag: "\(labelBase)_precursor")

        // 本番: 別インスタンス
        let target = AudioSynthesizer()
        try target.loadModels(
            precision: targetPrecision,
            computeUnits: targetComputeUnit.mlComputeUnits,
            shapeMode: pickShapeMode(for: targetPrecision)
        )
        for i in 1...targetIters {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelBase)_target\(i)", 1)
            try await synthesizeOnce(target, precision: targetPrecision, computeUnit: targetComputeUnit, tag: "\(labelBase)_target\(i)")
        }
    }

    // MARK: - 共通の synthesize 呼び出し + 計測

    private func pickShapeMode(for precision: ModelPrecision) -> ShapeModeOption {
        if ShapeModeOption.fixed262.isAvailable(for: precision) { return .fixed262 }
        return .range1
    }

    private func synthesize(
        _ synth: AudioSynthesizer,
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption
    ) async throws -> SynthesisResult {
        guard let inputURL = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
            throw NSError(domain: "Phase2RescueTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "input_sample.wav が host app bundle に見つかりません"])
        }
        return try await synth.synthesize(
            inputURL: inputURL,
            precision: precision,
            computeUnit: computeUnit,
            onProgress: { _, _ in }
        )
    }

    private func synthesizeOnce(
        _ synth: AudioSynthesizer,
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        tag: String
    ) async throws {
        let result = try await synthesize(synth, precision: precision, computeUnit: computeUnit)
        let classification = classify(result: result)
        logResult(result: result, tag: tag, classification: classification)
    }

    private func classify(result: SynthesisResult) -> String {
        classify(waveform: result.outputWaveform)
    }

    private func classify(waveform: [Float]) -> String {
        var sumSq: Double = 0
        var peak: Float = 0
        var hasNaN = false
        var hasInf = false
        for v in waveform {
            if v.isNaN { hasNaN = true; continue }
            if v.isInfinite { hasInf = true; continue }
            sumSq += Double(v) * Double(v)
            if abs(v) > peak { peak = abs(v) }
        }
        let count = waveform.count
        let rms = count == 0 ? Float.nan : Float((sumSq / Double(count)).squareRoot())
        let rmsInt16 = rms * 32767
        let peakInt16 = peak * 32767
        if hasNaN || hasInf { return "nan_inf" }
        if rms.isNaN { return "predict_failed" }
        if peakInt16 > 32000 && rmsInt16 > 7000 { return "clipped" }
        if rmsInt16 < 3000 { return "quiet" }
        return "normal_loud"
    }

    // MARK: - 正規化ヘルパ（Phase 2 mini）

    /// peak (max abs) を `target` に揃える。0 入力はそのまま返す。
    private func normalizePeak(_ waveform: [Float], target: Float) -> [Float] {
        var peak: Float = 0
        for v in waveform where v.isFinite { if abs(v) > peak { peak = abs(v) } }
        guard peak > 0 else { return waveform }
        let scale = target / peak
        return waveform.map { $0 * scale }
    }

    /// rms を `target` (float -1..1 scale) に揃える。0 入力はそのまま返す。
    private func normalizeRms(_ waveform: [Float], target: Float) -> [Float] {
        var sumSq: Double = 0
        var count = 0
        for v in waveform where v.isFinite { sumSq += Double(v) * Double(v); count += 1 }
        guard count > 0 else { return waveform }
        let rms = Float((sumSq / Double(count)).squareRoot())
        guard rms > 0 else { return waveform }
        let scale = target / rms
        return waveform.map { $0 * scale }
    }

    private func scaleWaveform(_ waveform: [Float], factor: Float) -> [Float] {
        waveform.map { $0 * factor }
    }

    /// `waveform_postdeemph.npy` という命名で snapshot を書き、`npy_to_wav.py` に拾わせる。
    private func writeNormalizedSnapshot(
        waveform: [Float],
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        tag: String,
        strategyDescription: String
    ) {
        setenv("CMLA_DEBUG_RUN_LABEL", tag, 1)
        guard let snap = DebugRunSnapshot.makeIfEnabled(
            precision: precision.rawValue,
            computeUnit: computeUnit.rawValue
        ) else { return }
        snap.appendSummaryLine("strategy=\(tag)")
        snap.appendSummaryLine("note=\(strategyDescription)")
        snap.writeFloat1D(name: "waveform_postdeemph", data: waveform)
    }

    private func logWaveform(waveform: [Float], tag: String, classification: String) {
        var sumSq: Double = 0
        var peak: Float = 0
        var hasNaN = false
        var hasInf = false
        for v in waveform {
            if v.isNaN { hasNaN = true; continue }
            if v.isInfinite { hasInf = true; continue }
            sumSq += Double(v) * Double(v)
            if abs(v) > peak { peak = abs(v) }
        }
        let rms = waveform.isEmpty
            ? Float.nan
            : Float((sumSq / Double(waveform.count)).squareRoot())
        let rmsInt16 = rms * 32767
        let peakInt16 = peak * 32767
        print(String(
            format: "[Phase2RescueTests] tag=%@ count=%d int16_rms=%.1f int16_peak=%.1f has_nan=%@ has_inf=%@ class=%@",
            tag, waveform.count, rmsInt16, peakInt16,
            hasNaN ? "true" : "false", hasInf ? "true" : "false", classification
        ))
    }

    private func logResult(result: SynthesisResult, tag: String, classification: String) {
        var sumSq: Double = 0
        var peak: Float = 0
        var hasNaN = false
        var hasInf = false
        for v in result.outputWaveform {
            if v.isNaN { hasNaN = true; continue }
            if v.isInfinite { hasInf = true; continue }
            sumSq += Double(v) * Double(v)
            if abs(v) > peak { peak = abs(v) }
        }
        let rms = result.outputWaveform.isEmpty
            ? Float.nan
            : Float((sumSq / Double(result.outputWaveform.count)).squareRoot())
        let rmsInt16 = rms * 32767
        let peakInt16 = peak * 32767
        print(String(
            format: "[Phase2RescueTests] tag=%@ count=%d int16_rms=%.1f int16_peak=%.1f has_nan=%@ has_inf=%@ class=%@",
            tag, result.outputWaveform.count, rmsInt16, peakInt16,
            hasNaN ? "true" : "false", hasInf ? "true" : "false", classification
        ))
    }
}
