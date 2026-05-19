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
        let count = result.outputWaveform.count
        let rms = count == 0 ? Float.nan : Float((sumSq / Double(count)).squareRoot())
        let rmsInt16 = rms * 32767
        let peakInt16 = peak * 32767
        if hasNaN || hasInf { return "nan_inf" }
        if rms.isNaN { return "predict_failed" }
        if peakInt16 > 32000 && rmsInt16 > 7000 { return "clipped" }
        if rmsInt16 < 3000 { return "quiet" }
        return "normal_loud"
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
