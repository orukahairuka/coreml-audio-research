import Testing
@testable import CoreMLAudioApp
import AVFoundation
import CoreML
import Foundation

/// F32 × cpuAndGPU が「XCUITest 経由だと quiet, 手動だと loud」になる問題の切り分け用テスト。
///
/// 仮説: UI 自動操作 (XCUITest runner プロセス) ではなく、XCTest 環境全体が影響している可能性。
/// このテストは UI を介さず AudioSynthesizer を直接呼ぶ。Unit テストはアプリホストプロセス
/// 内で走るので XCTestRunner は介在しない。結果:
///   - loud になる → 「UI 自動操作のみが原因」
///   - quiet になる → 「XCTest 環境全般 (テストホスト初期化, ヒープ状態, signal handler 等) が原因」
///
/// CMLA_DEBUG_SNAPSHOT=1 をプロセス内で setenv して、`Documents/Result/debug/` に
/// mel / encoder / postnet / waveform を書き出す。extract_ui_test_results.sh で吸い出せる。
///
/// 2026-05-19 追加: 他組合せの Repeat3, decoder step 別 sha, dummy warm-up テストも追加した。
/// Swift Testing は並列実行がデフォルトだが `setenv` で run label を切り替える都合上
/// `.serialized` を付けて直列化する。
@Suite(.serialized)
struct Fp32QuietInvestigationTests {

    /// F32 × cpuAndGPU を直接 1 回実行する (UI 自動操作なし)。
    @Test
    func directSynthesisFp32GpuOnce() async throws {
        try await runDirect(label: "directRun1", precision: .float32, computeUnit: .cpuAndGPU)
    }

    /// F32 × cpuAndGPU を 3 回連続で直接実行する (同一プロセス内、AudioSynthesizer インスタンス共有)。
    @Test
    func directSynthesisFp32GpuRepeat3() async throws {
        try await runRepeat3(precision: .float32, computeUnit: .cpuAndGPU, labelPrefix: "directRepeat")
    }

    // MARK: - 2026-05-19 追加: 他組合せの Repeat3 (収束現象が F32×cpuAndGPU 固有か検証)

    @Test
    func repeat3F16Gpu() async throws {
        try await runRepeat3(precision: .float16, computeUnit: .cpuAndGPU, labelPrefix: "f16GpuRepeat")
    }

    @Test
    func repeat3Int8Gpu() async throws {
        try await runRepeat3(precision: .int8, computeUnit: .cpuAndGPU, labelPrefix: "int8GpuRepeat")
    }

    @Test
    func repeat3F32CpuOnly() async throws {
        try await runRepeat3(precision: .float32, computeUnit: .cpuOnly, labelPrefix: "f32CpuRepeat")
    }

    @Test
    func repeat3F32All() async throws {
        try await runRepeat3(precision: .float32, computeUnit: .all, labelPrefix: "f32AllRepeat")
    }

    // MARK: - 2026-05-19 追加: dummy warm-up → 本番 synthesize テスト

    /// `loadModels` 直後に dummy synthesize を 1 回入れ、その後の本番 synthesize が manual と
    /// bit-identical な loud (postnet sha `23c51a5431...`, int16_rms 5029, int16_peak 24326) に
    /// なるか確認する。
    @Test
    func fp32GpuDummyWarmupThenReal() async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = ShapeModeOption.fixed262
        try synth.loadModels(
            precision: .float32,
            computeUnits: ComputeUnitOption.cpuAndGPU.mlComputeUnits,
            shapeMode: shapeMode
        )
        guard let inputURL = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
            #expect(Bool(false), "input_sample.wav が host app bundle に見つかりません")
            return
        }

        // 1 回目: warm-up. 出力は捨てる
        setenv("CMLA_DEBUG_RUN_LABEL", "warmupDummy", 1)
        _ = try await synth.synthesize(
            inputURL: inputURL,
            precision: .float32,
            computeUnit: .cpuAndGPU,
            onProgress: { _, _ in }
        )

        // 2 回目: 本番。これが manual と一致するはず
        setenv("CMLA_DEBUG_RUN_LABEL", "warmupReal", 1)
        let result = try await synth.synthesize(
            inputURL: inputURL,
            precision: .float32,
            computeUnit: .cpuAndGPU,
            onProgress: { _, _ in }
        )
        logWavStats(result: result, tag: "warmupReal")
    }

    // MARK: - helpers

    private func runDirect(label: String, precision: ModelPrecision, computeUnit: ComputeUnitOption) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        setenv("CMLA_DEBUG_RUN_LABEL", label, 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        try await synthesizeOnce(synthesizer: synth, precision: precision, computeUnit: computeUnit, tag: label)
    }

    private func runRepeat3(precision: ModelPrecision, computeUnit: ComputeUnitOption, labelPrefix: String) async throws {
        setenv("CMLA_DEBUG_SNAPSHOT", "1", 1)
        let synth = AudioSynthesizer()
        let shapeMode = pickShapeMode(for: precision)
        try synth.loadModels(
            precision: precision,
            computeUnits: computeUnit.mlComputeUnits,
            shapeMode: shapeMode
        )
        for i in 1...3 {
            setenv("CMLA_DEBUG_RUN_LABEL", "\(labelPrefix)\(i)", 1)
            try await synthesizeOnce(synthesizer: synth, precision: precision, computeUnit: computeUnit, tag: "\(labelPrefix)\(i)")
        }
    }

    private func pickShapeMode(for precision: ModelPrecision) -> ShapeModeOption {
        if ShapeModeOption.fixed262.isAvailable(for: precision) { return .fixed262 }
        return .range1
    }

    private func synthesizeOnce(
        synthesizer: AudioSynthesizer,
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption,
        tag: String
    ) async throws {
        guard let inputURL = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
            #expect(Bool(false), "input_sample.wav が host app bundle に見つかりません")
            return
        }

        let result = try await synthesizer.synthesize(
            inputURL: inputURL,
            precision: precision,
            computeUnit: computeUnit,
            onProgress: { _, _ in }
        )
        logWavStats(result: result, tag: tag)

        var sumSq: Double = 0
        for v in result.outputWaveform { sumSq += Double(v) * Double(v) }
        let rms = result.outputWaveform.isEmpty
            ? Float.nan
            : Float((sumSq / Double(result.outputWaveform.count)).squareRoot())
        // 緩い sanity: rms が極端に小さい (デエンファシス後ほぼ無音) なら異常
        #expect(rms > 1e-4, "rms が極端に低い (\(tag)): \(rms)")
    }

    private func logWavStats(result: SynthesisResult, tag: String) {
        var sumSq: Double = 0
        var peak: Float = 0
        for v in result.outputWaveform {
            sumSq += Double(v) * Double(v)
            if abs(v) > peak { peak = abs(v) }
        }
        let rms = result.outputWaveform.isEmpty
            ? Float.nan
            : Float((sumSq / Double(result.outputWaveform.count)).squareRoot())
        let rmsInt16 = rms * 32767
        let peakInt16 = peak * 32767
        print(String(
            format: "[Fp32QuietInvestigationTests] tag=%@ count=%d float_rms=%.6f float_peak=%.6f int16_rms=%.1f int16_peak=%.1f",
            tag, result.outputWaveform.count, rms, peak, rmsInt16, peakInt16
        ))
    }
}
