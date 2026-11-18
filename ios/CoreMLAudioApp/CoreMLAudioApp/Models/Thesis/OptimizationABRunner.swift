import CoreML
import Foundation

/// 高速化の前後を同じ起動の中で比べる計測モード（`CMLA_THESIS_BENCH=optab`）
///
/// iPhone 13 ではモデルの読み込みから約 10 秒の間 Decoder の呼び出しが短いので、最初に空運転をして捨てる。
/// その後、入力ごとに A と B を交互に合成する。順番の偏りを避けるため、回ごとに A→B と B→A を入れ替える。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_THESIS_OPT_A` | 変更前の高速化の指定（`none` またはカンマ区切りの名前） |
/// | `CMLA_THESIS_OPT_B` | 変更後の高速化の指定 |
/// | `CMLA_THESIS_PREROLL_SEC` | 空運転の秒数（既定 15） |
final class OptimizationABRunner {

    enum ConfigError: LocalizedError {
        case invalid(String, String)

        var errorDescription: String? {
            switch self {
            case .invalid(let key, let value): return "環境変数 \(key) の値が不正です: \(value)"
            }
        }
    }

    static let csvName = "optab_runs.csv"

    static let header = ThesisRecorder.commonHeader + [
        "input_name", "round", "position", "variant", "optimizations", "status", "error",
        "thermal_before", "thermal_after", "started_at", "T", "T_prime", "decoder_steps",
        "e2e_ms", "predict_sum_ms", "non_predict_ms", "encoder_ms", "decoder_total_ms", "hifigan_ms",
        "mel_in_sha256", "decoder_mel_sha256", "hifigan_wave_sha256", "deemph_wave_sha256",
    ] + ProfileStage.all.map({ "\($0)_ms" }) + ThesisRecorder.versionHeader

    private let config: ThesisBenchConfig
    private let recorder: ThesisRecorder
    private let variantA: SynthesisOptimizations
    private let variantB: SynthesisOptimizations
    private let prerollSeconds: Double

    init(
        config: ThesisBenchConfig, recorder: ThesisRecorder,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        self.config = config
        self.recorder = recorder
        func variant(_ key: String) throws -> SynthesisOptimizations {
            let raw = environment[key] ?? "none"
            guard let parsed = SynthesisOptimizations.parse(raw) else { throw ConfigError.invalid(key, raw) }
            return parsed
        }
        variantA = try variant("CMLA_THESIS_OPT_A")
        variantB = try variant("CMLA_THESIS_OPT_B")
        let rawPreroll = environment["CMLA_THESIS_PREROLL_SEC"] ?? "15"
        guard let preroll = Double(rawPreroll), preroll >= 0 else {
            throw ConfigError.invalid("CMLA_THESIS_PREROLL_SEC", rawPreroll)
        }
        prerollSeconds = preroll
    }

    func run() async throws {
        let precision = config.precision ?? .float16
        let computeUnit = config.computeUnit ?? .cpuOnly
        let urls = try ThesisPipeline.modelURLs(precision: precision)
        recorder.writeCrumb("mode=optab stage=load condition=\(config.conditionLabel)")
        let enc = try ThesisPipeline.loadModel(at: urls.encoder, computeUnit: computeUnit)
        let dec = try ThesisPipeline.loadModel(at: urls.decoder, computeUnit: computeUnit)
        let hifi = try ThesisPipeline.loadModel(at: urls.hifigan, computeUnit: config.hifiganComputeUnit ?? computeUnit)
        let pipeline = ThesisPipeline(encoder: enc.model, decoder: dec.model, hifigan: hifi.model)
        let names = try config.inputNames ?? ThesisPipeline.inputInfos().map({ $0.input_name })

        // 空運転（結果は記録しない）
        let prerollStart = CFAbsoluteTimeGetCurrent()
        recorder.writeCrumb("mode=optab stage=preroll")
        while CFAbsoluteTimeGetCurrent() - prerollStart < prerollSeconds {
            _ = try await pipeline.synthesize(
                inputURL: try ThesisPipeline.inputURL(name: names[0]), optimizations: variantA
            )
        }

        for name in names {
            let url = try ThesisPipeline.inputURL(name: name)
            for round in 0..<config.steadyCount {
                let order: [(String, SynthesisOptimizations)] = round % 2 == 0
                    ? [("A", variantA), ("B", variantB)] : [("B", variantB), ("A", variantA)]
                for (position, entry) in order.enumerated() {
                    await measure(pipeline, name: name, url: url, round: round, position: position,
                                  variant: entry.0, optimizations: entry.1)
                }
            }
        }
    }

    private func measure(
        _ pipeline: ThesisPipeline, name: String, url: URL, round: Int, position: Int,
        variant: String, optimizations: SynthesisOptimizations
    ) async {
        recorder.writeCrumb("mode=optab input=\(name) round=\(round) variant=\(variant)")
        var values = recorder.commonValues()
        values["input_name"] = name
        values["round"] = String(round)
        values["position"] = String(position)
        values["variant"] = variant
        values["optimizations"] = optimizations.label
        values["thermal_before"] = ThesisRecorder.thermalState
        values["started_at"] = ThesisRecorder.timestamp()
        do {
            let profiler = StageProfiler()
            let m = try await pipeline.synthesize(inputURL: url, profiler: profiler, optimizations: optimizations)
            let predictMs = ProfileStage.predictStages.map({ profiler.ms($0) }).reduce(0, +)
            values["status"] = "success"
            values["T"] = String(m.totalFrames)
            values["T_prime"] = String(m.effectiveFrames)
            values["decoder_steps"] = String(m.decoderStepMs.count)
            values["e2e_ms"] = ThesisBenchRunner.ms(m.e2eMs)
            values["predict_sum_ms"] = ThesisBenchRunner.ms(predictMs)
            values["non_predict_ms"] = ThesisBenchRunner.ms(profiler.sumMs - predictMs)
            values["encoder_ms"] = ThesisBenchRunner.ms(m.encoderMs)
            values["decoder_total_ms"] = ThesisBenchRunner.ms(m.decoderTotalMs)
            values["hifigan_ms"] = ThesisBenchRunner.ms(m.hifiganMs)
            values["mel_in_sha256"] = ThesisRecorder.sha256(m.melIn)
            values["decoder_mel_sha256"] = ThesisRecorder.sha256(m.decoderMel)
            values["hifigan_wave_sha256"] = ThesisRecorder.sha256(m.hifiganWaveform)
            values["deemph_wave_sha256"] = ThesisRecorder.sha256(m.deemphWaveform)
            for stage in ProfileStage.all {
                values["\(stage)_ms"] = ThesisBenchRunner.ms(profiler.ms(stage))
            }
        } catch {
            values["status"] = "failed"
            values["error"] = error.localizedDescription
        }
        values["thermal_after"] = ThesisRecorder.thermalState
        recorder.appendRow(csvName: Self.csvName, header: Self.header, values: values)
    }
}
