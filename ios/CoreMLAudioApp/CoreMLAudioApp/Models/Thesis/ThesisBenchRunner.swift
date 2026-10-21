import CoreML
import Foundation

/// 卒論第4章の計測モード本体。UI を使わずに 1 つのモードを実行して終わる
///
/// - `load`：最初の読み込みを 1 回（起動直後）測り、解放してからの再読み込みを `reloadCount` 回測る
/// - `plan`：3 モデルの MLComputePlan を JSON で保存する（参考用）
/// - `infer`：3 モデルを読み込み、入力ごとにウォームアップ＋定常時の合成を測る
/// - `vocoder`：HiFi-GAN だけを読み込み、PyTorch 版 Decoder の出力メルを入れて測る
/// - `thermal`：`thermalState` を出力するだけ（PC 側が nominal に戻るのを待つのに使う）
///
/// 失敗した実行も行として残す。最後に `THESIS_BENCH_DONE` をコンソールに出す
/// （PC 側はこの行が無ければ異常終了とみなす）。
final class ThesisBenchRunner {

    static let loadCsv = "load_runs.csv"
    static let inferCsv = "infer_runs.csv"
    static let vocoderCsv = "vocoder_runs.csv"

    private let config: ThesisBenchConfig
    private let recorder: ThesisRecorder

    init(config: ThesisBenchConfig) throws {
        self.config = config
        self.recorder = try ThesisRecorder(config: config)
    }

    /// 環境変数から設定を読んで実行し、終了コードを返す
    static func runFromEnvironment() async -> Int32 {
        do {
            let config = try ThesisBenchConfig.fromEnvironment()
            ThesisRecorder.log("THESIS_BENCH_START mode=\(config.mode.rawValue) launch=\(config.launchID) "
                + "condition=\(config.conditionLabel) device=\(ThesisRecorder.deviceModel) os=\(ThesisRecorder.osVersion) "
                + "thermal=\(ThesisRecorder.thermalState)")
            if config.mode == .thermal {
                ThesisRecorder.log("THESIS_THERMAL=\(ThesisRecorder.thermalState)")
                ThesisRecorder.log("THESIS_BENCH_DONE status=ok")
                return 0
            }
            let runner = try ThesisBenchRunner(config: config)
            try await runner.run()
            runner.recorder.removeCrumb()
            ThesisRecorder.log("THESIS_BENCH_DONE status=ok")
            return 0
        } catch {
            ThesisRecorder.log("THESIS_BENCH_DONE status=setup_failed error=\(error.localizedDescription)")
            return 1
        }
    }

    func run() async throws {
        switch config.mode {
        case .load: runLoad()
        case .plan: try await runPlan()
        case .infer: try await runInfer()
        case .vocoder: try await runVocoder()
        case .thermal: break
        }
    }

    private var precision: ModelPrecision { config.precision ?? .float32 }
    private var computeUnit: ComputeUnitOption { config.computeUnit ?? .cpuOnly }

    // MARK: - load

    static let loadHeader = ThesisRecorder.commonHeader + [
        "kind", "index", "status", "error", "thermal_before", "thermal_after", "started_at",
        "encoder_ms", "decoder_ms", "hifigan_ms", "total_ms",
    ]

    private func runLoad() {
        let urls: ThesisPipeline.ModelURLs
        do {
            urls = try ThesisPipeline.modelURLs(precision: precision)
        } catch {
            recordLoad(kind: "first", index: 0, thermalBefore: ThesisRecorder.thermalState,
                       startedAt: ThesisRecorder.timestamp(), times: nil, error: error)
            return
        }
        for index in 0...config.reloadCount {
            let kind = index == 0 ? "first" : "reload"
            recorder.writeCrumb("mode=load kind=\(kind) index=\(index) condition=\(config.conditionLabel)")
            let thermalBefore = ThesisRecorder.thermalState
            let startedAt = ThesisRecorder.timestamp()
            // autoreleasepool の中で読み込み、抜けたら 3 モデルとも解放される
            let outcome: Result<[Double], Error> = autoreleasepool(invoking: {
                do {
                    let enc = try ThesisPipeline.loadModel(at: urls.encoder, computeUnit: computeUnit)
                    let dec = try ThesisPipeline.loadModel(at: urls.decoder, computeUnit: computeUnit)
                    let hifi = try ThesisPipeline.loadModel(at: urls.hifigan, computeUnit: computeUnit)
                    return .success([enc.ms, dec.ms, hifi.ms])
                } catch {
                    return .failure(error)
                }
            })
            switch outcome {
            case .success(let times):
                recordLoad(kind: kind, index: index, thermalBefore: thermalBefore, startedAt: startedAt, times: times, error: nil)
            case .failure(let error):
                recordLoad(kind: kind, index: index, thermalBefore: thermalBefore, startedAt: startedAt, times: nil, error: error)
            }
        }
    }

    private func recordLoad(kind: String, index: Int, thermalBefore: String, startedAt: String, times: [Double]?, error: Error?) {
        var values = recorder.commonValues()
        values["kind"] = kind
        values["index"] = String(index)
        values["status"] = error == nil ? "success" : "failed"
        values["error"] = error?.localizedDescription ?? ""
        values["thermal_before"] = thermalBefore
        values["thermal_after"] = ThesisRecorder.thermalState
        values["started_at"] = startedAt
        if let times {
            values["encoder_ms"] = Self.ms(times[0])
            values["decoder_ms"] = Self.ms(times[1])
            values["hifigan_ms"] = Self.ms(times[2])
            values["total_ms"] = Self.ms(times.reduce(0, +))
        }
        recorder.appendRow(csvName: Self.loadCsv, header: Self.loadHeader, values: values)
        ThesisRecorder.log("[thesis] load \(config.conditionLabel) \(kind)#\(index) "
            + "total_ms=\(values["total_ms"] ?? "-") status=\(values["status"] ?? "")")
    }

    // MARK: - plan

    private func runPlan() async throws {
        let urls = try ThesisPipeline.modelURLs(precision: precision)
        let dir = recorder.directory
            .appendingPathComponent("compute_plan", isDirectory: true)
            .appendingPathComponent(config.conditionLabel, isDirectory: true)
        let models: [(name: String, url: URL)] = [
            ("encoder", urls.encoder), ("decoder", urls.decoder), ("hifigan", urls.hifigan),
        ]
        for model in models {
            recorder.writeCrumb("mode=plan model=\(model.name) condition=\(config.conditionLabel)")
            let inspector = ComputePlanInspector(
                modelURL: model.url,
                modelName: model.url.deletingPathExtension().lastPathComponent,
                precision: precision,
                computeUnits: computeUnit,
                outputDirectory: dir
            )
            do {
                let summary = try await inspector.inspectAndWrite()
                ThesisRecorder.log("[thesis] plan \(config.conditionLabel) \(model.name) \(summary)")
            } catch {
                ThesisRecorder.log("[thesis] plan \(config.conditionLabel) \(model.name) failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - infer / vocoder の行

    static let runHeader = ThesisRecorder.commonHeader + [
        "input_name", "T", "T_prime", "run_index", "phase", "status", "error",
        "thermal_before", "thermal_after", "started_at",
        "mel_in_sha256", "decoder_mel_sha256", "hifigan_wave_sha256", "deemph_wave_sha256",
        "encoder_in_nonfinite", "decoder_in_nonfinite_max", "hifigan_in_nonfinite",
        "preprocess_ms", "encoder_ms", "decoder_total_ms", "decoder_steps",
        "decoder_step_ms_median", "decoder_step_ms_min", "decoder_step_ms_max",
        "hifigan_ms", "synth_ms", "deemph_ms", "e2e_ms", "audio_sec", "rtf",
        "hifigan_min", "hifigan_max", "hifigan_rms", "hifigan_nan", "hifigan_inf",
        "hifigan_over1_count", "hifigan_over1_ratio",
        "deemph_min", "deemph_max", "deemph_rms", "deemph_nan", "deemph_inf",
        "deemph_over1_count", "deemph_over1_ratio", "deemph_over099_ratio",
        "file_prefix",
    ]

    /// 何回目の実行か（読み込み直後の 1 回目は initial）
    private func phase(isFirstAfterLoad: Bool, runIndex: Int) -> String {
        if isFirstAfterLoad { return "initial" }
        return runIndex < config.warmupCount ? "warmup" : "steady"
    }

    private func waveformValues(prefix: String, _ waveform: [Float]) -> [String: String] {
        let s = WaveformStatistics(waveform: waveform)
        var values: [String: String] = [
            "\(prefix)_min": Self.sci(s.min),
            "\(prefix)_max": Self.sci(s.max),
            "\(prefix)_rms": Self.sci(s.rms),
            "\(prefix)_nan": String(s.nanCount),
            "\(prefix)_inf": String(s.infCount),
            "\(prefix)_over1_count": String(s.over1Count),
            "\(prefix)_over1_ratio": String(format: "%.8f", s.ratioAbsOver1),
        ]
        if prefix == "deemph" {
            values["deemph_over099_ratio"] = String(format: "%.8f", s.ratioOver099 + s.ratioUnderMinus099)
        }
        return values
    }

    // MARK: - infer

    private func runInfer() async throws {
        let urls = try ThesisPipeline.modelURLs(precision: precision)
        recorder.writeCrumb("mode=infer stage=load condition=\(config.conditionLabel)")
        let enc = try ThesisPipeline.loadModel(at: urls.encoder, computeUnit: computeUnit)
        let dec = try ThesisPipeline.loadModel(at: urls.decoder, computeUnit: computeUnit)
        let hifi = try ThesisPipeline.loadModel(at: urls.hifigan, computeUnit: computeUnit)
        ThesisRecorder.log("[thesis] infer \(config.conditionLabel) loaded in "
            + Self.ms(enc.ms + dec.ms + hifi.ms) + " ms")
        let pipeline = ThesisPipeline(encoder: enc.model, decoder: dec.model, hifigan: hifi.model)

        let names = try config.inputNames ?? ThesisPipeline.inputInfos().map({ $0.input_name })
        var isFirstAfterLoad = true
        for name in names {
            for runIndex in 0..<(config.warmupCount + config.steadyCount) {
                let phase = phase(isFirstAfterLoad: isFirstAfterLoad, runIndex: runIndex)
                isFirstAfterLoad = false
                let prefix = "infer_\(config.conditionLabel)_in\(name)_r\(runIndex)"
                recorder.writeCrumb("mode=infer input=\(name) run_index=\(runIndex) phase=\(phase) condition=\(config.conditionLabel)")

                var values = recorder.commonValues()
                values["input_name"] = name
                values["run_index"] = String(runIndex)
                values["phase"] = phase
                values["thermal_before"] = ThesisRecorder.thermalState
                values["started_at"] = ThesisRecorder.timestamp()
                do {
                    let m = try await pipeline.synthesize(inputURL: try ThesisPipeline.inputURL(name: name))
                    values["thermal_after"] = ThesisRecorder.thermalState
                    values.merge(inferValues(m), uniquingKeysWith: { _, new in new })
                    values["status"] = "success"
                    values["file_prefix"] = "runs/\(config.launchID)/\(prefix)"
                    try saveInferFiles(m, prefix: prefix, saveMelIn: runIndex == 0)
                } catch {
                    values["thermal_after"] = ThesisRecorder.thermalState
                    values["status"] = "failed"
                    values["error"] = error.localizedDescription
                }
                recorder.appendRow(csvName: Self.inferCsv, header: Self.runHeader, values: values)
                ThesisRecorder.log("[thesis] infer \(config.conditionLabel) in=\(name) r=\(runIndex) \(phase) "
                    + "synth_ms=\(values["synth_ms"] ?? "-") rtf=\(values["rtf"] ?? "-") "
                    + "over1=\(values["deemph_over1_ratio"] ?? "-") status=\(values["status"] ?? "")")
            }
        }
    }

    private func inferValues(_ m: ThesisPipeline.Measurement) -> [String: String] {
        let audioSec = Double(m.effectiveFrames * 256) / AudioFeatureExtractor.sampleRate
        let sortedSteps = m.decoderStepMs.sorted()
        var values: [String: String] = [
            "T": String(m.totalFrames),
            "T_prime": String(m.effectiveFrames),
            "mel_in_sha256": ThesisRecorder.sha256(m.melIn),
            "decoder_mel_sha256": ThesisRecorder.sha256(m.decoderMel),
            "hifigan_wave_sha256": ThesisRecorder.sha256(m.hifiganWaveform),
            "deemph_wave_sha256": ThesisRecorder.sha256(m.deemphWaveform),
            "encoder_in_nonfinite": String(m.encoderInputNonFinite),
            "decoder_in_nonfinite_max": String(m.decoderInputNonFiniteMax),
            "hifigan_in_nonfinite": String(m.hifiganInputNonFinite),
            "preprocess_ms": Self.ms(m.preprocessMs),
            "encoder_ms": Self.ms(m.encoderMs),
            "decoder_total_ms": Self.ms(m.decoderTotalMs),
            "decoder_steps": String(m.decoderStepMs.count),
            "decoder_step_ms_median": sortedSteps.isEmpty ? "" : Self.ms(Self.median(sortedSteps)),
            "decoder_step_ms_min": sortedSteps.first.map({ Self.ms($0) }) ?? "",
            "decoder_step_ms_max": sortedSteps.last.map({ Self.ms($0) }) ?? "",
            "hifigan_ms": Self.ms(m.hifiganMs),
            "synth_ms": Self.ms(m.synthMs),
            "deemph_ms": Self.ms(m.deemphMs),
            "e2e_ms": Self.ms(m.e2eMs),
            "audio_sec": String(format: "%.6f", audioSec),
            "rtf": audioSec > 0 ? String(format: "%.6f", m.synthMs / 1000 / audioSec) : "",
        ]
        values.merge(waveformValues(prefix: "hifigan", m.hifiganWaveform), uniquingKeysWith: { _, new in new })
        values.merge(waveformValues(prefix: "deemph", m.deemphWaveform), uniquingKeysWith: { _, new in new })
        return values
    }

    private func saveInferFiles(_ m: ThesisPipeline.Measurement, prefix: String, saveMelIn: Bool) throws {
        let nMels = ThesisPipeline.nMels
        if saveMelIn {
            try recorder.writeNpy(m.melIn, rows: m.effectiveFrames, cols: nMels, fileName: "\(prefix)_mel_in.npy")
        }
        try recorder.writeNpy(m.decoderMel, rows: m.decoderMel.count / nMels, cols: nMels, fileName: "\(prefix)_decoder_mel.npy")
        try recorder.writeNpy(m.decoderStepMs.map({ Float($0) }), rows: 1, cols: m.decoderStepMs.count,
                              fileName: "\(prefix)_decoder_step_ms.npy")
        try recorder.writeNpy(m.hifiganWaveform, rows: 1, cols: m.hifiganWaveform.count, fileName: "\(prefix)_hifigan.npy")
        try recorder.writeNpy(m.deemphWaveform, rows: 1, cols: m.deemphWaveform.count, fileName: "\(prefix)_deemph.npy")
        try recorder.writeWav(m.deemphWaveform, sampleRate: AudioFeatureExtractor.sampleRate, fileName: "\(prefix)_deemph.wav")
    }

    // MARK: - vocoder

    private func runVocoder() async throws {
        let urls = try ThesisPipeline.modelURLs(precision: precision)
        recorder.writeCrumb("mode=vocoder stage=load condition=\(config.conditionLabel)")
        let hifi = try ThesisPipeline.loadModel(at: urls.hifigan, computeUnit: computeUnit)
        ThesisRecorder.log("[thesis] vocoder \(config.conditionLabel) loaded in \(Self.ms(hifi.ms)) ms")
        let runner = VocoderRunner(model: hifi.model, policy: ThesisBenchConfig.shapeMode.inputPolicy)

        let infos = try ThesisPipeline.inputInfos()
        let names = config.inputNames ?? infos.map({ $0.input_name })
        let nMels = ThesisPipeline.nMels
        var isFirstAfterLoad = true
        for name in names {
            guard let info = infos.first(where: { $0.input_name == name }) else {
                throw ThesisPipeline.PipelineError.inputNotFound(name)
            }
            let mel = try MinimalNpyReader.readFloat32(contentsOf: try ThesisPipeline.pytorchMelURL(name: name))
            let frames = mel.count / nMels
            let melArray = try ModelInputArray.zeros(shape: [1, frames, nMels], dataType: .float32)
            for i in 0..<mel.count {
                melArray[i] = NSNumber(value: mel[i])
            }
            let melSha = ThesisRecorder.sha256(mel)

            for runIndex in 0..<(config.warmupCount + config.steadyCount) {
                let phase = phase(isFirstAfterLoad: isFirstAfterLoad, runIndex: runIndex)
                isFirstAfterLoad = false
                let prefix = "vocoder_\(config.conditionLabel)_in\(name)_r\(runIndex)"
                recorder.writeCrumb("mode=vocoder input=\(name) run_index=\(runIndex) phase=\(phase) condition=\(config.conditionLabel)")

                var values = recorder.commonValues()
                values["input_name"] = name
                values["T"] = String(info.T)
                values["T_prime"] = String(info.T_prime)
                values["run_index"] = String(runIndex)
                values["phase"] = phase
                values["mel_in_sha256"] = melSha
                values["thermal_before"] = ThesisRecorder.thermalState
                values["started_at"] = ThesisRecorder.timestamp()
                do {
                    let result = try await runner.run(postnetOut: melArray, totalFrames: info.T_prime, nMels: nMels)
                    let deemph = AudioFeatureExtractor.applyDeemphasis(result.waveform)
                    values["thermal_after"] = ThesisRecorder.thermalState
                    values["status"] = "success"
                    values["hifigan_in_nonfinite"] = String(result.inputNonFiniteCount)
                    values["hifigan_ms"] = Self.ms(result.predictMs)
                    values["hifigan_wave_sha256"] = ThesisRecorder.sha256(result.waveform)
                    values["deemph_wave_sha256"] = ThesisRecorder.sha256(deemph)
                    values.merge(waveformValues(prefix: "hifigan", result.waveform), uniquingKeysWith: { _, new in new })
                    values.merge(waveformValues(prefix: "deemph", deemph), uniquingKeysWith: { _, new in new })
                    values["file_prefix"] = "runs/\(config.launchID)/\(prefix)"
                    try recorder.writeNpy(result.waveform, rows: 1, cols: result.waveform.count, fileName: "\(prefix)_hifigan.npy")
                    try recorder.writeNpy(deemph, rows: 1, cols: deemph.count, fileName: "\(prefix)_deemph.npy")
                    try recorder.writeWav(deemph, sampleRate: AudioFeatureExtractor.sampleRate, fileName: "\(prefix)_deemph.wav")
                } catch {
                    values["thermal_after"] = ThesisRecorder.thermalState
                    values["status"] = "failed"
                    values["error"] = error.localizedDescription
                }
                recorder.appendRow(csvName: Self.vocoderCsv, header: Self.runHeader, values: values)
                ThesisRecorder.log("[thesis] vocoder \(config.conditionLabel) in=\(name) r=\(runIndex) \(phase) "
                    + "hifigan_ms=\(values["hifigan_ms"] ?? "-") over1=\(values["deemph_over1_ratio"] ?? "-") "
                    + "status=\(values["status"] ?? "")")
            }
        }
    }

    // MARK: - 書式

    static func ms(_ value: Double) -> String { String(format: "%.3f", value) }

    static func sci(_ value: Float) -> String { String(format: "%.6e", value) }

    static func median(_ sorted: [Double]) -> Double {
        let n = sorted.count
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}
