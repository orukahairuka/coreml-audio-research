import CoreML
import Foundation
import UIKit

/// 正式 RTF 測定モード（起動時の環境変数 `CMLA_RTF_BENCH=1`）
///
/// 合成の経路は正式評価の一括合成（`FormalEvalRunner.synthesize`）と同じ:
/// range1・fullInput・Encoder → Decoder（T 回）→ HiFi-GAN → デエンファシス。違いは次の 3 点だけ。
///   - 同じ入力を、ウォームアップ `W` 回のあと `R` 回続けて合成し、1 回ごとに時間を記録する
///   - 出力の wav・npy は保存しない（ファイル I/O を測定に入れない。既存の正式合成結果にも触れない）
///   - 1 回ごとにメモリ（phys_footprint とその最大値）・端末の熱状態（`ProcessInfo.thermalState`）・電池を記録する
///
/// 計測範囲（開発者の `synthesis.py` の `with t.no_grad():` 以降、`sf.write` の前までに対応）:
///   `rtf_wall_ms` = Encoder の入力配列の作成から、デエンファシスの終わりまで
///   （Encoder・Decoder・HiFi-GAN の predict、その間の配列の処理、HiFi-GAN 出力の [Float] への読み出し、デエンファシス）
///   入力 wav の読み込みとメルの計算は含めない（入力ごとに 1 回だけ行い、反復では使い回す）。モデルの読み込みも含めない。
///
/// 入力の順番・ブロック・構成の順番はホスト（`metrics/rtf_official/run_rtf_official.py`）が決めて渡す。
/// アプリは状態を持たず、起動ごとに `Documents/Result/rtf_official/<study>/` に 1 起動 1 ファイルで書く
/// （再開はホストが、取り出した CSV を見て、終わっていない入力だけを渡して起動し直す）。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_RTF_STUDY` | 出力先のフォルダ名 |
/// | `CMLA_RTF_LAUNCH` | 起動の ID（ホストが付ける。ファイル名になる） |
/// | `CMLA_RTF_LABEL` | 構成のラベル（A / B / C） |
/// | `CMLA_RTF_PRECISION` | `Float32` / `Float16` / `Int8` |
/// | `CMLA_RTF_CU_ENC` / `_DEC` / `_HIFI` | モデルごとの Compute Units |
/// | `CMLA_RTF_DEC_INPUT` | `grow` / `padT` |
/// | `CMLA_RTF_SPEAKER` | 入力の話者フォルダ（`Documents/RTFInputs/<話者>/`） |
/// | `CMLA_RTF_NAMES` | 入力の名前（`<参照>_<話者>`）をカンマ区切りで。この順に測る |
/// | `CMLA_RTF_WARMUPS` | 入力ごとのウォームアップ回数 |
/// | `CMLA_RTF_REPEATS` | 入力ごとの本測定の回数 |
/// | `CMLA_RTF_THERMAL_WAIT_S` | 起動直後に熱状態が nominal になるまで待つ上限（秒）。0 なら待たない |
enum RTFBenchRunner {

    struct Config {
        let study: String
        let launch: String
        let label: String
        let precision: ModelPrecision
        let units: [String: ComputeUnitOption]
        let padDecoderInput: Bool
        let speaker: String
        let names: [String]
        let warmups: Int
        let repeats: Int
        let thermalWaitSeconds: Double

        static var isRequested: Bool { ProcessInfo.processInfo.environment["CMLA_RTF_BENCH"] == "1" }

        static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> Config {
            func value(_ key: String) throws -> String {
                guard let v = env[key], !v.isEmpty else { throw FormalEvalConfig.ConfigError.missing(key) }
                return v
            }
            func safe(_ key: String) throws -> String {
                let v = try value(key)
                let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
                guard v.unicodeScalars.allSatisfy({ allowed.contains($0) }), !v.hasPrefix(".") else {
                    throw FormalEvalConfig.ConfigError.invalid(key, v)
                }
                return v
            }
            func int(_ key: String) throws -> Int {
                let v = try value(key)
                guard let n = Int(v), n >= 0 else { throw FormalEvalConfig.ConfigError.invalid(key, v) }
                return n
            }
            let p = try value("CMLA_RTF_PRECISION")
            guard let precision = ModelPrecision(rawValue: p) else { throw FormalEvalConfig.ConfigError.invalid("CMLA_RTF_PRECISION", p) }
            var units: [String: ComputeUnitOption] = [:]
            for (role, key) in [("encoder", "CMLA_RTF_CU_ENC"), ("decoder", "CMLA_RTF_CU_DEC"), ("hifigan", "CMLA_RTF_CU_HIFI")] {
                let v = try value(key)
                guard let unit = ComputeUnitOption(rawValue: v) else { throw FormalEvalConfig.ConfigError.invalid(key, v) }
                units[role] = unit
            }
            let decInput = try value("CMLA_RTF_DEC_INPUT")
            guard decInput == "grow" || decInput == "padT" else {
                throw FormalEvalConfig.ConfigError.invalid("CMLA_RTF_DEC_INPUT", decInput)
            }
            let names = try value("CMLA_RTF_NAMES").split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) })
            let repeats = try int("CMLA_RTF_REPEATS")
            guard repeats > 0 else { throw FormalEvalConfig.ConfigError.invalid("CMLA_RTF_REPEATS", "0") }
            return Config(
                study: try safe("CMLA_RTF_STUDY"), launch: try safe("CMLA_RTF_LAUNCH"), label: try safe("CMLA_RTF_LABEL"),
                precision: precision, units: units, padDecoderInput: decInput == "padT", speaker: try safe("CMLA_RTF_SPEAKER"),
                names: names, warmups: try int("CMLA_RTF_WARMUPS"), repeats: repeats,
                thermalWaitSeconds: Double(try int("CMLA_RTF_THERMAL_WAIT_S"))
            )
        }

        var directory: URL {
            FormalEvalConfig.documents.appendingPathComponent("Result/rtf_official/\(study)", isDirectory: true)
        }
        var inputDirectory: URL {
            FormalEvalConfig.documents.appendingPathComponent("RTFInputs/\(speaker)", isDirectory: true)
        }
    }

    static let iterHeader = [
        "launch", "label", "precision", "input", "T", "input_samples", "position_in_launch", "phase", "rep",
        "global_iter", "started_at", "rtf_wall_ms", "encoder_wall_ms", "decoder_wall_ms", "hifigan_wall_ms", "deemph_ms",
        "encoder_predict_ms", "decoder_predict_ms", "hifigan_predict_ms", "output_samples", "audio_sec", "rtf",
        "wave_nonfinite", "wave_rms", "wave_peak", "footprint_mb", "footprint_peak_mb", "thermal_state",
        "battery_level", "battery_state", "low_power_mode", "status", "error",
    ]

    @MainActor
    static func runFromEnvironment() async -> Int32 {
        print("RTF_BENCH_START")
        do {
            let config = try Config.fromEnvironment()
            let (ok, errors) = try await run(config: config)
            print("RTF_BENCH_DONE status=\(errors == 0 ? "ok" : "partial") ok=\(ok) errors=\(errors)")
            return errors == 0 ? 0 : 2
        } catch {
            print("RTF_BENCH_DONE status=failed error=\(error)")
            return 1
        }
    }

    @MainActor
    static func run(config: Config) async throws -> (ok: Int, errors: Int) {
        let fm = FileManager.default
        try fm.createDirectory(at: config.directory, withIntermediateDirectories: true)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let iterURL = config.directory.appendingPathComponent("iter_\(config.launch).csv")
        let launchURL = config.directory.appendingPathComponent("launch_\(config.launch).json")
        guard !fm.fileExists(atPath: iterURL.path) else { throw RunError.launchExists(config.launch) }

        // 1. 起動直後: 熱状態が nominal になるまで待つ（上限あり）。待った時間と状態を記録する
        let thermalAtStart = ProcessInfo.processInfo.thermalState
        let waitStart = CFAbsoluteTimeGetCurrent()
        while ProcessInfo.processInfo.thermalState != .nominal,
              CFAbsoluteTimeGetCurrent() - waitStart < config.thermalWaitSeconds {
            try await Task.sleep(nanoseconds: 5_000_000_000)
        }
        let waitedSeconds = CFAbsoluteTimeGetCurrent() - waitStart
        let thermalAfterWait = ProcessInfo.processInfo.thermalState

        // 2. モデルを読み込む（時間は記録するが RTF には入れない）
        let resource = [
            "encoder": FormalEvalConfig.shapeMode.transformerEncoderResourceName(for: config.precision),
            "decoder": FormalEvalConfig.shapeMode.transformerDecoderResourceName(for: config.precision),
            "hifigan": FormalEvalConfig.shapeMode.hifiganResourceName(for: config.precision) ?? "?",
        ]
        var models: [String: MLModel] = [:]
        var loadMs: [String: Double] = [:]
        for role in ["encoder", "decoder", "hifigan"] {
            guard let name = resource[role], let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw FormalEvalRunner.RunError.modelNotFound(resource[role] ?? role)
            }
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = config.units[role]!.mlComputeUnits
            let t0 = CFAbsoluteTimeGetCurrent()
            models[role] = try MLModel(contentsOf: url, configuration: mlConfig)
            loadMs[role] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }
        let encoder = models["encoder"]!, decoder = models["decoder"]!, hifigan = models["hifigan"]!

        let launchInfo: [String: Any] = [
            "launch": config.launch, "label": config.label, "precision": config.precision.rawValue,
            "compute_units": config.units.mapValues({ $0.rawValue }),
            "decoder_input": config.padDecoderInput ? "padT" : "grow", "models": resource,
            "shape_mode": FormalEvalConfig.shapeMode.rawValue,
            "waveform_preprocess": FormalEvalConfig.waveformPreprocess.rawValue,
            "names": config.names, "warmups_per_input": config.warmups, "repeats_per_input": config.repeats,
            "started_at": ISO8601DateFormatter().string(from: Date()),
            "thermal_state_at_launch": thermalName(thermalAtStart), "thermal_wait_s": waitedSeconds,
            "thermal_state_after_wait": thermalName(thermalAfterWait),
            "thermal_wait_limit_s": config.thermalWaitSeconds,
            "load_ms": loadMs, "footprint_after_load_mb": FormalEvalRunner.footprintMB(),
            "battery_level": UIDevice.current.batteryLevel, "battery_state": batteryName(UIDevice.current.batteryState),
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "device": ["model": hardwareModel(), "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"],
            "timing_scope": "rtf_wall_ms = EncoderRunner.run start .. applyDeemphasis end (excludes wav load, mel, model load, save)",
        ]
        try JSONSerialization.data(withJSONObject: launchInfo, options: [.prettyPrinted, .sortedKeys]).write(to: launchURL)
        print("[RTF] launch=\(config.launch) label=\(config.label) thermal=\(thermalName(thermalAtStart))→"
              + "\(thermalName(thermalAfterWait)) waited=\(Int(waitedSeconds))s load=\(loadMs)")

        // 3. 入力ごとに、ウォームアップ W 回 → 本測定 R 回
        var okCount = 0
        var errorCount = 0
        var globalIter = 0
        for (position, name) in config.names.enumerated() {
            let url = config.inputDirectory.appendingPathComponent("\(name).wav")
            var mel: [Float] = []
            var T = 0
            var inputSamples = 0
            do {
                let samples = try AudioFeatureExtractor.loadAudio(from: url)
                let front = AudioFeatureExtractor.extractMelSpectrogram(
                    fromSamples: samples, waveformPreprocess: FormalEvalConfig.waveformPreprocess
                )
                mel = front.mel
                T = front.frameCount
                inputSamples = samples.count
            } catch {
                var row = baseRow(config: config, name: name, position: position)
                row["phase"] = "input"
                row["status"] = "error"
                row["error"] = "\(error)"
                try appendCSV(url: iterURL, header: iterHeader, row: iterHeader.map({ row[$0] ?? "" }))
                errorCount += 1
                continue
            }
            let phases = Array(repeating: "warmup", count: config.warmups) + Array(repeating: "timed", count: config.repeats)
            var repByPhase: [String: Int] = [:]
            for phase in phases {
                let rep = repByPhase[phase, default: 0]
                repByPhase[phase] = rep + 1
                var row = baseRow(config: config, name: name, position: position)
                row["phase"] = phase
                row["rep"] = "\(rep)"
                row["global_iter"] = "\(globalIter)"
                row["T"] = "\(T)"
                row["input_samples"] = "\(inputSamples)"
                row["started_at"] = ISO8601DateFormatter().string(from: Date())
                globalIter += 1
                do {
                    try await measureOnce(
                        mel: mel, T: T, encoder: encoder, decoder: decoder, hifigan: hifigan,
                        padDecoderInput: config.padDecoderInput, row: &row
                    )
                    row["status"] = "ok"
                    if phase == "timed" { okCount += 1 }
                } catch {
                    row["status"] = "error"
                    row["error"] = "\(error)"
                    errorCount += 1
                }
                row["footprint_mb"] = format(FormalEvalRunner.footprintMB())
                row["footprint_peak_mb"] = format(footprintPeakMB())
                row["thermal_state"] = thermalName(ProcessInfo.processInfo.thermalState)
                row["battery_level"] = String(format: "%.2f", UIDevice.current.batteryLevel)
                row["battery_state"] = batteryName(UIDevice.current.batteryState)
                row["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled ? "1" : "0"
                try appendCSV(url: iterURL, header: iterHeader, row: iterHeader.map({ row[$0] ?? "" }))
                print("[RTF] \(config.label) \(position + 1)/\(config.names.count) \(name) \(phase)#\(rep) T=\(T) "
                      + "wall=\(row["rtf_wall_ms"] ?? "-")ms rtf=\(row["rtf"] ?? "-") thermal=\(row["thermal_state"] ?? "?") "
                      + "\(row["status"] ?? "") \(row["error"] ?? "")")
            }
        }
        // 終わりの印（ホストはこのファイルがあれば起動が最後まで進んだとみなす）
        let done: [String: Any] = [
            "finished_at": ISO8601DateFormatter().string(from: Date()), "timed_ok": okCount, "errors": errorCount,
            "thermal_state_at_end": thermalName(ProcessInfo.processInfo.thermalState),
            "footprint_peak_mb": footprintPeakMB(),
        ]
        try JSONSerialization.data(withJSONObject: done, options: [.prettyPrinted, .sortedKeys])
            .write(to: config.directory.appendingPathComponent("done_\(config.launch).json"))
        return (okCount, errorCount)
    }

    /// 1 回合成して時間を測る。経路は `FormalEvalRunner.synthesize` の推論・後処理部分と同じ（保存はしない）
    private static func measureOnce(
        mel: [Float], T: Int, encoder: MLModel, decoder: MLModel, hifigan: MLModel, padDecoderInput: Bool,
        row: inout [String: String]
    ) async throws {
        let nMels = AudioFeatureExtractor.nMels
        let policy = FormalEvalConfig.shapeMode.transformerInputPolicy
        let decoderPolicy: TransformerInputPolicy = padDecoderInput ? .fixed(targetT: T) : policy

        let t0 = CFAbsoluteTimeGetCurrent()
        let (memory, encMs, _) = try await EncoderRunner(model: encoder, policy: policy)
            .run(mel: mel, frameCount: T, nMels: nMels)
        let t1 = CFAbsoluteTimeGetCurrent()
        let (postnet, _, decMs, _, _) = try await DecoderRunner(model: decoder, policy: decoderPolicy)
            .run(memory: memory, frameCount: T, nMels: nMels, onStep: { _ in }, collectStepStats: false)
        let t2 = CFAbsoluteTimeGetCurrent()
        let vocoder = try await VocoderRunner(model: hifigan, policy: FormalEvalConfig.shapeMode.inputPolicy)
            .run(postnetOut: postnet, totalFrames: T, nMels: nMels)
        let t3 = CFAbsoluteTimeGetCurrent()
        let wave = AudioFeatureExtractor.applyDeemphasis(vocoder.waveform)
        let t4 = CFAbsoluteTimeGetCurrent()

        let wallMs = (t4 - t0) * 1000
        let audioSec = Double(T * 256) / Double(AudioFeatureExtractor.sampleRate)
        row["rtf_wall_ms"] = format(wallMs)
        row["encoder_wall_ms"] = format((t1 - t0) * 1000)
        row["decoder_wall_ms"] = format((t2 - t1) * 1000)
        row["hifigan_wall_ms"] = format((t3 - t2) * 1000)
        row["deemph_ms"] = format((t4 - t3) * 1000)
        row["encoder_predict_ms"] = format(encMs)
        row["decoder_predict_ms"] = format(decMs)
        row["hifigan_predict_ms"] = format(vocoder.predictMs)
        row["output_samples"] = "\(wave.count)"
        row["audio_sec"] = String(format: "%.6f", audioSec)
        row["rtf"] = String(format: "%.6f", wallMs / 1000 / audioSec)
        // 統計は時間を測り終えてから取る
        row["wave_nonfinite"] = "\(wave.filter({ !$0.isFinite }).count)"
        row["wave_rms"] = String(format: "%.8f", (wave.reduce(0, { $0 + Double($1) * Double($1) }) / Double(max(wave.count, 1))).squareRoot())
        row["wave_peak"] = String(format: "%.8f", Double(wave.map({ abs($0) }).max() ?? 0))
        guard postnet.shape[1].intValue == T, wave.count == T * 256 else {
            throw FormalEvalRunner.RunError.unexpectedLength(T: T, samples: wave.count)
        }
    }

    // MARK: - 補助

    private static func baseRow(config: Config, name: String, position: Int) -> [String: String] {
        ["launch": config.launch, "label": config.label, "precision": config.precision.rawValue, "input": name,
         "position_in_launch": "\(position)"]
    }

    private static func appendCSV(url: URL, header: [String], row: [String]) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try (header.joined(separator: ",") + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data((row.map(FormalEvalRunner.escape).joined(separator: ",") + "\n").utf8))
    }

    private static func format(_ value: Double) -> String { String(format: "%.3f", value) }

    /// プロセスのメモリ使用量の最大値（ledger の phys_footprint の peak、MB）
    static func footprintPeakMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info, { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count), { reboundPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPointer, &count)
            })
        })
        return status == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1_048_576 : -1
    }

    static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func batteryName(_ state: UIDevice.BatteryState) -> String {
        switch state {
        case .unplugged: return "unplugged"
        case .charging: return "charging"
        case .full: return "full"
        case .unknown: return "unknown"
        @unknown default: return "unknown"
        }
    }

    private static func hardwareModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine, { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        })
    }

    enum RunError: LocalizedError {
        case launchExists(String)
        var errorDescription: String? {
            switch self {
            case .launchExists(let id): return "起動 \(id) の記録が既にある（上書きしない）"
            }
        }
    }
}
