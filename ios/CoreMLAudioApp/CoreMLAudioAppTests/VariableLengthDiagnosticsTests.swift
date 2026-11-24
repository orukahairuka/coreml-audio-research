import CoreML
import Foundation
import Testing
@testable import CoreMLAudioApp

/// range1 の GPU・ANE 実行の問題を切り分けるための実機テスト（実験用。アプリと正式評価の動作は変えない）
///
/// 1 回の実行で 1 つの構成を試す。構成は起動時の環境変数で渡す（xcodebuild では `TEST_RUNNER_` を付ける。
/// `scripts/run_variable_length_matrix.py --test diagnostics` が行う）。`CMLA_VLD_LABEL` が無ければ何もしない。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_VLD_LABEL` | 構成の名前（出力のフォルダ名） |
/// | `CMLA_VLD_PRECISION` | `Float32` / `Float16` / `Int8` |
/// | `CMLA_VLD_CU_ENC` / `_DEC` / `_HIFI` | モデルごとの Compute Units（`cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all`） |
/// | `CMLA_VLD_HIFI_SHAPE` | HiFi-GAN のモデル（`range1` 既定 / `range16`。どちらも同梱済みの既存モデル） |
/// | `CMLA_VLD_RESHAPE` | `MLOptimizationHints.reshapeFrequency`（`default` 既定 / `frequent` / `infrequent`） |
/// | `CMLA_VLD_DEC_INPUT` | Decoder 入力の形。`grow`（既定。1 フレームずつ伸ばす）/ `padT`（毎回 T フレームにして残りを 0 で埋める） |
/// | `CMLA_VLD_AUTORELEASE` | `1` で Decoder の predict を 1 回ずつ autoreleasepool で囲む |
/// | `CMLA_VLD_INPUTS` | 入力の名前（既定 `s0p5`） |
/// | `CMLA_VLD_PASSES` | 同じ入力の列を何回通すか（既定 2。1 回目が初回、2 回目がウォームアップ後） |
/// | `CMLA_VLD_MEM_LIMIT_MB` | メモリ使用量（phys_footprint）がこれを超えたらその入力の Decoder を止める（既定 1500） |
/// | `CMLA_VLD_SWEEP` | 数を指定すると、通常の推論の代わりに T = 48〜735 をその数だけ流し、形ごとのメモリの増え方を記録する |
/// | `CMLA_VLD_SWEEP_MODELS` / `_START` / `_END` | 掃引で動かすモデル（`+` 区切り、既定は 3 つとも）と T の範囲（既定 48〜735） |
///
/// 出力: `Documents/Result/variable_length_diag/<label>/`
///   `status.json`（段階ごとに上書き）、`compute_plan/*.json`、`<name>_postnet.npy`・`<name>_wave.npy`（1 回目の結果）
struct VariableLengthDiagnosticsTests {

    private static let nMels = AudioFeatureExtractor.nMels

    struct Options {
        let label: String
        let precision: ModelPrecision
        let units: [String: ComputeUnitOption]
        let hifiShape: ShapeModeOption
        let reshape: String
        let padDecoderInput: Bool
        let autorelease: Bool
        let names: [String]
        let passes: Int
        let memLimitMB: Double
    }

    enum Stop: Error {
        case memoryLimit(step: Int, mb: Double)
    }

    @Test func runDiagnosticCondition() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let label = env["CMLA_VLD_LABEL"] else {
            print("[VariableLengthDiagnosticsTests] CMLA_VLD_LABEL が無いので何もしない")
            return
        }
        func unit(_ key: String) throws -> ComputeUnitOption {
            try #require(ComputeUnitOption(rawValue: env[key] ?? "cpuOnly"), "\(key) が不正")
        }
        let options = Options(
            label: label,
            precision: try #require(ModelPrecision(rawValue: env["CMLA_VLD_PRECISION"] ?? "Float32")),
            units: ["encoder": try unit("CMLA_VLD_CU_ENC"), "decoder": try unit("CMLA_VLD_CU_DEC"),
                    "hifigan": try unit("CMLA_VLD_CU_HIFI")],
            hifiShape: try #require(ShapeModeOption(rawValue: env["CMLA_VLD_HIFI_SHAPE"] ?? "range1")),
            reshape: env["CMLA_VLD_RESHAPE"] ?? "default",
            padDecoderInput: (env["CMLA_VLD_DEC_INPUT"] ?? "grow") == "padT",
            autorelease: env["CMLA_VLD_AUTORELEASE"] == "1",
            names: (env["CMLA_VLD_INPUTS"] ?? "s0p5").split(separator: ",").map(String.init),
            passes: Int(env["CMLA_VLD_PASSES"] ?? "2") ?? 2,
            memLimitMB: Double(env["CMLA_VLD_MEM_LIMIT_MB"] ?? "1500") ?? 1500
        )
        try await Self.run(options)
    }

    private static func run(_ o: Options) async throws {
        let inputDir = try #require(Self.inputDirectory(), "VariableLengthInputs/ がテストバンドルに無い")
        let outDir = try Self.outputDirectory(label: o.label)
        let names = [
            "encoder": ShapeModeOption.range1.transformerEncoderResourceName(for: o.precision),
            "decoder": ShapeModeOption.range1.transformerDecoderResourceName(for: o.precision),
            "hifigan": o.hifiShape.hifiganResourceName(for: o.precision) ?? "?",
        ]
        var status: [String: Any] = [
            "label": o.label, "precision": o.precision.rawValue,
            "compute_units": o.units.mapValues({ $0.rawValue }), "models": names, "reshape_frequency": o.reshape,
            "decoder_input": o.padDecoderInput ? "padT" : "grow", "autorelease": o.autorelease,
            "inputs": o.names, "passes": o.passes, "mem_limit_mb": o.memLimitMB,
            "phase": "start", "memory_start_mb": footprintMB(),
        ]
        try write(status, to: outDir)

        // 1. MLComputePlan（予定。実際の実行先ではない）
        var plans: [String: String] = [:]
        for role in ["encoder", "decoder", "hifigan"] {
            guard let name = names[role], let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
                  let cu = o.units[role] else { continue }
            status["phase"] = "plan_\(role)"
            try write(status, to: outDir)
            do {
                plans[role] = try await ComputePlanInspector(
                    modelURL: url, modelName: name, precision: o.precision, computeUnits: cu,
                    outputDirectory: outDir.appendingPathComponent("compute_plan", isDirectory: true)
                ).inspectAndWrite()
            } catch {
                plans[role] = "error: \(error)"
            }
        }
        status["compute_plan"] = plans

        // 2. 読み込み
        var models: [String: MLModel] = [:]
        var loadMs: [String: Double] = [:]
        for role in ["encoder", "decoder", "hifigan"] {
            status["phase"] = "load_\(role)"
            try write(status, to: outDir)
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                models[role] = try loadModel(name: names[role] ?? "?", unit: o.units[role] ?? .cpuOnly, reshape: o.reshape)
                loadMs[role] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            } catch {
                status["load_ms"] = loadMs
                status["phase"] = "load_failed_\(role)"
                status["error"] = "\(error)"
                try write(status, to: outDir)
                return
            }
        }
        status["load_ms"] = loadMs
        status["memory_after_load_mb"] = footprintMB()

        // 3'. 形の掃引（CMLA_VLD_SWEEP=<数>）: 異なる T を次々に流し、形ごとにメモリが増えるかを見る
        if let count = Int(ProcessInfo.processInfo.environment["CMLA_VLD_SWEEP"] ?? ""), count > 0 {
            status["sweep"] = try await sweep(count: count, inputDir: inputDir, models: models, options: o, status: &status,
                                              outDir: outDir)
            // モデルを解放したら、形ごとに増えたメモリが戻るか
            models.removeAll()
            status["memory_after_release_mb"] = footprintMB()
            status["phase"] = "done"
            try write(status, to: outDir)
            return
        }

        // 3. 推論（同じ入力の列を passes 回）
        var rows: [[String: Any]] = []
        for pass in 1...max(o.passes, 1) {
            for name in o.names {
                status["phase"] = "infer_pass\(pass)_\(name)"
                status["rows"] = rows
                try write(status, to: outDir)
                var row: [String: Any] = ["name": name, "pass": pass]
                do {
                    try await infer(name: name, inputDir: inputDir, models: models, options: o,
                                    save: pass == 1 ? outDir : nil, row: &row)
                    row["status"] = "ok"
                } catch Stop.memoryLimit(let step, let mb) {
                    row["status"] = "aborted_memory_limit"
                    row["error"] = "Decoder の step \(step) でメモリ \(Int(mb)) MB が上限 \(Int(o.memLimitMB)) MB を超えた"
                } catch {
                    row["status"] = "error"
                    row["error"] = "\(error)"
                }
                rows.append(row)
                print("[VariableLengthDiagnosticsTests] \(o.label) pass\(pass) \(name) \(row["status"] ?? "?") "
                      + "enc=\(row["encoder_ms"] ?? "-") dec=\(row["decoder_ms"] ?? "-") hifi=\(row["hifigan_ms"] ?? "-") "
                      + "mem=\(row["memory_after_mb"] ?? "-") \(row["error"] ?? "")")
                // 上限を超えたら、これ以上は進めない（プロセスが落ちるのを避ける）
                if row["status"] as? String == "aborted_memory_limit" { break }
            }
            if rows.last?["status"] as? String == "aborted_memory_limit" { break }
        }
        status["rows"] = rows

        // 4. モデルを解放したときと、Decoder を読み込み直したときのメモリ
        status["phase"] = "release"
        try write(status, to: outDir)
        models.removeAll()
        status["memory_after_release_mb"] = footprintMB()
        if let first = o.names.first, rows.contains(where: { $0["status"] as? String == "ok" }) {
            var reloaded: [String: MLModel] = [:]
            let t0 = CFAbsoluteTimeGetCurrent()
            for role in ["encoder", "decoder", "hifigan"] {
                reloaded[role] = try loadModel(name: names[role] ?? "?", unit: o.units[role] ?? .cpuOnly, reshape: o.reshape)
            }
            var row: [String: Any] = ["name": first, "pass": "reloaded"]
            row["reload_ms"] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            do {
                try await infer(name: first, inputDir: inputDir, models: reloaded, options: o, save: nil, row: &row)
                row["status"] = "ok"
            } catch {
                row["status"] = "error"
                row["error"] = "\(error)"
            }
            status["reloaded_row"] = row
        }
        status["phase"] = "done"
        try write(status, to: outDir)
    }

    // MARK: - 形の掃引

    /// T = 48〜735 の count 通りについて、Encoder（メルの先頭 T フレーム）、Decoder（入力を T フレームにして 2 ステップ）、
    /// HiFi-GAN（同じメルを入力の代わりに使う）を 1 回ずつ動かし、そのたびにメモリを記録する。
    /// 出力の値は使わない（形ごとの特殊化やキャッシュでメモリが増えるかだけを見る）
    private static func sweep(
        count: Int, inputDir: URL, models: [String: MLModel], options o: Options, status: inout [String: Any], outDir: URL
    ) async throws -> [[Double]] {
        let samples = try AudioFeatureExtractor.loadAudio(from: inputDir.appendingPathComponent("s8p5.wav"))
        let full = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples, waveformPreprocess: .fullInput)
        let env = ProcessInfo.processInfo.environment
        // 動かすモデル（切り分け用。既定は 3 つとも）と、T の開始値（既定 48）
        let use = Set((env["CMLA_VLD_SWEEP_MODELS"] ?? "encoder+decoder+hifigan").split(whereSeparator: { $0 == "+" || $0 == "," }).map(String.init))
        let start = Int(env["CMLA_VLD_SWEEP_START"] ?? "48") ?? 48
        let end = min(Int(env["CMLA_VLD_SWEEP_END"] ?? "") ?? full.frameCount, full.frameCount)
        let lengths = (0..<count).map({ start + ($0 * (end - start)) / max(count - 1, 1) })
        status["sweep_models"] = Array(use).sorted()
        let encoder = try #require(models["encoder"])
        let decoder = try #require(models["decoder"])
        let hifigan = try #require(models["hifigan"])
        var rows: [[Double]] = []  // [T, 後のメモリ（Encoder・Decoder・HiFi-GAN）, 時間（ms）]
        for T in lengths {
            let mel = Array(full.mel.prefix(T * nMels))
            // どのモデルで落ちたか分かるよう、呼ぶ前に段階を書いておく
            status["phase"] = "sweep_T\(T)_encoder"
            try write(status, to: outDir)
            var t0 = CFAbsoluteTimeGetCurrent()
            let memory = use.contains("encoder")
                ? try await EncoderRunner(model: encoder, policy: .dynamic(maxT: 1000)).run(mel: mel, frameCount: T, nMels: nMels).memory
                : try ModelInputArray.zeros(shape: [1, T, 512], dataType: .float32)
            let encMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let memEnc = footprintMB()
            status["phase"] = "sweep_T\(T)_decoder"
            try write(status, to: outDir)
            t0 = CFAbsoluteTimeGetCurrent()
            for step in 0..<(use.contains("decoder") ? 2 : 0) {
                let input = try ModelInputArray.zeros(shape: [1, T, nMels], dataType: .float32)
                try ModelInputArray.copyFloat32(mel, count: (step + 1) * nMels, into: input, name: "decoder_input")
                let pos = try MLMultiArray(shape: [1, T as NSNumber], dataType: .int32)
                for i in 0..<T { pos[i] = NSNumber(value: Int32(i + 1)) }
                _ = try await decoder.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
                    "memory": MLFeatureValue(multiArray: memory), "decoder_input": MLFeatureValue(multiArray: input),
                    "pos": MLFeatureValue(multiArray: pos),
                ]))
            }
            let decMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let memDec = footprintMB()
            let melArray = try ModelInputArray.zeros(shape: [1, T, nMels], dataType: .float32)
            try ModelInputArray.copyFloat32(mel, count: mel.count, into: melArray, name: "mel")
            status["phase"] = "sweep_T\(T)_hifigan"
            try write(status, to: outDir)
            t0 = CFAbsoluteTimeGetCurrent()
            if use.contains("hifigan") {
                _ = try await VocoderRunner(model: hifigan, policy: .dynamic(maxT: 1000))
                    .run(postnetOut: melArray, totalFrames: T, nMels: nMels)
            }
            let hifiMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let memHifi = footprintMB()
            rows.append([Double(T), memEnc, memDec, memHifi, encMs, decMs, hifiMs])
            status["sweep"] = rows
            try write(status, to: outDir)
            if memHifi > o.memLimitMB {
                status["sweep_stopped"] = "T=\(T) でメモリ \(Int(memHifi)) MB が上限を超えた"
                break
            }
        }
        return rows
    }

    // MARK: - 推論

    private static func infer(
        name: String, inputDir: URL, models: [String: MLModel], options o: Options, save: URL?,
        row: inout [String: Any]
    ) async throws {
        let wavURL = inputDir.appendingPathComponent("\(name).wav")
        let samples = try AudioFeatureExtractor.loadAudio(from: wavURL)
        let front = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples, waveformPreprocess: .fullInput)
        let T = front.frameCount
        row["T"] = T
        row["memory_before_mb"] = footprintMB()

        var t0 = CFAbsoluteTimeGetCurrent()
        let (memory, _, _) = try await EncoderRunner(model: try #require(models["encoder"]), policy: .dynamic(maxT: 1000))
            .run(mel: front.mel, frameCount: T, nMels: nMels)
        row["encoder_ms"] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        row["memory_after_encoder_mb"] = footprintMB()

        t0 = CFAbsoluteTimeGetCurrent()
        var trace: [[Double]] = []
        let postnet = try decode(model: try #require(models["decoder"]), memory: memory, T: T, options: o, trace: &trace)
        row["decoder_ms"] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        row["decoder_memory_trace"] = trace  // [step, MB]
        row["memory_after_decoder_mb"] = footprintMB()

        let postnetArray = try ModelInputArray.zeros(shape: [1, T, nMels], dataType: .float32)
        try ModelInputArray.copyFloat32(postnet, count: postnet.count, into: postnetArray, name: "postnet")
        t0 = CFAbsoluteTimeGetCurrent()
        let vocoder = try await VocoderRunner(model: try #require(models["hifigan"]), policy: .dynamic(maxT: 1000))
            .run(postnetOut: postnetArray, totalFrames: T, nMels: nMels)
        row["hifigan_ms"] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        row["memory_after_mb"] = footprintMB()

        let wave = AudioFeatureExtractor.applyDeemphasis(vocoder.waveform)
        row["samples"] = wave.count
        row["expected_samples"] = T * 256
        row["postnet_nonfinite"] = postnet.filter({ !$0.isFinite }).count
        row["wave_nonfinite"] = wave.filter({ !$0.isFinite }).count
        row["wave_peak"] = wave.map({ abs($0) }).max() ?? 0
        row["wave_clipped_samples"] = wave.filter({ abs($0) >= 1 }).count
        if let dir = save {
            try NpyWriter.writeFloat32(postnet, rows: T, cols: nMels, to: dir.appendingPathComponent("\(name)_postnet.npy"))
            try NpyWriter.writeFloat32(wave, rows: 1, cols: wave.count, to: dir.appendingPathComponent("\(name)_wave.npy"))
        }
    }

    /// synthesis.py と同じ自己回帰（0 の 1 フレームから始め、mel_out の最後のフレームを足して T 回）。
    /// padT では入力を毎回 T フレームにし、生成済みより後ろを 0 で埋める（因果マスクにより mel_out の生成済みの部分は同じ。
    /// 最後のステップは T フレームちょうどなので postnet も同じ）。postnet [T × nMels] を返す
    private static func decode(
        model: MLModel, memory: MLMultiArray, T: Int, options o: Options, trace: inout [[Double]]
    ) throws -> [Float] {
        var frames = [Float](repeating: 0, count: nMels)
        var postnet: [Float] = []
        for step in 0..<T {
            let length = step + 1
            let inputLength = o.padDecoderInput ? T : length
            let input = try ModelInputArray.zeros(shape: [1, inputLength, nMels], dataType: .float32)
            try ModelInputArray.copyFloat32(frames, count: length * nMels, into: input, name: "decoder_input")
            let pos = try MLMultiArray(shape: [1, inputLength as NSNumber], dataType: .int32)
            for i in 0..<inputLength { pos[i] = NSNumber(value: Int32(i + 1)) }
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "memory": MLFeatureValue(multiArray: memory),
                "decoder_input": MLFeatureValue(multiArray: input),
                "pos": MLFeatureValue(multiArray: pos),
            ])
            let readStep = { (output: MLFeatureProvider) throws -> Void in
                guard let melOut = output.featureValue(for: "mel_out")?.multiArrayValue else {
                    throw AudioSynthesizer.SynthesisError.decoderFailed
                }
                frames += MLMultiArrayReader.rows(melOut, rows: (length - 1)..<length, columns: nMels)
                if step == T - 1 {
                    let key = output.featureNames.first(where: { $0 != "mel_out" }) ?? ""
                    guard let post = output.featureValue(for: key)?.multiArrayValue else {
                        throw AudioSynthesizer.SynthesisError.decoderFailed
                    }
                    postnet = MLMultiArrayReader.rows(post, rows: 0..<T, columns: nMels)
                }
            }
            if o.autorelease {
                try autoreleasepool(invoking: { try readStep(try model.prediction(from: provider)) })
            } else {
                try readStep(try model.prediction(from: provider))
            }
            if step % 16 == 0 || step == T - 1 {
                let mb = footprintMB()
                trace.append([Double(step), mb])
                if mb > o.memLimitMB { throw Stop.memoryLimit(step: step, mb: mb) }
            }
        }
        return postnet
    }

    // MARK: - 補助

    private static func loadModel(name: String, unit: ComputeUnitOption, reshape: String) throws -> MLModel {
        let url = try #require(Bundle.main.url(forResource: name, withExtension: "mlmodelc"))
        let config = MLModelConfiguration()
        config.computeUnits = unit.mlComputeUnits
        switch reshape {
        case "frequent": config.optimizationHints.reshapeFrequency = .frequent
        case "infrequent": config.optimizationHints.reshapeFrequency = .infrequent
        default: break
        }
        return try MLModel(contentsOf: url, configuration: config)
    }

    private static func write(_ status: [String: Any], to dir: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("status.json"), options: .atomic)
    }

    /// プロセスのメモリ使用量（phys_footprint、MB）
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info, { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count), { reboundPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), reboundPointer, &count)
            })
        })
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    private static func inputDirectory() -> URL? {
        let bundle = Bundle(for: DiagnosticsBundleToken.self)
        guard let root = bundle.resourceURL,
              let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return nil
        }
        for case let url as URL in enumerator where url.lastPathComponent == "s0p5.wav" {
            return url.deletingLastPathComponent()
        }
        return nil
    }

    private static func outputDirectory(label: String) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result/variable_length_diag/\(label)", isDirectory: true)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

/// `Bundle(for:)` 用のトークン（テストバンドルを指す）
private final class DiagnosticsBundleToken {}
