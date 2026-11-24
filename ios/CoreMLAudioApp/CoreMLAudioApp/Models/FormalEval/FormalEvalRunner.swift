import CoreML
import Foundation
import UIKit

/// 正式評価（FAD・RTF）用の一括合成（設定は `FormalEvalConfig`）
///
/// 出力: `Documents/Result/formal_eval/<session>/`
///   - `session.json`: 条件（最初の起動で書く。再開のときは条件が同じことを確かめ、違えば何もせず終える）
///   - `launches.csv`: 起動ごとのモデルの読み込み時間など
///   - `results.csv`: サンプルごとの行（追記のみ）。status は ok / error / skipped_existing_output
///   - `<話者>/<参照>_<話者>_pronounse_ios.wav`: FAD 用（デエンファシス後、PCM16。±1 を超えた値は切られる）
///   - `<話者>/<参照>_<話者>_pronounse_ios_f32.npy`: クリッピングの解析用（デエンファシス後、PCM16 にする前の float32）
///
/// 再開と上書きの防止:
///   - `results.csv` に status=ok の行があり、出力の wav と npy がそろっているサンプルは合成しない
///   - 出力は一時ファイルに書いてから移す。移す先に同じ名前のファイルがあれば移さない（既存の出力は上書きしない）。
///     `results.csv` に記録の無い出力が既にあるときは、そのサンプルを skipped_existing_output として記録する
///
/// 時間は、前処理（読み込み・メル）、推論（Encoder・Decoder・HiFi-GAN の predict と、その間の配列の処理）、
/// 後処理（デエンファシス）、保存を分けて記録する。
enum FormalEvalRunner {

    static let csvHeader = [
        "finished_at", "launch_id", "speaker", "reference", "input_file", "output_wav", "output_f32", "status", "error",
        "input_samples", "T", "output_samples", "expected_samples", "preprocess_ms", "encoder_predict_ms",
        "decoder_predict_ms", "hifigan_predict_ms", "inference_wall_ms", "postprocess_ms", "save_ms",
        "postnet_nonfinite", "wave_nonfinite", "raw_peak", "wave_peak", "wave_rms", "clipped_samples",
    ]

    @MainActor
    static func runFromEnvironment() async -> Int32 {
        print("FORMAL_EVAL_START")
        do {
            let config = try FormalEvalConfig.fromEnvironment()
            let (ok, errors) = try await run(config: config)
            print("FORMAL_EVAL_DONE status=\(errors == 0 ? "ok" : "partial") ok=\(ok) errors=\(errors)")
            return errors == 0 ? 0 : 2
        } catch {
            print("FORMAL_EVAL_DONE status=failed error=\(error)")
            return 1
        }
    }

    /// 合成する。(このセッションで ok のサンプル数, この起動で失敗したサンプル数) を返す
    static func run(config: FormalEvalConfig) async throws -> (ok: Int, errors: Int) {
        let fm = FileManager.default
        let dir = config.sessionDirectory
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let launchID = ISO8601DateFormatter().string(from: Date())

        // 1. 条件（再開のときは同じ条件であることを確かめる）
        let models = [
            "encoder": FormalEvalConfig.shapeMode.transformerEncoderResourceName(for: config.precision),
            "decoder": FormalEvalConfig.shapeMode.transformerDecoderResourceName(for: config.precision),
            "hifigan": FormalEvalConfig.shapeMode.hifiganResourceName(for: config.precision) ?? "?",
        ]
        var condition: [String: Any] = [
            "precision": config.precision.rawValue, "compute_unit": config.computeUnit.rawValue,
            "shape_mode": FormalEvalConfig.shapeMode.rawValue,
            "waveform_preprocess": FormalEvalConfig.waveformPreprocess.rawValue,
            "models": models, "sample_rate": AudioFeatureExtractor.sampleRate, "hop_size": 256,
            "deemphasis": "y[n] = x[n] + 0.97 y[n-1] (float32)",
            "wav": "PCM16 via AVAudioFile (values beyond ±1 are clipped)", "f32": "deemphasized float32 before PCM16",
        ]
        // 実験用の設定は、使ったときだけ条件に入れる（使わないときの条件は従来と同じなので、既存のセッションを再開できる）
        if config.usesExperimentalOptions {
            condition["compute_units_by_model"] = Dictionary(uniqueKeysWithValues: ["encoder", "decoder", "hifigan"]
                .map({ ($0, config.computeUnit(for: $0).rawValue) }))
            condition["decoder_input"] = config.padDecoderInput ? "padT" : "grow"
            condition["reload_mb"] = config.reloadThresholdMB ?? 0
        }
        let sessionURL = dir.appendingPathComponent("session.json")
        if fm.fileExists(atPath: sessionURL.path) {
            let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: sessionURL)) as? [String: Any]
            guard let savedCondition = saved?["condition"] as? NSDictionary, savedCondition == condition as NSDictionary else {
                throw RunError.conditionMismatch(sessionURL.path)
            }
        } else {
            let device: [String: Any] = [
                "model": Self.hardwareModel(), "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            ]
            let payload: [String: Any] = ["condition": condition, "device": device, "created_at": launchID]
            try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]).write(to: sessionURL)
        }

        // 2. 入力の一覧と、済んでいるサンプル
        let (inputs, rejected) = try FormalEvalInput.list(root: FormalEvalConfig.inputRoot, config: config)
        for name in rejected { print("[FormalEval] 命名規則に合わない入力を飛ばす: \(name)") }
        let resultsURL = dir.appendingPathComponent("results.csv")
        let done = try Self.completedStems(resultsURL: resultsURL)
        // 前回の起動で書きかけた一時ファイルは消す（このモードが作るファイルだけ）
        if let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator where url.lastPathComponent.contains(".partial.") {
                try? fm.removeItem(at: url)
            }
        }

        // 3. モデルを読み込む（モデルごとの Compute Units は実験用の設定。既定は 3 つとも同じ）
        func loadModels() throws -> (models: [String: MLModel], ms: [String: Double]) {
            var loaded: [String: MLModel] = [:]
            var loadMs: [String: Double] = [:]
            for role in ["encoder", "decoder", "hifigan"] {
                guard let name = models[role], let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                    throw RunError.modelNotFound(models[role] ?? role)
                }
                let mlConfig = MLModelConfiguration()
                mlConfig.computeUnits = config.computeUnit(for: role).mlComputeUnits
                let t0 = CFAbsoluteTimeGetCurrent()
                loaded[role] = try MLModel(contentsOf: url, configuration: mlConfig)
                loadMs[role] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            }
            return (loaded, loadMs)
        }
        var (loaded, loadMs) = try loadModels()
        try Self.appendCSV(url: dir.appendingPathComponent("launches.csv"),
                           header: ["launch_id", "inputs_listed", "already_done", "rejected", "encoder_load_ms",
                                    "decoder_load_ms", "hifigan_load_ms"],
                           row: [launchID, "\(inputs.count)", "\(done.count)", "\(rejected.count)",
                                 Self.format(loadMs["encoder"]), Self.format(loadMs["decoder"]), Self.format(loadMs["hifigan"])])
        guard var encoder = loaded["encoder"], var decoder = loaded["decoder"], var hifigan = loaded["hifigan"] else {
            throw RunError.modelNotFound("?")
        }

        // 4. サンプルごとに合成する（失敗しても次へ進む）
        var okCount = done.count
        var errorCount = 0
        for (index, input) in inputs.enumerated() where !done.contains(input.stem) {
            let outDir = dir.appendingPathComponent(input.speaker, isDirectory: true)
            try fm.createDirectory(at: outDir, withIntermediateDirectories: true)
            let wavURL = outDir.appendingPathComponent("\(input.outputStem).wav")
            let npyURL = outDir.appendingPathComponent("\(input.outputStem)_f32.npy")
            var row: [String: String] = [
                "launch_id": launchID, "speaker": input.speaker, "reference": input.reference,
                "input_file": "\(input.speaker)/\(input.url.lastPathComponent)",
                "output_wav": "\(input.speaker)/\(wavURL.lastPathComponent)",
                "output_f32": "\(input.speaker)/\(npyURL.lastPathComponent)",
            ]
            if fm.fileExists(atPath: wavURL.path) || fm.fileExists(atPath: npyURL.path) {
                row["status"] = "skipped_existing_output"
                row["error"] = "results.csv に記録の無い出力が既にある（上書きしない）"
                try Self.appendResult(url: resultsURL, row: row)
                errorCount += 1
                continue
            }
            do {
                try await Self.synthesize(
                    input: input, encoder: encoder, decoder: decoder, hifigan: hifigan,
                    padDecoderInput: config.padDecoderInput, wavURL: wavURL, npyURL: npyURL, row: &row
                )
                row["status"] = "ok"
                okCount += 1
            } catch {
                row["status"] = "error"
                row["error"] = "\(error)"
                errorCount += 1
            }
            try Self.appendResult(url: resultsURL, row: row)
            // 実験用: メモリを記録し、上限を超えていたらモデルを読み込み直す
            if config.usesExperimentalOptions {
                let mb = Self.footprintMB()
                var reloadMs = ""
                if let limit = config.reloadThresholdMB, mb > limit {
                    loaded.removeAll()
                    (loaded, loadMs) = try loadModels()
                    if let e = loaded["encoder"], let d = loaded["decoder"], let h = loaded["hifigan"] {
                        (encoder, decoder, hifigan) = (e, d, h)
                    }
                    reloadMs = Self.format(loadMs.values.reduce(0, +))
                }
                let after = reloadMs.isEmpty ? nil : Self.footprintMB()
                try Self.appendCSV(url: dir.appendingPathComponent("memory.csv"),
                                   header: ["launch_id", "input", "T", "status", "footprint_mb", "reload_ms", "footprint_after_reload_mb"],
                                   row: [launchID, input.stem, row["T"] ?? "", row["status"] ?? "", Self.format(mb), reloadMs,
                                         after.map({ Self.format($0) }) ?? ""])
                // 読み込み直しても上限を下回らなければ、異常な増え方なので止める（ここまでの出力は残り、同じセッションで再開できる）
                if let limit = config.reloadThresholdMB, let after, after > limit {
                    throw RunError.memoryNotRecovered(afterReloadMB: after, limitMB: limit)
                }
            }
            print("[FormalEval] \(index + 1)/\(inputs.count) \(input.stem) \(row["status"] ?? "?") T=\(row["T"] ?? "-") "
                  + "infer=\(row["inference_wall_ms"] ?? "-")ms peak=\(row["wave_peak"] ?? "-") \(row["error"] ?? "")")
        }
        return (okCount, errorCount)
    }

    /// 1 件を合成し、wav と npy を保存して、行に時間と統計を書き込む
    private static func synthesize(
        input: FormalEvalInput, encoder: MLModel, decoder: MLModel, hifigan: MLModel, padDecoderInput: Bool,
        wavURL: URL, npyURL: URL, row: inout [String: String]
    ) async throws {
        let nMels = AudioFeatureExtractor.nMels
        let policy = FormalEvalConfig.shapeMode.transformerInputPolicy

        let tPre = CFAbsoluteTimeGetCurrent()
        let samples = try AudioFeatureExtractor.loadAudio(from: input.url)
        let front = AudioFeatureExtractor.extractMelSpectrogram(
            fromSamples: samples, waveformPreprocess: FormalEvalConfig.waveformPreprocess
        )
        row["preprocess_ms"] = format((CFAbsoluteTimeGetCurrent() - tPre) * 1000)
        let T = front.frameCount
        row["input_samples"] = "\(samples.count)"
        row["T"] = "\(T)"
        row["expected_samples"] = "\(T * 256)"

        let tInfer = CFAbsoluteTimeGetCurrent()
        let (memory, encMs, _) = try await EncoderRunner(model: encoder, policy: policy)
            .run(mel: front.mel, frameCount: T, nMels: nMels)
        // padT（実験用）: DecoderRunner の固定長の動作（入力を targetT にして残りを 0 で埋める）を、このサンプルの T で使う。
        // targetT = T なので切り詰めは起きず、memory もちょうど T のまま
        let decoderPolicy: TransformerInputPolicy = padDecoderInput ? .fixed(targetT: T) : policy
        let (postnet, _, decMs, _, _) = try await DecoderRunner(model: decoder, policy: decoderPolicy)
            .run(memory: memory, frameCount: T, nMels: nMels, onStep: { _ in }, collectStepStats: false)
        let vocoder = try await VocoderRunner(model: hifigan, policy: FormalEvalConfig.shapeMode.inputPolicy)
            .run(postnetOut: postnet, totalFrames: T, nMels: nMels)
        row["inference_wall_ms"] = format((CFAbsoluteTimeGetCurrent() - tInfer) * 1000)
        row["encoder_predict_ms"] = format(encMs)
        row["decoder_predict_ms"] = format(decMs)
        row["hifigan_predict_ms"] = format(vocoder.predictMs)

        let tPost = CFAbsoluteTimeGetCurrent()
        let wave = AudioFeatureExtractor.applyDeemphasis(vocoder.waveform)
        row["postprocess_ms"] = format((CFAbsoluteTimeGetCurrent() - tPost) * 1000)

        // 統計（クリッピングで隠れないよう、PCM16 にする前の値で数える）
        let postnetValues = MLMultiArrayReader.rows(postnet, rows: 0..<postnet.shape[1].intValue, columns: nMels)
        row["output_samples"] = "\(wave.count)"
        row["postnet_nonfinite"] = "\(postnetValues.filter({ !$0.isFinite }).count)"
        row["wave_nonfinite"] = "\(wave.filter({ !$0.isFinite }).count)"
        row["raw_peak"] = format(Double(vocoder.waveform.map({ abs($0) }).max() ?? 0))
        row["wave_peak"] = format(Double(wave.map({ abs($0) }).max() ?? 0))
        row["wave_rms"] = format((wave.reduce(0, { $0 + Double($1) * Double($1) }) / Double(max(wave.count, 1))).squareRoot())
        row["clipped_samples"] = "\(wave.filter({ abs($0) >= 1 }).count)"
        guard postnet.shape[1].intValue == T, wave.count == T * 256 else {
            throw RunError.unexpectedLength(T: T, samples: wave.count)
        }

        // 保存（一時ファイルに書いてから移す。移す先があれば失敗して、上書きしない）
        let tSave = CFAbsoluteTimeGetCurrent()
        let fm = FileManager.default
        let tmpWav = wavURL.deletingLastPathComponent().appendingPathComponent("\(input.outputStem).partial.wav")
        let tmpNpy = npyURL.deletingLastPathComponent().appendingPathComponent("\(input.outputStem)_f32.partial.npy")
        try NpyWriter.writeFloat32(wave, rows: 1, cols: wave.count, to: tmpNpy)
        try WavWriter.writeInt16(waveform: wave, sampleRate: AudioFeatureExtractor.sampleRate, to: tmpWav)
        try fm.moveItem(at: tmpNpy, to: npyURL)
        try fm.moveItem(at: tmpWav, to: wavURL)
        row["save_ms"] = format((CFAbsoluteTimeGetCurrent() - tSave) * 1000)
    }

    // MARK: - CSV

    /// results.csv で status=ok の入力（`<参照>_<話者>`）
    static func completedStems(resultsURL: URL) throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: resultsURL.path) else { return [] }
        let text = try String(contentsOf: resultsURL, encoding: .utf8)
        let lines = text.split(separator: "\n").map(String.init)
        guard let header = lines.first.map(parseCSVLine) else { return [] }
        guard let iStatus = header.firstIndex(of: "status"), let iSpeaker = header.firstIndex(of: "speaker"),
              let iReference = header.firstIndex(of: "reference") else { return [] }
        var stems = Set<String>()
        for line in lines.dropFirst() {
            let f = parseCSVLine(line)
            guard f.count == header.count, f[iStatus] == "ok" else { continue }
            stems.insert("\(f[iReference])_\(f[iSpeaker])")
        }
        return stems
    }

    private static func appendResult(url: URL, row: [String: String]) throws {
        var values = row
        values["finished_at"] = ISO8601DateFormatter().string(from: Date())
        try appendCSV(url: url, header: csvHeader, row: csvHeader.map({ values[$0] ?? "" }))
    }

    private static func appendCSV(url: URL, header: [String], row: [String]) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try (header.map(escape).joined(separator: ",") + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data((row.map(escape).joined(separator: ",") + "\n").utf8))
    }

    static func escape(_ s: String) -> String {
        guard s.contains(",") || s.contains("\"") || s.contains("\n") else { return s }
        return "\"" + s.replacingOccurrences(of: "\"", with: "\"\"").replacingOccurrences(of: "\n", with: " ") + "\""
    }

    static func parseCSVLine(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var inQuotes = false
        var chars = Array(line)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count, chars[i + 1] == "\"" { current.append("\""); i += 1 } else { inQuotes = false }
                } else {
                    current.append(c)
                }
            } else if c == "\"" {
                inQuotes = true
            } else if c == "," {
                fields.append(current)
                current = ""
            } else {
                current.append(c)
            }
            i += 1
        }
        fields.append(current)
        chars.removeAll()
        return fields
    }

    private static func format(_ value: Double?) -> String {
        guard let value else { return "" }
        return String(format: "%.3f", value)
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

    private static func hardwareModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine, { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        })
    }

    enum RunError: LocalizedError {
        case conditionMismatch(String)
        case modelNotFound(String)
        case unexpectedLength(T: Int, samples: Int)
        case memoryNotRecovered(afterReloadMB: Double, limitMB: Double)

        var errorDescription: String? {
            switch self {
            case .conditionMismatch(let path): return "既存のセッションと条件が違う（\(path)）。別のセッション ID を使う"
            case .modelNotFound(let name): return "\(name).mlmodelc がバンドルに無い"
            case .unexpectedLength(let T, let samples): return "出力の長さが T × 256 と違う（T=\(T)、\(samples) サンプル）"
            case .memoryNotRecovered(let after, let limit):
                return "モデルを読み込み直してもメモリが \(Int(after)) MB で上限 \(Int(limit)) MB を下回らないので止めた"
            }
        }
    }
}
