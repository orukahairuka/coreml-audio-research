import CoreML
import Foundation
import Testing
@testable import CoreMLAudioApp

/// 可変長（`range1`）× 精度 × Compute Units の実機動作確認（正式評価の前の確認）
///
/// 1 回の実行で 1 つの構成だけを試す（GPU の E5RT エラーや ANE の読み込みでプロセスごと落ちても、
/// 他の構成の結果に影響しないようにするため）。構成は起動時の環境変数で渡す
/// （xcodebuild では `TEST_RUNNER_` を付けて渡す。`scripts/run_variable_length_matrix.py` が行う）:
///   - `CMLA_VLM_PRECISION`: `Float32` / `Float16` / `Int8`（無ければこのテストは何もしない）
///   - `CMLA_VLM_CU`: `cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all`
///   - `CMLA_VLM_INPUTS`: 入力の名前をカンマ区切りで（既定 `s0p5`）
///
/// 入力と前処理は正式評価と同じ（`VariableLengthInputs/<name>.wav` は 22050 Hz の float32 の wav、前処理は `.fullInput`）。
///
/// 出力: `Documents/Result/variable_length_matrix/<precision>_<cu>/`
///   - `status.json`: 段階ごとに上書きする（途中でプロセスが落ちても、どこまで進んだかが残る）
///   - `compute_plan/*.json`: MLComputePlan による演算ごとの割り当て（予定。実際に使われた装置そのものではない）
///   - `<name>_postnet.npy`、`<name>_hifigan_raw.npy`（デエンファシス前）、`<name>_wave.npy`（デエンファシス後、PCM16 にする前）
struct VariableLengthMatrixTests {

    private static let nMels = AudioFeatureExtractor.nMels
    private static let hopSize = 256
    private static let shapeMode = ShapeModeOption.range1

    @Test func runConfiguredCondition() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let precisionValue = env["CMLA_VLM_PRECISION"] else {
            print("[VariableLengthMatrixTests] CMLA_VLM_PRECISION が無いので何もしない")
            return
        }
        let precision = try #require(ModelPrecision(rawValue: precisionValue))
        let computeUnit = try #require(ComputeUnitOption(rawValue: env["CMLA_VLM_CU"] ?? "cpuOnly"))
        let names = (env["CMLA_VLM_INPUTS"] ?? "s0p5").split(separator: ",").map(String.init)
        let inputDir = try #require(Self.inputDirectory(), "VariableLengthInputs/ がテストバンドルに無い")

        let outDir = try Self.outputDirectory(label: "\(precision.rawValue)_\(computeUnit.rawValue)")
        var status: [String: Any] = [
            "precision": precision.rawValue, "compute_unit": computeUnit.rawValue, "shape_mode": Self.shapeMode.rawValue,
            "waveform_preprocess": AudioFeatureExtractor.WaveformPreprocess.fullInput.rawValue,
            "inputs": names, "phase": "start", "memory_start_mb": Self.footprintMB(),
        ]
        try Self.write(status, to: outDir)

        let resourceNames = [
            ("encoder", Self.shapeMode.transformerEncoderResourceName(for: precision)),
            ("decoder", Self.shapeMode.transformerDecoderResourceName(for: precision)),
            ("hifigan", Self.shapeMode.hifiganResourceName(for: precision) ?? "?"),
        ]
        status["models"] = Dictionary(uniqueKeysWithValues: resourceNames.map({ ($0.0, $0.1) }))

        // 1. MLComputePlan（読み込みの前に、演算ごとの割り当ての予定を記録する）
        var plans: [String: String] = [:]
        for (role, name) in resourceNames {
            status["phase"] = "plan_\(role)"
            try Self.write(status, to: outDir)
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                plans[role] = "model not found"
                continue
            }
            do {
                plans[role] = try await ComputePlanInspector(
                    modelURL: url, modelName: name, precision: precision, computeUnits: computeUnit,
                    outputDirectory: outDir.appendingPathComponent("compute_plan", isDirectory: true)
                ).inspectAndWrite()
            } catch {
                plans[role] = "error: \(error)"
            }
        }
        status["compute_plan"] = plans

        // 2. 読み込み（1 つずつ。失敗したらそこで記録して終える）
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit.mlComputeUnits
        var models: [String: MLModel] = [:]
        var loadMs: [String: Double] = [:]
        for (role, name) in resourceNames {
            status["phase"] = "load_\(role)"
            try Self.write(status, to: outDir)
            do {
                let url = try #require(Bundle.main.url(forResource: name, withExtension: "mlmodelc"))
                let t0 = CFAbsoluteTimeGetCurrent()
                models[role] = try MLModel(contentsOf: url, configuration: config)
                loadMs[role] = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            } catch {
                status["load_ms"] = loadMs
                status["phase"] = "load_failed_\(role)"
                status["error"] = "\(error)"
                try Self.write(status, to: outDir)
                Issue.record("読み込みに失敗: \(role) \(error)")
                return
            }
        }
        status["load_ms"] = loadMs
        status["memory_after_load_mb"] = Self.footprintMB()

        // 3. 推論（入力ごと）
        let policy = Self.shapeMode.transformerInputPolicy
        var rows: [[String: Any]] = []
        for name in names {
            status["phase"] = "infer_\(name)"
            status["rows"] = rows
            try Self.write(status, to: outDir)
            var row: [String: Any] = ["name": name]
            do {
                let wavURL = inputDir.appendingPathComponent("\(name).wav")
                let tPre = CFAbsoluteTimeGetCurrent()
                let samples = try AudioFeatureExtractor.loadAudio(from: wavURL)
                let front = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples, waveformPreprocess: .fullInput)
                row["preprocess_ms"] = (CFAbsoluteTimeGetCurrent() - tPre) * 1000
                let T = front.frameCount
                row["T"] = T
                row["input_samples"] = samples.count

                let tInfer = CFAbsoluteTimeGetCurrent()
                let (memory, encMs, _) = try await EncoderRunner(model: try #require(models["encoder"]), policy: policy)
                    .run(mel: front.mel, frameCount: T, nMels: Self.nMels)
                let (postnet, _, decMs, _, _) = try await DecoderRunner(model: try #require(models["decoder"]), policy: policy)
                    .run(memory: memory, frameCount: T, nMels: Self.nMels, onStep: { _ in }, collectStepStats: false)
                let vocoder = try await VocoderRunner(model: try #require(models["hifigan"]), policy: Self.shapeMode.inputPolicy)
                    .run(postnetOut: postnet, totalFrames: T, nMels: Self.nMels)
                row["inference_wall_ms"] = (CFAbsoluteTimeGetCurrent() - tInfer) * 1000
                row["encoder_predict_ms"] = encMs
                row["decoder_predict_ms"] = decMs
                row["hifigan_predict_ms"] = vocoder.predictMs
                let tPost = CFAbsoluteTimeGetCurrent()
                let wave = AudioFeatureExtractor.applyDeemphasis(vocoder.waveform)
                row["postprocess_ms"] = (CFAbsoluteTimeGetCurrent() - tPost) * 1000
                row["memory_after_infer_mb"] = Self.footprintMB()

                let postnetT = postnet.shape[1].intValue
                let postnetValues = MLMultiArrayReader.rows(postnet, rows: 0..<postnetT, columns: Self.nMels)
                row["memory_T"] = memory.shape[1].intValue
                row["postnet_T"] = postnetT
                row["samples"] = wave.count
                row["expected_samples"] = T * Self.hopSize
                row["postnet_nonfinite"] = postnetValues.filter({ !$0.isFinite }).count
                row["wave_nonfinite"] = wave.filter({ !$0.isFinite }).count
                row["raw_peak"] = vocoder.waveform.map({ abs($0) }).max() ?? 0
                row["wave_peak"] = wave.map({ abs($0) }).max() ?? 0
                row["wave_rms"] = (wave.reduce(0, { $0 + Double($1) * Double($1) }) / Double(max(wave.count, 1))).squareRoot()
                row["wave_clipped_samples"] = wave.filter({ abs($0) >= 1 }).count
                row["status"] = "ok"
                try NpyWriter.writeFloat32(postnetValues, rows: postnetT, cols: Self.nMels,
                                           to: outDir.appendingPathComponent("\(name)_postnet.npy"))
                try NpyWriter.writeFloat32(vocoder.waveform, rows: 1, cols: vocoder.waveform.count,
                                           to: outDir.appendingPathComponent("\(name)_hifigan_raw.npy"))
                try NpyWriter.writeFloat32(wave, rows: 1, cols: wave.count,
                                           to: outDir.appendingPathComponent("\(name)_wave.npy"))
            } catch {
                row["status"] = "error"
                row["error"] = "\(error)"
                Issue.record("推論に失敗: \(name) \(error)")
            }
            rows.append(row)
            print("[VariableLengthMatrixTests] \(precision.rawValue)_\(computeUnit.rawValue) \(row)")
        }
        status["rows"] = rows
        status["phase"] = "done"
        try Self.write(status, to: outDir)
    }

    // MARK: - 補助

    private static func write(_ status: [String: Any], to dir: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("status.json"), options: .atomic)
    }

    /// プロセスのメモリ使用量（phys_footprint、MB）
    private static func footprintMB() -> Double {
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
        let bundle = Bundle(for: MatrixBundleToken.self)
        guard let root = bundle.resourceURL,
              let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return nil
        }
        for case let url as URL in enumerator where url.lastPathComponent == "s0p5.wav" {
            return url.deletingLastPathComponent()
        }
        return nil
    }

    /// 出力先。前回の同じ構成の結果は残さない（構成ごとのフォルダを作り直す）
    private static func outputDirectory(label: String) throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result/variable_length_matrix/\(label)", isDirectory: true)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

/// `Bundle(for:)` 用のトークン（テストバンドルを指す）
private final class MatrixBundleToken {}
