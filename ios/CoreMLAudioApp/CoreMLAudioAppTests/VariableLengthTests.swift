import CoreML
import Foundation
import Testing
@testable import CoreMLAudioApp

/// 可変長（RangeDim 1〜1000、`ShapeModeOption.range1`）のモデルで、入力の全フレームを切らずに合成できるかを実機で確かめる
///
/// 条件は Float32 × cpuOnly だけ（段階的な検証の最初の条件）。前処理は正式評価と同じ `.fullInput`
/// （オリジナルの合成音と同じく、先頭の無音を削らず、フェードしない）。
/// 入力は `VariableLengthInputs/`（評価データセットの口真似音から選んだもの。再配布しないので Git には入れない）:
///   - `<name>.wav`: 口真似音を librosa.load(sr=22050) で 22050 Hz にした float32 の wav
///     （48 kHz のまま渡すと AVAudioConverter の変換がオリジナルと合わない。`scripts/verify_fullinput_preprocess.py`）
///   - `<name>_mel_in.npy`: 同じ条件で PyTorch 側が作ったメル [T × 256]
///     （`scripts/verify_variable_length.py --preprocess full` が書き出したもの）
///   入力が無ければテストは何もせずに終わる。
///
/// 出力: `Documents/Result/variable_length/`
///   - `<name>_ios_postnet.npy` [T × 256]、`<name>_ios_wave.npy`（デエンファシス後）: PyTorch のメルを入れた経路
///   - `<name>_iosfront_mel.npy`、`<name>_iosfront_wave.npy`: iOS の前処理を通した経路
///   - `summary.json`
/// PyTorch 版との比較はホストで行う（`scripts/compare_variable_length_device.py`）。
struct VariableLengthTests {

    private static let names = ["s0p5", "s3", "s5", "s8p5"]
    private static let nMels = AudioFeatureExtractor.nMels
    private static let hopSize = 256
    private static let shapeMode = ShapeModeOption.range1
    private static let precision = ModelPrecision.float32
    private static let computeUnit = ComputeUnitOption.cpuOnly

    struct Pipeline {
        let encoder: MLModel
        let decoder: MLModel
        let hifigan: MLModel

        /// AudioSynthesizer.synthesize と同じ順に Runner をまわす
        func run(mel: [Float], frameCount: Int) async throws -> (postnet: MLMultiArray, wave: [Float], memoryT: Int, inputT: Int) {
            let policy = VariableLengthTests.shapeMode.transformerInputPolicy
            let (memory, _, _) = try await EncoderRunner(model: encoder, policy: policy)
                .run(mel: mel, frameCount: frameCount, nMels: VariableLengthTests.nMels)
            let (postnet, _, _, _, _) = try await DecoderRunner(model: decoder, policy: policy).run(
                memory: memory, frameCount: frameCount, nMels: VariableLengthTests.nMels,
                onStep: { _ in }, collectStepStats: false
            )
            let vocoder = try await VocoderRunner(model: hifigan, policy: VariableLengthTests.shapeMode.inputPolicy)
                .run(postnetOut: postnet, totalFrames: frameCount, nMels: VariableLengthTests.nMels)
            let wave = AudioFeatureExtractor.applyDeemphasis(vocoder.waveform)
            return (postnet, wave, memory.shape[1].intValue, vocoder.inputT)
        }
    }

    @Test func fullLengthFloat32CpuOnly() async throws {
        guard let inputDir = Self.inputDirectory() else {
            print("[VariableLengthTests] VariableLengthInputs/ が無いので何もしない")
            return
        }
        let outDir = try Self.outputDirectory()
        let pipeline = try Self.loadPipeline()
        var rows = [[String: Any]]()

        for name in Self.names {
            // A. PyTorch と同じ前処理のメルを入れる
            let melURL = inputDir.appendingPathComponent("\(name)_mel_in.npy")
            let mel = try MinimalNpyReader.readFloat32(contentsOf: melURL)
            let frameCount = mel.count / Self.nMels
            let t0 = CFAbsoluteTimeGetCurrent()
            let a = try await pipeline.run(mel: mel, frameCount: frameCount)
            let secA = CFAbsoluteTimeGetCurrent() - t0
            let postnet = MLMultiArrayReader.rows(a.postnet, rows: 0..<a.postnet.shape[1].intValue, columns: Self.nMels)
            try NpyWriter.writeFloat32(postnet, rows: a.postnet.shape[1].intValue, cols: Self.nMels,
                                       to: outDir.appendingPathComponent("\(name)_ios_postnet.npy"))
            try NpyWriter.writeFloat32(a.wave, rows: 1, cols: a.wave.count,
                                       to: outDir.appendingPathComponent("\(name)_ios_wave.npy"))

            // B. wav から iOS の前処理（正式評価の設定 `.fullInput`）を通す
            let wavURL = inputDir.appendingPathComponent("\(name).wav")
            let inputSamples = try AudioFeatureExtractor.loadAudio(from: wavURL).count
            let front = try AudioFeatureExtractor.extractMelSpectrogram(from: wavURL, waveformPreprocess: .fullInput)
            try NpyWriter.writeFloat32(front.mel, rows: front.frameCount, cols: Self.nMels,
                                       to: outDir.appendingPathComponent("\(name)_iosfront_mel.npy"))
            let b = try await pipeline.run(mel: front.mel, frameCount: front.frameCount)
            try NpyWriter.writeFloat32(b.wave, rows: 1, cols: b.wave.count,
                                       to: outDir.appendingPathComponent("\(name)_iosfront_wave.npy"))

            let nonFiniteA = a.wave.filter({ !$0.isFinite }).count + postnet.filter({ !$0.isFinite }).count
            let nonFiniteB = b.wave.filter({ !$0.isFinite }).count
            let peakA = a.wave.map({ abs($0) }).max() ?? 0
            let row: [String: Any] = [
                "name": name, "T": frameCount, "memory_T": a.memoryT, "postnet_T": a.postnet.shape[1].intValue,
                "hifigan_input_T": a.inputT, "samples": a.wave.count, "expected_samples": frameCount * Self.hopSize,
                "nonfinite": nonFiniteA, "peak": peakA, "clipped_samples": a.wave.filter({ abs($0) >= 1 }).count,
                "seconds": secA,
                "iosfront_T": front.frameCount, "iosfront_memory_T": b.memoryT, "iosfront_samples": b.wave.count,
                "iosfront_nonfinite": nonFiniteB, "input_samples": inputSamples,
                "waveform_preprocess": AudioFeatureExtractor.WaveformPreprocess.fullInput.rawValue,
            ]
            rows.append(row)
            print("[VariableLengthTests] \(row)")

            // 入力が途中で切られず、出力のサンプル数が T × 256 になる
            #expect(a.memoryT == frameCount)
            #expect(a.postnet.shape[1].intValue == frameCount)
            #expect(a.inputT == frameCount)
            #expect(a.wave.count == frameCount * Self.hopSize)
            #expect(nonFiniteA == 0)
            // 入力の全体を使う: フレーム数は 1 + floor(サンプル数 / 256)（オリジナルの合成音と同じ）で、PyTorch 側とも同じ
            #expect(front.frameCount == 1 + inputSamples / Self.hopSize)
            #expect(front.frameCount == frameCount)
            #expect(b.memoryT == front.frameCount)
            #expect(b.wave.count == front.frameCount * Self.hopSize)
            #expect(nonFiniteB == 0)
        }

        let summary: [String: Any] = [
            "precision": Self.precision.rawValue, "compute_unit": Self.computeUnit.rawValue,
            "shape_mode": Self.shapeMode.rawValue,
            "models": [Self.shapeMode.transformerEncoderResourceName(for: Self.precision),
                       Self.shapeMode.transformerDecoderResourceName(for: Self.precision),
                       Self.shapeMode.hifiganResourceName(for: Self.precision) ?? "?"],
            "rows": rows,
        ]
        let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("summary.json"))
    }

    // MARK: - 準備

    private static func loadPipeline() throws -> Pipeline {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit.mlComputeUnits
        func load(_ name: String?) throws -> MLModel {
            guard let name, let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw AudioSynthesizer.SynthesisError.modelNotFound(precision: name ?? "?")
            }
            return try MLModel(contentsOf: url, configuration: config)
        }
        return Pipeline(
            encoder: try load(shapeMode.transformerEncoderResourceName(for: precision)),
            decoder: try load(shapeMode.transformerDecoderResourceName(for: precision)),
            hifigan: try load(shapeMode.hifiganResourceName(for: precision))
        )
    }

    /// 入力のあるフォルダ（同期グループで平らにコピーされることもあるので、mel の npy の場所から決める）
    private static func inputDirectory() -> URL? {
        let bundle = Bundle(for: BundleToken.self)
        guard let root = bundle.resourceURL,
              let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return nil
        }
        for case let url as URL in enumerator where url.lastPathComponent == "\(names[0])_mel_in.npy" {
            return url.deletingLastPathComponent()
        }
        return nil
    }

    private static func outputDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result/variable_length", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

/// `Bundle(for:)` 用のトークン（テストバンドルを指す）
private final class BundleToken {}
