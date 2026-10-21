import CoreML
import Foundation

/// 卒論第4章の計測用の合成経路
///
/// `AudioSynthesizer.synthesize` と同じ Runner（Encoder / Decoder / Vocoder）を使うが、
/// 表示用のメル（STFT をもう 1 回行う）や UI の進捗表示は行わない。
/// End-to-End 時間は「入力ファイルの読み込みと前処理の開始」から
/// 「デエンファシスを適用した波形が得られるまで」とする（npy / wav の保存は含めない）。
final class ThesisPipeline {

    /// 1 回の合成の計測結果
    struct Measurement {
        let totalFrames: Int
        let effectiveFrames: Int
        /// Encoder に入れたメル [T' × nMels]
        let melIn: [Float]
        let preprocessMs: Double
        let encoderMs: Double
        let encoderInputNonFinite: Int
        let decoderTotalMs: Double
        let decoderStepMs: [Double]
        let decoderInputNonFiniteMax: Int
        /// Decoder（postnet）の出力メル [T' × nMels]
        let decoderMel: [Float]
        let hifiganMs: Double
        let hifiganInputNonFinite: Int
        /// HiFi-GAN の出力（デエンファシス前）
        let hifiganWaveform: [Float]
        let deemphMs: Double
        /// デエンファシス後の波形（保存前）
        let deemphWaveform: [Float]
        let e2eMs: Double

        var synthMs: Double { encoderMs + decoderTotalMs + hifiganMs }
    }

    static let nMels = AudioFeatureExtractor.nMels

    private let encoder: MLModel
    private let decoder: MLModel
    private let hifigan: MLModel
    private let transformerPolicy: TransformerInputPolicy
    private let vocoderPolicy: VocoderInputPolicy

    init(encoder: MLModel, decoder: MLModel, hifigan: MLModel) {
        self.encoder = encoder
        self.decoder = decoder
        self.hifigan = hifigan
        self.transformerPolicy = ThesisBenchConfig.shapeMode.transformerInputPolicy
        self.vocoderPolicy = ThesisBenchConfig.shapeMode.inputPolicy
    }

    // MARK: - モデル

    struct ModelURLs {
        let encoder: URL
        let decoder: URL
        let hifigan: URL
    }

    enum PipelineError: LocalizedError {
        case modelNotFound(String)
        case inputNotFound(String)
        case outputMissing

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let name): return "\(name).mlmodelc がバンドルにありません"
            case .inputNotFound(let name): return "入力 \(name) がバンドルにありません"
            case .outputMissing: return "モデルの出力を取得できませんでした"
            }
        }
    }

    /// fixed262 の 3 モデルの URL
    static func modelURLs(precision: ModelPrecision) throws -> ModelURLs {
        let shape = ThesisBenchConfig.shapeMode
        let encoderName = shape.transformerEncoderResourceName(for: precision)
        let decoderName = shape.transformerDecoderResourceName(for: precision)
        let hifiganName = shape.hifiganResourceName(for: precision) ?? "HiFiGAN_Generator_\(precision.suffix)_fixed262"
        func url(_ name: String) throws -> URL {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw PipelineError.modelNotFound(name)
            }
            return url
        }
        return ModelURLs(encoder: try url(encoderName), decoder: try url(decoderName), hifigan: try url(hifiganName))
    }

    static func configuration(computeUnit: ComputeUnitOption) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit.mlComputeUnits
        return config
    }

    /// モデルを 1 つ読み込み、所要時間 (ms) を返す
    static func loadModel(at url: URL, computeUnit: ComputeUnitOption) throws -> (model: MLModel, ms: Double) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try MLModel(contentsOf: url, configuration: configuration(computeUnit: computeUnit))
        return (model, (CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }

    // MARK: - 入力

    /// 同梱した入力 wav（`thesis_src_<name>.wav`）
    static func inputURL(name: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: "thesis_src_\(name)", withExtension: "wav") else {
            throw PipelineError.inputNotFound("thesis_src_\(name).wav")
        }
        return url
    }

    /// 同梱した PyTorch 版 Decoder の出力メル（`thesis_pytorch_postnet_<name>.npy`, [262 × 256]）
    static func pytorchMelURL(name: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: "thesis_pytorch_postnet_\(name)", withExtension: "npy") else {
            throw PipelineError.inputNotFound("thesis_pytorch_postnet_\(name).npy")
        }
        return url
    }

    /// 入力の一覧（`thesis_inputs.json`。PyTorch の前処理での T と T'）
    struct InputInfo: Decodable {
        let input_name: String
        let T: Int
        let T_prime: Int
    }

    static func inputInfos() throws -> [InputInfo] {
        struct File: Decodable { let inputs: [InputInfo] }
        guard let url = Bundle.main.url(forResource: "thesis_inputs", withExtension: "json") else {
            throw PipelineError.inputNotFound("thesis_inputs.json")
        }
        return try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).inputs
    }

    // MARK: - 合成

    func synthesize(inputURL: URL) async throws -> Measurement {
        let nMels = Self.nMels
        let tStart = CFAbsoluteTimeGetCurrent()

        // 1. 前処理（読み込み・リサンプリング・メル抽出）
        let samples = try AudioFeatureExtractor.loadAudio(from: inputURL)
        let (mel, totalFrames) = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples)
        let effectiveFrames: Int
        switch transformerPolicy {
        case .fixed(let targetT): effectiveFrames = min(totalFrames, targetT)
        case .dynamic(let maxT): effectiveFrames = min(totalFrames, maxT)
        }
        let preprocessMs = (CFAbsoluteTimeGetCurrent() - tStart) * 1000

        // 2. Encoder
        let encoderRunner = EncoderRunner(model: encoder, policy: transformerPolicy)
        let encoderResult = try await encoderRunner.run(mel: mel, frameCount: effectiveFrames, nMels: nMels)

        // 3. Decoder
        let decoderRunner = DecoderRunner(model: decoder, policy: transformerPolicy)
        let decoderResult = try await decoderRunner.run(
            memory: encoderResult.memory,
            frameCount: effectiveFrames,
            nMels: nMels,
            onStep: { _ in }
        )

        // 4. HiFi-GAN
        let vocoderRunner = VocoderRunner(model: hifigan, policy: vocoderPolicy)
        let vocoderResult = try await vocoderRunner.run(
            postnetOut: decoderResult.postnetOut, totalFrames: effectiveFrames, nMels: nMels
        )

        // 5. デエンファシス
        let tDeemph = CFAbsoluteTimeGetCurrent()
        let deemph = AudioFeatureExtractor.applyDeemphasis(vocoderResult.waveform)
        let tEnd = CFAbsoluteTimeGetCurrent()

        return Measurement(
            totalFrames: totalFrames,
            effectiveFrames: effectiveFrames,
            melIn: Array(mel.prefix(effectiveFrames * nMels)),
            preprocessMs: preprocessMs,
            encoderMs: encoderResult.predictMs,
            encoderInputNonFinite: encoderResult.inputNonFiniteCount,
            decoderTotalMs: decoderResult.totalPredictMs,
            decoderStepMs: decoderResult.stepPredictMs,
            decoderInputNonFiniteMax: decoderResult.maxInputNonFiniteCount,
            decoderMel: Self.leadingFrames(of: decoderResult.postnetOut, frames: effectiveFrames, nMels: nMels),
            hifiganMs: vocoderResult.predictMs,
            hifiganInputNonFinite: vocoderResult.inputNonFiniteCount,
            hifiganWaveform: vocoderResult.waveform,
            deemphMs: (tEnd - tDeemph) * 1000,
            deemphWaveform: deemph,
            e2eMs: (tEnd - tStart) * 1000
        )
    }

    /// [1, T, nMels] の先頭 `frames` フレームを行優先の [Float] にする
    static func leadingFrames(of array: MLMultiArray, frames: Int, nMels: Int) -> [Float] {
        let available = min(frames, array.shape.count == 3 ? array.shape[1].intValue : 0)
        var out = [Float](repeating: 0, count: available * nMels)
        for t in 0..<available {
            for m in 0..<nMels {
                out[t * nMels + m] = array[[0, t as NSNumber, m as NSNumber]].floatValue
            }
        }
        return out
    }
}
