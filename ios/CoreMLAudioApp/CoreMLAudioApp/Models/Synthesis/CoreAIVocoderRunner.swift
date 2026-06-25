#if canImport(CoreAI)
import CoreAI
import Foundation

/// HiFi-GAN ボコーダーの **Core AI (.aimodel)** 推論を 1 回実行する。
///
/// 既存 `VocoderRunner`（CoreML / `MLModel`）の Core AI 版。研究目的は
/// 「CoreML で観測した HiFi-GAN×Neural Engine の破綻（rms 約4.25倍→clip）が
/// 別ランタイム Core AI でも再現するか」の切り分け。
///
/// CoreML との対応:
/// - `MLModel` → `AIModel`
/// - `MLModelConfiguration.computeUnits` → `SpecializationOptions.preferredComputeUnitKind`
/// - `model.prediction(from:)` → `function.run(inputs:states:outputViews:)`
/// - `MLMultiArray` → `NDArray`
///
/// MVP では固定 shape（mel `[1, 256, 262]` → audio `[1, 1, 262*256]`）のみ扱う。
@available(iOS 27.0, macOS 27.0, *)
final class CoreAIVocoderRunner {

    enum RunnerError: Error {
        case functionNotFound
    }

    private let model: AIModel
    private let function: InferenceFunction
    private let hopSize: Int

    private init(model: AIModel, function: InferenceFunction, hopSize: Int) {
        self.model = model
        self.function = function
        self.hopSize = hopSize
    }

    /// `.aimodel` を指定 compute unit でロードする。
    ///
    /// - Parameters:
    ///   - modelURL: `.aimodel` ディレクトリの URL
    ///   - computeUnit: cpu / gpu / neuralEngine
    ///   - hopSize: 1 frame あたりの出力サンプル数（HiFi-GAN config の hop_size = 256）
    /// - Returns: ロード済み runner と、ロード（＝初回コンパイル含む）所要時間 ms
    static func load(
        modelURL: URL,
        computeUnit: CoreAIComputeUnit,
        hopSize: Int = 256
    ) async throws -> (runner: CoreAIVocoderRunner, loadMs: Double) {
        let options = SpecializationOptions(preferredComputeUnitKind: computeUnit.kind)
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await AIModel(contentsOf: modelURL, options: options)
        guard let function = try model.loadFunction(named: "main") else {
            throw RunnerError.functionNotFound
        }
        let loadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let runner = CoreAIVocoderRunner(model: model, function: function, hopSize: hopSize)
        return (runner: runner, loadMs: loadMs)
    }

    /// HiFi-GAN を 1 回実行する。
    ///
    /// - Parameters:
    ///   - mel: `[1, nMels, frames]` レイアウトの mel（長さ = nMels * frames）。
    ///     参照 npy がこの順なのでそのまま渡せる（CoreML 版のような転置は不要）。
    ///   - nMels: メルビン数（256）
    ///   - frames: 時間方向フレーム数 T（固定 262）
    /// - Returns: 合成波形（長さ = frames * hopSize）と `run` の所要時間 ms
    func run(mel: [Float], nMels: Int = 256, frames: Int = 262) async throws -> (waveform: [Float], predictMs: Double) {
        precondition(mel.count == nMels * frames, "mel length \(mel.count) != \(nMels * frames)")

        // 入力 NDArray [1, nMels, frames] を mel で埋める
        var input = NDArray(shape: [1, nMels, frames], scalarType: .float32)
        var inputView = input.mutableView(as: Float.self)
        inputView.copyElements(fromContentsOf: mel)

        // 出力 NDArray [1, 1, frames*hopSize] を事前確保
        let outputCount = frames * hopSize
        var output = NDArray(shape: [1, 1, outputCount], scalarType: .float32)
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&output, for: "audio")

        // 推論（時間計測は run のみ）
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try await function.run(
            inputs: ["mel": input],
            states: InferenceFunction.MutableViews(),
            outputViews: consume outputViews
        )
        let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // 出力 NDArray → [Float]
        let waveform = output.view(as: Float.self).withUnsafePointer { pointer, _, _ -> [Float] in
            Array(UnsafeBufferPointer(start: pointer, count: outputCount))
        }
        return (waveform: waveform, predictMs: predictMs)
    }
}
#endif
