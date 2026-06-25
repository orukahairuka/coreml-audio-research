#if canImport(CoreAI)
import CoreAI

/// Core AI の compute unit 選択。
///
/// CoreML の `ComputeUnitOption`（`.all` / `.cpuAndGPU` / `.cpuAndNE` / `.cpuOnly` の
/// 組み合わせ指定）と異なり、Core AI は「優先する単一 unit」を
/// `SpecializationOptions.preferredComputeUnitKind` に渡す方式。
/// よって研究の比較軸は cpu / gpu / neuralEngine の 3 値とする。
///
/// 同じ fp32 `.aimodel` を 3 通りでロードすると、精度（fp16 vs fp32）は
/// 実行時に切り替わる（Neural Engine は本質的に fp16）。これが
/// 「fp16 専用モデルを別途用意しない」理由。
@available(iOS 27.0, macOS 27.0, *)
enum CoreAIComputeUnit: String, CaseIterable, Identifiable {
    case cpu
    case gpu
    case neuralEngine

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cpu:          return "CPU"
        case .gpu:          return "GPU"
        case .neuralEngine: return "Neural Engine"
        }
    }

    /// CSV / ログ用の安定したラベル。
    var label: String { rawValue }

    /// 比較のために CoreML 側の近い設定を記録しておく（厳密一致ではない）。
    var coreMLEquivalentNote: String {
        switch self {
        case .cpu:          return "cpuOnly 相当"
        case .gpu:          return "cpuAndGPU 寄り"
        case .neuralEngine: return "cpuAndNE 寄り"
        }
    }

    /// Core AI の compute unit 種別。
    var kind: ComputeUnitKind {
        switch self {
        case .cpu:          return .cpu
        case .gpu:          return .gpu
        case .neuralEngine: return .neuralEngine
        }
    }
}
#endif
