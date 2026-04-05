import Foundation

/// MLMultiArray の統計情報（デバッグ用）
struct ArrayStats {
    let min: Float
    let max: Float
    let mean: Float
    let hasNaN: Bool
    let hasInf: Bool

    var summary: String {
        var flags = [String]()
        if hasNaN { flags.append("NaN") }
        if hasInf { flags.append("Inf") }
        let flagStr = flags.isEmpty ? "" : " [" + flags.joined(separator: ", ") + "]"
        return String(format: "min=%.4e  max=%.4e  mean=%.4e%@", min, max, mean, flagStr)
    }
}

/// デコーダー1ステップ分の統計
struct DecoderStepStats {
    let step: Int
    let melOut: ArrayStats
    let postnetOut: ArrayStats
}

/// パイプライン全体のデバッグ情報
struct PipelineDebugInfo {
    let encoderOutput: ArrayStats
    let decoderSteps: [DecoderStepStats]
    let hifiganInput: ArrayStats
    let hifiganOutput: ArrayStats
    let waveformBeforeDeemphasis: ArrayStats
    let waveformAfterDeemphasis: ArrayStats
}
