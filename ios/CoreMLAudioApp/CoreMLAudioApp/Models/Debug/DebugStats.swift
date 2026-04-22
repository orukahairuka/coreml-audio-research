import CoreML
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

extension ArrayStats {
    /// MLMultiArray の全要素を走査して統計を計算する
    static func compute(from array: MLMultiArray) -> ArrayStats {
        let count = array.count
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        var sum: Float = 0
        var hasNaN = false
        var hasInf = false
        for i in 0..<count {
            let v = array[i].floatValue
            if v.isNaN { hasNaN = true }
            if v.isInfinite { hasInf = true }
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
            sum += v
        }
        return ArrayStats(
            min: minVal, max: maxVal, mean: sum / Float(count),
            hasNaN: hasNaN, hasInf: hasInf
        )
    }

    /// Float 配列を走査して統計を計算する
    static func compute(from array: [Float]) -> ArrayStats {
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        var sum: Float = 0
        var hasNaN = false
        var hasInf = false
        for v in array {
            if v.isNaN { hasNaN = true }
            if v.isInfinite { hasInf = true }
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
            sum += v
        }
        return ArrayStats(
            min: minVal, max: maxVal, mean: sum / Float(array.count),
            hasNaN: hasNaN, hasInf: hasInf
        )
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
    let waveformAfterDeemphasis: ArrayStats
}
