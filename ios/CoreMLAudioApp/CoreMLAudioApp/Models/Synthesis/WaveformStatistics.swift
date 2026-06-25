import Foundation

/// 合成波形 [Float] の統計量。
///
/// 既存 `VocoderStabilityTester.WaveformStats` と同じ指標を、Core AI 経路でも
/// 使えるよう独立化したもの。CoreML の結果（rms 約4.25倍膨張・clipping）と
/// そのまま並べて比較するため、CSV 列名も合わせてある。
///
/// min / max / mean / rms は **有限値のみ**で計算し、nan / inf は別途カウントする
/// （nan が混じると min/max/mean が伝播して壊れるため）。
struct WaveformStatistics {
    let count: Int
    let finiteCount: Int
    let min: Float
    let max: Float
    let mean: Float
    let rms: Float
    let nanCount: Int
    let infCount: Int
    /// x > 0.99 の割合（正側 clipping 指標）。
    let ratioOver099: Double
    /// x < -0.99 の割合（負側 clipping 指標）。
    let ratioUnderMinus099: Double

    init(waveform: [Float]) {
        count = waveform.count

        var minV = Float.greatestFiniteMagnitude
        var maxV = -Float.greatestFiniteMagnitude
        var sum = 0.0
        var sumSq = 0.0
        var finite = 0
        var nan = 0
        var inf = 0
        var over = 0
        var under = 0

        for x in waveform {
            if x.isNaN {
                nan += 1
                continue
            }
            if x.isInfinite {
                inf += 1
                continue
            }
            finite += 1
            if x < minV { minV = x }
            if x > maxV { maxV = x }
            sum += Double(x)
            sumSq += Double(x) * Double(x)
            if x > 0.99 { over += 1 }
            if x < -0.99 { under += 1 }
        }

        finiteCount = finite
        nanCount = nan
        infCount = inf
        min = finite > 0 ? minV : .nan
        max = finite > 0 ? maxV : .nan
        mean = finite > 0 ? Float(sum / Double(finite)) : .nan
        rms = finite > 0 ? Float((sumSq / Double(finite)).squareRoot()) : .nan
        let denom = Double(count > 0 ? count : 1)
        ratioOver099 = Double(over) / denom
        ratioUnderMinus099 = Double(under) / denom
    }

    static var csvHeader: String {
        "count,finite_count,min,max,mean,rms,nan_count,inf_count,ratio_over_099,ratio_under_minus_099"
    }

    var csvRow: String {
        [
            String(count),
            String(finiteCount),
            String(format: "%.6e", min),
            String(format: "%.6e", max),
            String(format: "%.6e", mean),
            String(format: "%.6e", rms),
            String(nanCount),
            String(infCount),
            String(format: "%.6f", ratioOver099),
            String(format: "%.6f", ratioUnderMinus099),
        ].joined(separator: ",")
    }

    /// PSNR(dB) = 20 * log10(peak / rmse)。peak = max(|reference|)。
    ///
    /// 長さが違う場合は短い方に合わせる。non-finite が混じる場合は無限大を避けて nan を返す。
    static func psnr(reference: [Float], test: [Float]) -> Double {
        let n = Swift.min(reference.count, test.count)
        guard n > 0 else { return .nan }
        var sumSq = 0.0
        var peak = 0.0
        var valid = 0
        for i in 0..<n {
            let r = reference[i]
            let t = test[i]
            if r.isNaN || t.isNaN || r.isInfinite || t.isInfinite { continue }
            let diff = Double(r) - Double(t)
            sumSq += diff * diff
            peak = Swift.max(peak, Double(abs(r)))
            valid += 1
        }
        guard valid > 0, peak > 0 else { return .nan }
        let rmse = (sumSq / Double(valid)).squareRoot()
        if rmse == 0 { return .infinity }
        return 20.0 * log10(peak / rmse)
    }
}
