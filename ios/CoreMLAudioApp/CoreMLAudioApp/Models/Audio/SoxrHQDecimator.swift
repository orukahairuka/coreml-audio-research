import Accelerate

/// 44.1 kHz の波形を 22.05 kHz に変換する 2:1 の FIR フィルタ
///
/// 公開実装の `librosa.load`（soxr_hq）の出力に一致させるためのもの。
/// 係数は `scripts/generate_resampler_fir.py` が soxr_hq のインパルス応答から生成する。
enum SoxrHQDecimator {

    static let sourceRate: Double = 44100
    static let targetRate: Double = 22050

    /// y[n] = Σ_j c[j] · x[2n - firstOffset - j]。出力の長さは librosa.resample と同じ ceil(N / 2)。
    /// 信号の範囲外のサンプルは 0 として扱う
    static func decimate(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        let coefficients = SoxrHQDecimatorCoefficients.coefficients
        let firstOffset = SoxrHQDecimatorCoefficients.firstOffset
        let tapCount = coefficients.count
        let reversed = [Float](coefficients.reversed())

        // フィルタが信号の外にはみ出す分を 0 で埋める
        let margin = tapCount + abs(firstOffset) + 2
        var padded = [Float](repeating: 0, count: margin + samples.count + margin)
        padded.replaceSubrange(margin..<(margin + samples.count), with: samples)

        let outputCount = (samples.count + 1) / 2
        var output = [Float](repeating: 0, count: outputCount)
        padded.withUnsafeBufferPointer({ source in
            reversed.withUnsafeBufferPointer({ taps in
                output.withUnsafeMutableBufferPointer({ destination in
                    for n in 0..<outputCount {
                        let oldest = 2 * n - firstOffset - (tapCount - 1) + margin
                        vDSP_dotpr(
                            source.baseAddress! + oldest, 1,
                            taps.baseAddress!, 1,
                            destination.baseAddress! + n,
                            vDSP_Length(tapCount)
                        )
                    }
                })
            })
        })
        return output
    }
}
