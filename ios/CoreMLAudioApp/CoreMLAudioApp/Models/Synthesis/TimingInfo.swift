import Foundation

/// CoreML パイプライン各段の `predict()` 所要時間 (ミリ秒)
///
/// メル抽出やデエンファシスなどの前後処理は含めず、純粋に CoreML 推論にかかった時間を保持する。
struct TimingInfo: Codable {
    /// Encoder.predict() (1 回呼び出し)
    let encoderMs: Double

    /// Decoder.predict() の合計 (frameCount 回ぶんの総和)
    let decoderTotalMs: Double

    /// Decoder のステップ回数 (= frameCount)
    let decoderStepCount: Int

    /// HiFi-GAN.predict() (1 回呼び出し)
    let hifiganMs: Double

    /// Decoder の 1 ステップあたり平均 (decoderTotalMs / decoderStepCount)
    var decoderAvgPerStepMs: Double {
        decoderStepCount > 0 ? decoderTotalMs / Double(decoderStepCount) : 0
    }

    /// 純粋な CoreML 推論時間の合計 (encoder + decoder + hifigan)
    var totalPredictMs: Double {
        encoderMs + decoderTotalMs + hifiganMs
    }
}
