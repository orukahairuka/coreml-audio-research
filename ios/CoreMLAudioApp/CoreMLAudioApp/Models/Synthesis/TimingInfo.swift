import Foundation

/// CoreML パイプライン各段の `predict()` 所要時間 (ミリ秒) と関連メトリクス
///
/// メル抽出やデエンファシスなどの前後処理は含めず、純粋に CoreML 推論にかかった時間を保持する。
/// あわせてモデルサイズ・出力音声長など Pareto 分析で必要な量も持つ。
struct TimingInfo: Codable {
    /// Encoder.predict() (1 回呼び出し)
    let encoderMs: Double

    /// Decoder.predict() の合計 (frameCount 回ぶんの総和)
    let decoderTotalMs: Double

    /// Decoder のステップ回数 (= frameCount)
    let decoderStepCount: Int

    /// HiFi-GAN.predict() (1 回呼び出し)
    let hifiganMs: Double

    /// 出力波形の長さ (ミリ秒)
    let outputDurationMs: Double

    /// 使用した 3 モデル (Encoder + Decoder + HiFi-GAN) の .mlmodelc 合計バイト数
    let modelSizeBytes: Int64

    /// Decoder の 1 ステップあたり平均 (decoderTotalMs / decoderStepCount)
    var decoderAvgPerStepMs: Double {
        decoderStepCount > 0 ? decoderTotalMs / Double(decoderStepCount) : 0
    }

    /// 純粋な CoreML 推論時間の合計 (encoder + decoder + hifigan)
    var totalPredictMs: Double {
        encoderMs + decoderTotalMs + hifiganMs
    }

    /// Real-time factor: 出力1秒を作るのに何秒かかるか (totalPredictMs / outputDurationMs)
    /// 1.0 未満ならリアルタイム合成可
    var realTimeFactor: Double {
        outputDurationMs > 0 ? totalPredictMs / outputDurationMs : 0
    }
}
