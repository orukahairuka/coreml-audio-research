/// 合成パイプラインの入出力データを保持する
struct SynthesisResult {
    /// 合成時の精度設定
    let precision: ModelPrecision
    /// 合成時の compute unit 設定
    let computeUnit: ComputeUnitOption
    /// 入力音声の生波形 (22050 Hz)
    let inputWaveform: [Float]
    /// 合成後の波形 (デエンファシス適用済み)
    let outputWaveform: [Float]
    /// 入力メルスペクトログラム [T_in × nMels] (dB スケール, -80〜0)
    let inputMelSpectrogram: [Float]
    /// 出力メルスペクトログラム [T_out × nMels] (dB スケール, -80〜0)
    let outputMelSpectrogram: [Float]
    let inputFrameCount: Int
    let outputFrameCount: Int
    let nMels: Int
    let sampleRate: Double
    /// パイプライン各ステップのデバッグ統計情報
    let debugInfo: PipelineDebugInfo
    /// CoreML 各段の predict() 所要時間
    let timing: TimingInfo
}
