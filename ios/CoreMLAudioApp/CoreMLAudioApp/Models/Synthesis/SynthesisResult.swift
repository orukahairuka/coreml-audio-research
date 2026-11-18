/// 合成パイプラインの入出力データを保持する
struct SynthesisResult {
    /// 合成時の精度設定
    let precision: ModelPrecision
    /// 合成時の compute unit 設定（Encoder・Decoder）
    let computeUnit: ComputeUnitOption
    /// HiFi-GAN の compute unit 設定（3 モデル同じ指定なら computeUnit と同じ）
    let hifiganComputeUnit: ComputeUnitOption
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

    /// ファイル名などに使う条件の表記。3 モデル同じ指定なら従来どおり computeUnit の rawValue、
    /// HiFi-GAN だけ別なら `cpuAndNE+hifi-cpuOnly` のように両方を書く（計測モードの compute_units 列と同じ表記）
    var computeLabel: String {
        hifiganComputeUnit == computeUnit ? computeUnit.rawValue : "\(computeUnit.rawValue)+hifi-\(hifiganComputeUnit.rawValue)"
    }

    /// 画面に表示する条件の名前
    var computeDisplayName: String {
        hifiganComputeUnit == computeUnit
            ? computeUnit.displayName
            : "\(computeUnit.displayName)（HiFi-GAN は \(hifiganComputeUnit.displayName)）"
    }
}
