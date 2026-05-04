/// HiFi-GAN モデルの入力 shape バリアント
///
/// 実機テストで判明した「広い RangeDim ＋ GPU 経路で `E5RT: No memory object bound to port`」を踏まえ、
/// 本番デフォルトは `fixed262`。可変長対応や RangeDim 上限の影響を再現したい場合に他のオプションを使う。
enum ShapeModeOption: String, CaseIterable, Identifiable {
    /// 固定 (1, 256, 262)。本番推奨。実機で Float32/Float16 とも全 computeUnits 成功
    case fixed262
    /// `RangeDim(16, 384, default=262)`。GPU 経路でも動作する可変長候補
    case range16_384
    /// `RangeDim(16, 1000, default=100)`。GPU 経路で E5RT 失敗 (検証用)
    case range16
    /// `RangeDim(1, 1000, default=100)`。Float32 のみ生成済み (検証用)
    case range1

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .fixed262:     return "fixed262 (推奨)"
        case .range16_384:  return "RangeDim 16-384"
        case .range16:      return "RangeDim 16-1000"
        case .range1:       return "RangeDim 1-1000"
        }
    }

    /// モデルファイル名のサフィックス: `HiFiGAN_Generator_<precision>_<modelSuffix>.mlmodelc`
    var modelSuffix: String { rawValue }

    /// VocoderRunner に渡す入力 T のポリシー
    var inputPolicy: VocoderInputPolicy {
        switch self {
        case .fixed262:                  return .fixed(targetT: 262)
        case .range16_384:               return .dynamic(maxT: 384)
        case .range16, .range1:          return .dynamic(maxT: 1000)
        }
    }

    /// 該当 precision でこの shape mode の mlpackage が生成されているか。
    /// Int8 はバリアント未生成だが `AudioSynthesizer` 側で legacy mlpackage に
    /// フォールバックするので、ここでは利用可能扱いにする (shape mode は無視される)。
    func isAvailable(for precision: ModelPrecision) -> Bool {
        if precision == .int8 { return true }  // Int8 は legacy にフォールバック
        if precision == .float16 && self == .range1 { return false }  // Float16+range1 は未生成
        return true
    }
}

/// VocoderRunner の入力 T 処理ポリシー
enum VocoderInputPolicy {
    /// 固定 shape モデル用: `totalFrames < targetT` なら zero-pad、`> targetT` なら crop
    case fixed(targetT: Int)
    /// 動的 shape モデル用: 実際の `totalFrames` をそのまま渡す。`> maxT` なら crop
    case dynamic(maxT: Int)
}
