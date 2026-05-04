import Foundation

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
    /// `RangeDim(1, 1000, default=100)`。shape 付きモデルが無い場合は legacy 命名モデルを使う。
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

    /// 該当 precision でこの shape mode の HiFi-GAN mlmodelc が Bundle にあるか。
    /// `range1` は旧来の `HiFiGAN_Generator_<precision>` も 1-1000 互換モデルとして許可する。
    func isAvailable(for precision: ModelPrecision) -> Bool {
        hifiganResourceName(for: precision) != nil
    }

    /// 選択 shape に対応する Bundle 内 HiFi-GAN リソース名を返す。
    /// shape 付きモデルを優先し、`range1` だけ legacy 命名にフォールバックする。
    func hifiganResourceName(for precision: ModelPrecision) -> String? {
        let shapedName = "HiFiGAN_Generator_\(precision.suffix)_\(modelSuffix)"
        if Bundle.main.url(forResource: shapedName, withExtension: "mlmodelc") != nil {
            return shapedName
        }

        if self == .range1 {
            let legacyName = "HiFiGAN_Generator_\(precision.suffix)"
            if Bundle.main.url(forResource: legacyName, withExtension: "mlmodelc") != nil {
                return legacyName
            }
        }

        return nil
    }

    /// ロードしたリソース名に対応する表示・ログ用 shape ラベル。
    func resolvedShapeLabel(for resourceName: String, precision: ModelPrecision) -> String {
        let legacyName = "HiFiGAN_Generator_\(precision.suffix)"
        if resourceName == legacyName {
            return "legacy_range1_1000"
        }
        return modelSuffix
    }
}

/// VocoderRunner の入力 T 処理ポリシー
enum VocoderInputPolicy {
    /// 固定 shape モデル用: `totalFrames < targetT` なら zero-pad、`> targetT` なら crop
    case fixed(targetT: Int)
    /// 動的 shape モデル用: 実際の `totalFrames` をそのまま渡す。`> maxT` なら crop
    case dynamic(maxT: Int)
}
