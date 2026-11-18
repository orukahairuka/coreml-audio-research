import Foundation

/// アプリ画面で選ぶ計算デバイスの設定
///
/// 3 つのモデル（Encoder・Decoder・HiFi-GAN）に同じ Compute Units を指定する 4 種類に加え、
/// Encoder・Decoder は CPU + Neural Engine、HiFi-GAN だけ CPU で実行する設定（卒論第4章の条件 C）を選べる。
/// HiFi-GAN を Neural Engine で実行すると Float16 / Int8 モデルで振幅が増大するため、
/// HiFi-GAN だけ Neural Engine から外して速さと振幅の両方を満たすための設定。
///
/// 卒論の計測モード（`ThesisBenchConfig`）は環境変数から `ComputeUnitOption` を直接読むので、この型は使わない。
enum AppComputeSetting: String, CaseIterable, Identifiable {
    case all = "all"
    case cpuAndGPU = "cpuAndGPU"
    case cpuAndNE = "cpuAndNE"
    case cpuOnly = "cpuOnly"
    /// Encoder・Decoder は cpuAndNE、HiFi-GAN は cpuOnly。
    /// rawValue は計測モードの CSV の compute_units 列（`ThesisBenchConfig.computeUnitsLabel`）と同じ表記
    case cpuAndNEHifiganCPU = "cpuAndNE+hifi-cpuOnly"

    var id: String { rawValue }

    /// Encoder と Decoder に指定する Compute Units
    var encoderDecoderUnit: ComputeUnitOption {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNE, .cpuAndNEHifiganCPU: return .cpuAndNE
        case .cpuOnly: return .cpuOnly
        }
    }

    /// HiFi-GAN に指定する Compute Units
    var hifiganUnit: ComputeUnitOption {
        switch self {
        case .cpuAndNEHifiganCPU: return .cpuOnly
        default: return encoderDecoderUnit
        }
    }

    var displayName: String {
        switch self {
        case .cpuAndNEHifiganCPU: return "CPU + ANE（HiFi-GAN は CPU のみ）"
        default: return encoderDecoderUnit.displayName
        }
    }
}
