import CoreML
import Foundation

/// 卒論第4章の計測モードの設定（起動時の環境変数から読む）
///
/// PC 側の `scripts/run_thesis_bench.py` が `devicectl device process launch --environment-variables`
/// で渡す。`CMLA_THESIS_BENCH` が無ければ通常の UI で起動する。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_THESIS_BENCH` | `load` / `plan` / `infer` / `vocoder` / `thermal` |
/// | `CMLA_THESIS_SESSION` | 計測セッションの ID（出力先 `Documents/Result/thesis/<session>/`） |
/// | `CMLA_THESIS_LAUNCH` | この起動の ID（PC 側の `host_runs.csv` と突き合わせる） |
/// | `CMLA_THESIS_PRECISION` | `Float32` / `Float16` / `Int8` |
/// | `CMLA_THESIS_CU` | `cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all` |
/// | `CMLA_THESIS_RELOADS` | load：再読み込みの回数（既定 5） |
/// | `CMLA_THESIS_WARMUP` | infer / vocoder：入力ごとのウォームアップ回数（既定 1） |
/// | `CMLA_THESIS_STEADY` | infer / vocoder：入力ごとの定常時の回数（既定 5） |
/// | `CMLA_THESIS_INPUTS` | infer / vocoder：入力の名前をカンマ区切りで（既定は thesis_inputs.json の順） |
struct ThesisBenchConfig {

    enum Mode: String {
        case load, plan, infer, vocoder, thermal
    }

    enum ConfigError: LocalizedError {
        case missing(String)
        case invalid(String, String)

        var errorDescription: String? {
            switch self {
            case .missing(let key): return "環境変数 \(key) がありません"
            case .invalid(let key, let value): return "環境変数 \(key) の値が不正です: \(value)"
            }
        }
    }

    static let shapeMode: ShapeModeOption = .fixed262

    let mode: Mode
    let sessionID: String
    let launchID: String
    let precision: ModelPrecision?
    let computeUnit: ComputeUnitOption?
    let reloadCount: Int
    let warmupCount: Int
    let steadyCount: Int
    let inputNames: [String]?

    /// 計測モードで起動されたか
    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["CMLA_THESIS_BENCH"] != nil
    }

    static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> ThesisBenchConfig {
        guard let rawMode = env["CMLA_THESIS_BENCH"] else { throw ConfigError.missing("CMLA_THESIS_BENCH") }
        guard let mode = Mode(rawValue: rawMode) else { throw ConfigError.invalid("CMLA_THESIS_BENCH", rawMode) }

        var precision: ModelPrecision?
        if let raw = env["CMLA_THESIS_PRECISION"] {
            guard let p = ModelPrecision(rawValue: raw) else { throw ConfigError.invalid("CMLA_THESIS_PRECISION", raw) }
            precision = p
        }
        var computeUnit: ComputeUnitOption?
        if let raw = env["CMLA_THESIS_CU"] {
            guard let cu = ComputeUnitOption(rawValue: raw) else { throw ConfigError.invalid("CMLA_THESIS_CU", raw) }
            computeUnit = cu
        }
        if mode != .thermal {
            if precision == nil { throw ConfigError.missing("CMLA_THESIS_PRECISION") }
            if computeUnit == nil { throw ConfigError.missing("CMLA_THESIS_CU") }
        }

        func intValue(_ key: String, default value: Int) throws -> Int {
            guard let raw = env[key] else { return value }
            guard let v = Int(raw), v >= 0 else { throw ConfigError.invalid(key, raw) }
            return v
        }

        return ThesisBenchConfig(
            mode: mode,
            sessionID: env["CMLA_THESIS_SESSION"] ?? "manual",
            launchID: env["CMLA_THESIS_LAUNCH"] ?? "manual",
            precision: precision,
            computeUnit: computeUnit,
            reloadCount: try intValue("CMLA_THESIS_RELOADS", default: 5),
            warmupCount: try intValue("CMLA_THESIS_WARMUP", default: 1),
            steadyCount: try intValue("CMLA_THESIS_STEADY", default: 5),
            inputNames: env["CMLA_THESIS_INPUTS"].map({ raw in
                raw.split(separator: ",").map({ String($0) })
            })
        )
    }

    /// 出力先 `Documents/Result/thesis/<session>/`
    var sessionDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("thesis", isDirectory: true)
            .appendingPathComponent(sessionID, isDirectory: true)
    }

    /// CSV や npy のファイル名に使う条件のラベル（例: `Float16_cpuAndNE`）
    var conditionLabel: String {
        "\(precision?.rawValue ?? "-")_\(computeUnit?.rawValue ?? "-")"
    }
}
