import Foundation

/// 通常操作の応答時間の計測モードの設定（起動時の環境変数から読む）
///
/// PC 側の `scripts/run_response_bench.py` が `devicectl device process launch --environment-variables` で渡す。
/// `CMLA_RESPONSE_BENCH` が無ければ何もしない（通常の起動）。
/// 卒論第4章の計測モード（`CMLA_THESIS_BENCH`）とは別で、こちらは通常の画面（ContentView）のまま
/// `SynthesisViewModel.runSynthesis()` を繰り返し呼ぶ。合成・保存・再生の処理経路は通常の操作と同じ。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_RESPONSE_BENCH` | `1` で計測モード |
/// | `CMLA_RESPONSE_SESSION` | 計測セッションの ID（出力先 `Documents/Result/response/<session>/`） |
/// | `CMLA_RESPONSE_LAUNCH` | この起動の ID（PC 側の `host_runs.csv` と突き合わせる） |
/// | `CMLA_RESPONSE_PRECISION` | `Float32` / `Float16` / `Int8`（既定 `Float16`） |
/// | `CMLA_RESPONSE_SETTING` | 画面の Compute Units の選択肢の rawValue（既定 `cpuAndNE+hifi-cpuOnly` ＝条件 C） |
/// | `CMLA_RESPONSE_WARMUP` | 入力ごとのウォームアップ回数（既定 1） |
/// | `CMLA_RESPONSE_STEADY` | 入力ごとの本計測の回数（既定 3） |
/// | `CMLA_RESPONSE_INPUTS` | 入力の名前をカンマ区切りで（既定は thesis_inputs.json の順 = 入力1〜4） |
/// | `CMLA_RESPONSE_GAP_MS` | 再生が終わってから次の合成を始めるまでの待ち（既定 1000 ms） |
struct ResponseBenchConfig {

    /// 計測コードの版。CSV の `bench_version` 列に出す
    static let benchVersion = "resp-v1"

    enum ConfigError: LocalizedError {
        case invalid(String, String)

        var errorDescription: String? {
            switch self {
            case .invalid(let key, let value): return "環境変数 \(key) の値が不正です: \(value)"
            }
        }
    }

    let sessionID: String
    let launchID: String
    let precision: ModelPrecision
    let setting: AppComputeSetting
    let warmupCount: Int
    let steadyCount: Int
    let inputNames: [String]?
    let gapMs: Int

    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["CMLA_RESPONSE_BENCH"] == "1"
    }

    static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> ResponseBenchConfig {
        let precisionRaw = env["CMLA_RESPONSE_PRECISION"] ?? ModelPrecision.float16.rawValue
        guard let precision = ModelPrecision(rawValue: precisionRaw) else {
            throw ConfigError.invalid("CMLA_RESPONSE_PRECISION", precisionRaw)
        }
        let settingRaw = env["CMLA_RESPONSE_SETTING"] ?? AppComputeSetting.cpuAndNEHifiganCPU.rawValue
        guard let setting = AppComputeSetting(rawValue: settingRaw) else {
            throw ConfigError.invalid("CMLA_RESPONSE_SETTING", settingRaw)
        }
        func intValue(_ key: String, default value: Int) throws -> Int {
            guard let raw = env[key] else { return value }
            guard let v = Int(raw), v >= 0 else { throw ConfigError.invalid(key, raw) }
            return v
        }
        return ResponseBenchConfig(
            sessionID: env["CMLA_RESPONSE_SESSION"] ?? "manual",
            launchID: env["CMLA_RESPONSE_LAUNCH"] ?? "manual",
            precision: precision,
            setting: setting,
            warmupCount: try intValue("CMLA_RESPONSE_WARMUP", default: 1),
            steadyCount: try intValue("CMLA_RESPONSE_STEADY", default: 3),
            inputNames: env["CMLA_RESPONSE_INPUTS"].map({ raw in
                raw.split(separator: ",").map({ String($0) })
            }),
            gapMs: try intValue("CMLA_RESPONSE_GAP_MS", default: 1000)
        )
    }

    /// 出力先 `Documents/Result/response/<session>/`（卒論第4章の `Result/thesis/` とは別）
    var sessionDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("response", isDirectory: true)
            .appendingPathComponent(sessionID, isDirectory: true)
    }

    /// ビルド構成（Release で計測する）
    static var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }
}
