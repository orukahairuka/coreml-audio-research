import CoreML
import Foundation

/// 卒論第4章の計測モードの設定（起動時の環境変数から読む）
///
/// PC 側の `scripts/run_thesis_bench.py` が `devicectl device process launch --environment-variables`
/// で渡す。`CMLA_THESIS_BENCH` が無ければ通常の UI で起動する。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_THESIS_BENCH` | `load` / `plan` / `infer` / `vocoder` / `preprocess` / `optab` / `thermal` |
/// | `CMLA_THESIS_SESSION` | 計測セッションの ID（出力先 `Documents/Result/thesis/<session>/`） |
/// | `CMLA_THESIS_LAUNCH` | この起動の ID（PC 側の `host_runs.csv` と突き合わせる） |
/// | `CMLA_THESIS_PRECISION` | `Float32` / `Float16` / `Int8` |
/// | `CMLA_THESIS_CU` | `cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all` |
/// | `CMLA_THESIS_RELOADS` | load：再読み込みの回数（既定 5） |
/// | `CMLA_THESIS_WARMUP` | infer / vocoder：入力ごとのウォームアップ回数（既定 1） |
/// | `CMLA_THESIS_STEADY` | infer / vocoder：入力ごとの定常時の回数（既定 5） |
/// | `CMLA_THESIS_INPUTS` | infer / vocoder：入力の名前をカンマ区切りで（既定は thesis_inputs.json の順） |
/// | `CMLA_THESIS_CU_HIFI` | 省略可。HiFi-GAN だけ別の Compute Units で読み込む（Encoder・Decoder は `CMLA_THESIS_CU`）。|
/// |  | 省略時と `CMLA_THESIS_CU` と同じ値のときは、従来の 3 モデル同一の条件になる |
/// | `CMLA_THESIS_PROFILE` | 省略可。`1` で infer の End-to-End を処理別に計測し `infer_stages.csv` に出す |
/// | `CMLA_THESIS_DECODER_INPUT_WRITE` | 省略可。Decoder 入力の書き込み方法 `bulk`（既定）/ `element`（ch4-v3 までの方法。比較用） |
/// | `CMLA_THESIS_STEP_STATS` | 省略可。`1` で Decoder のステップごとの詳細統計を取る（調査用。ch4-v2 までは常に取っていた）。既定は取らない |
struct ThesisBenchConfig {

    enum Mode: String {
        case load, plan, infer, vocoder, preprocess, optab, thermal
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
    /// HiFi-GAN だけに使う Compute Units（nil なら `computeUnit` と同じ）
    let hifiganComputeUnitOverride: ComputeUnitOption?
    /// infer で処理別の時間を計測するか
    let profileEnabled: Bool
    /// Decoder のステップごとの詳細統計を取るか
    let decoderStepStats: Bool
    /// Decoder 入力の書き込み方法
    let decoderInputWrite: DecoderInputWrite

    /// 計測コードの版。CSV の `bench_version` 列に出し、追加計測より前の結果（列が無い）と区別する
    /// ch4-v2: 混合 Compute Units・処理別計測を追加（Decoder のステップごとの統計は常に取っていた）
    /// ch4-v3: Decoder のステップごとの統計を既定で取らない（`decoder_step_stats` 列で区別する）
    /// ch4-v4: Decoder 入力をまとめてコピーする（`decoder_input_write` 列で区別する）。dec_input_alloc の区間を追加
    /// ch4-v5: 前処理を公開実装に合わせた（フレーム数が変わるので ch4-v4 までの結果と直接比べない）。preprocess モードを追加
    /// ch4-v6: 合成処理の高速化（`SynthesisOptimizations.adopted`）を既定で有効にした。optab モードを追加
    static let benchVersion = "ch4-v6"

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

        var hifiganComputeUnit: ComputeUnitOption?
        if let raw = env["CMLA_THESIS_CU_HIFI"], !raw.isEmpty {
            guard let cu = ComputeUnitOption(rawValue: raw) else { throw ConfigError.invalid("CMLA_THESIS_CU_HIFI", raw) }
            // CMLA_THESIS_CU と同じ値なら従来の条件と同じなので、指定が無いのと同じに扱う
            hifiganComputeUnit = cu == computeUnit ? nil : cu
        }
        var profileEnabled = false
        if let raw = env["CMLA_THESIS_PROFILE"] {
            guard raw == "0" || raw == "1" else { throw ConfigError.invalid("CMLA_THESIS_PROFILE", raw) }
            profileEnabled = raw == "1"
        }
        var decoderStepStats = false
        if let raw = env["CMLA_THESIS_STEP_STATS"] {
            guard raw == "0" || raw == "1" else { throw ConfigError.invalid("CMLA_THESIS_STEP_STATS", raw) }
            decoderStepStats = raw == "1"
        }
        var decoderInputWrite = DecoderInputWrite.bulk
        if let raw = env["CMLA_THESIS_DECODER_INPUT_WRITE"] {
            guard let method = DecoderInputWrite(rawValue: raw) else {
                throw ConfigError.invalid("CMLA_THESIS_DECODER_INPUT_WRITE", raw)
            }
            decoderInputWrite = method
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
            }),
            hifiganComputeUnitOverride: hifiganComputeUnit,
            profileEnabled: profileEnabled,
            decoderStepStats: decoderStepStats,
            decoderInputWrite: decoderInputWrite
        )
    }

    /// 出力先 `Documents/Result/thesis/<session>/`
    var sessionDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("thesis", isDirectory: true)
            .appendingPathComponent(sessionID, isDirectory: true)
    }

    /// HiFi-GAN の読み込みに使う Compute Units
    var hifiganComputeUnit: ComputeUnitOption? {
        hifiganComputeUnitOverride ?? computeUnit
    }

    /// モデルごとに Compute Units を分けた条件か
    var isMixedComputeUnits: Bool { hifiganComputeUnitOverride != nil }

    /// CSV の `compute_units` 列の値。従来の条件はそのまま（例: `cpuAndNE`）、
    /// HiFi-GAN だけ分けた条件は `cpuAndNE+hifi-cpuOnly` のようにして、従来の条件の行と混ざらないようにする
    var computeUnitsLabel: String {
        let base = computeUnit?.rawValue ?? ""
        guard let hifi = hifiganComputeUnitOverride else { return base }
        return "\(base)+hifi-\(hifi.rawValue)"
    }

    /// CSV や npy のファイル名に使う条件のラベル（例: `Float16_cpuAndNE`、`Float16_cpuAndNE+hifi-cpuOnly`）
    var conditionLabel: String {
        "\(precision?.rawValue ?? "-")_\(computeUnit == nil ? "-" : computeUnitsLabel)"
    }
}
