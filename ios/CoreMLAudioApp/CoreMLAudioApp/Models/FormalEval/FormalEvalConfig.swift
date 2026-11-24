import CoreML
import Foundation

/// 正式評価（FAD・RTF）用の一括合成モードの設定（起動時の環境変数から読む）
///
/// 第4章の計測（`ThesisBenchConfig`、`fixed262`）とは別の経路。条件は正式評価で確定したものに固定する:
/// モデルは `range1`（RangeDim 1〜1000）、波形の前処理は `.fullInput`（先頭の無音を削らず、フェードしない）、
/// Decoder は入力のフレーム数 T だけまわし、出力は T × 256 サンプル。
/// PC 側の `scripts/run_formal_eval.py` が `devicectl device process launch --environment-variables` で渡す。
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_FORMAL_EVAL` | `1` でこのモードで起動する |
/// | `CMLA_FORMAL_SESSION` | セッションの ID（出力先 `Documents/Result/formal_eval/<session>/`）。必須 |
/// | `CMLA_FORMAL_PRECISION` | `Float32` / `Float16` / `Int8`。必須 |
/// | `CMLA_FORMAL_CU` | `cpuOnly` / `cpuAndGPU` / `cpuAndNE` / `all`。必須 |
/// | `CMLA_FORMAL_SPEAKERS` | 省略可。話者 ID をカンマ区切りで（既定は入力フォルダにある全話者） |
/// | `CMLA_FORMAL_NAMES` | 省略可。入力のファイル名（拡張子なし、`<参照>_<話者>`）をカンマ区切りで。指定したものだけ合成する |
/// | `CMLA_FORMAL_LIMIT` | 省略可。話者ごとに先頭から何件まで合成するか（動作確認用） |
///
/// 実験用（GPU・ANE の切り分けのため。指定しなければ従来と同じ動作）:
///
/// | 変数 | 内容 |
/// |---|---|
/// | `CMLA_FORMAL_CU_ENC` / `_DEC` / `_HIFI` | モデルごとの Compute Units（省略したモデルは `CMLA_FORMAL_CU`） |
/// | `CMLA_FORMAL_DEC_INPUT` | `grow`（既定。Decoder 入力を 1 フレームずつ伸ばす）/ `padT`（毎回 T フレームにして残りを 0 で埋める。|
/// |  | 因果マスクにより結果は同じで、1 サンプルの中では入力の形が変わらない。GPU で形ごとにメモリが増えるのを避ける） |
/// | `CMLA_FORMAL_RELOAD_MB` | サンプルの後でメモリ使用量（phys_footprint）がこれを超えていたら、3 モデルを読み込み直す |
///   （GPU では形ごとにメモリが増え、モデルを解放すると戻るため）。使ったときは `memory.csv` に記録する |
///
/// 入力: `Documents/FormalEvalInputs/<話者>/<参照>_<話者>.wav`（librosa.load(sr=22050) で作った float32 の wav。
/// `scripts/prepare_formal_eval_inputs.py` で作り、`run_formal_eval.py` が端末に送る）
struct FormalEvalConfig {
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

    static let shapeMode = ShapeModeOption.range1
    static let waveformPreprocess = AudioFeatureExtractor.WaveformPreprocess.fullInput

    let session: String
    let precision: ModelPrecision
    let computeUnit: ComputeUnitOption
    let speakers: [String]?
    let names: Set<String>?
    let limitPerSpeaker: Int?
    /// モデルごとの Compute Units（実験用。既定は 3 つとも `computeUnit`）
    var modelComputeUnits: [String: ComputeUnitOption] = [:]
    /// Decoder 入力を毎回 T フレームにするか（実験用。既定 false）
    var padDecoderInput = false
    /// メモリがこれを超えたらモデルを読み込み直す（実験用。既定 nil）
    var reloadThresholdMB: Double?

    /// 実験用の設定を 1 つでも使っているか
    var usesExperimentalOptions: Bool { !modelComputeUnits.isEmpty || padDecoderInput || reloadThresholdMB != nil }

    /// モデル（`encoder` / `decoder` / `hifigan`）の Compute Units
    func computeUnit(for role: String) -> ComputeUnitOption { modelComputeUnits[role] ?? computeUnit }

    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["CMLA_FORMAL_EVAL"] == "1"
    }

    static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> FormalEvalConfig {
        guard let session = env["CMLA_FORMAL_SESSION"], !session.isEmpty else {
            throw ConfigError.missing("CMLA_FORMAL_SESSION")
        }
        // セッション ID はフォルダ名になるので、区切り文字などを含むものは受け付けない
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        guard session.unicodeScalars.allSatisfy({ allowed.contains($0) }), !session.hasPrefix(".") else {
            throw ConfigError.invalid("CMLA_FORMAL_SESSION", session)
        }
        guard let p = env["CMLA_FORMAL_PRECISION"] else { throw ConfigError.missing("CMLA_FORMAL_PRECISION") }
        guard let precision = ModelPrecision(rawValue: p) else { throw ConfigError.invalid("CMLA_FORMAL_PRECISION", p) }
        guard let c = env["CMLA_FORMAL_CU"] else { throw ConfigError.missing("CMLA_FORMAL_CU") }
        guard let computeUnit = ComputeUnitOption(rawValue: c) else { throw ConfigError.invalid("CMLA_FORMAL_CU", c) }
        let list: (String) -> [String]? = { key in
            guard let value = env[key], !value.isEmpty else { return nil }
            return value.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) })
        }
        var limit: Int?
        if let value = env["CMLA_FORMAL_LIMIT"], !value.isEmpty {
            guard let n = Int(value), n > 0 else { throw ConfigError.invalid("CMLA_FORMAL_LIMIT", value) }
            limit = n
        }
        var config = FormalEvalConfig(
            session: session, precision: precision, computeUnit: computeUnit,
            speakers: list("CMLA_FORMAL_SPEAKERS"), names: list("CMLA_FORMAL_NAMES").map(Set.init),
            limitPerSpeaker: limit
        )
        for (role, key) in [("encoder", "CMLA_FORMAL_CU_ENC"), ("decoder", "CMLA_FORMAL_CU_DEC"), ("hifigan", "CMLA_FORMAL_CU_HIFI")] {
            guard let value = env[key], !value.isEmpty else { continue }
            guard let unit = ComputeUnitOption(rawValue: value) else { throw ConfigError.invalid(key, value) }
            config.modelComputeUnits[role] = unit
        }
        switch env["CMLA_FORMAL_DEC_INPUT"] ?? "grow" {
        case "grow", "": config.padDecoderInput = false
        case "padT": config.padDecoderInput = true
        case let value: throw ConfigError.invalid("CMLA_FORMAL_DEC_INPUT", value)
        }
        if let value = env["CMLA_FORMAL_RELOAD_MB"], !value.isEmpty {
            guard let mb = Double(value), mb > 0 else { throw ConfigError.invalid("CMLA_FORMAL_RELOAD_MB", value) }
            config.reloadThresholdMB = mb
        }
        return config
    }

    /// 条件のラベル（出力のフォルダ名にも使う）
    var conditionLabel: String { "\(precision.rawValue)_\(computeUnit.rawValue)" }

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var inputRoot: URL { documents.appendingPathComponent("FormalEvalInputs", isDirectory: true) }
    var sessionDirectory: URL {
        Self.documents.appendingPathComponent("Result/formal_eval/\(session)", isDirectory: true)
    }
}

/// 正式評価の入力 1 件（`<参照>_<話者>.wav`）
struct FormalEvalInput: Equatable {
    let speaker: String
    let reference: String
    let url: URL

    /// 入力のファイル名（拡張子なし）
    var stem: String { "\(reference)_\(speaker)" }
    /// 出力のファイル名（拡張子なし）。FAD で参照音 `<参照>.wav` と対応づけられるよう、参照音名と話者 ID を残す
    var outputStem: String { "\(reference)_\(speaker)_pronounse_ios" }

    /// ファイル名を `<参照>_<話者>` に分ける。話者フォルダの名前とファイル名の話者が違えば nil
    static func parse(url: URL, speaker: String) -> FormalEvalInput? {
        let stem = url.deletingPathExtension().lastPathComponent
        let suffix = "_\(speaker)"
        guard url.pathExtension.lowercased() == "wav", stem.hasSuffix(suffix), stem.count > suffix.count else {
            return nil
        }
        return FormalEvalInput(speaker: speaker, reference: String(stem.dropLast(suffix.count)), url: url)
    }

    /// 入力フォルダの一覧（話者・ファイル名の順）。命名規則に合わないファイルは `rejected` に入れる
    static func list(root: URL, config: FormalEvalConfig) throws -> (inputs: [FormalEvalInput], rejected: [String]) {
        let fm = FileManager.default
        let speakerDirs = try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey])
            .filter({ (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true })
            .map({ $0.lastPathComponent })
            .sorted()
        let speakers = config.speakers ?? speakerDirs
        var inputs: [FormalEvalInput] = []
        var rejected: [String] = []
        for speaker in speakers {
            let dir = root.appendingPathComponent(speaker, isDirectory: true)
            guard speakerDirs.contains(speaker) else {
                rejected.append("\(speaker)/（話者フォルダが無い）")
                continue
            }
            var perSpeaker: [FormalEvalInput] = []
            for url in try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where !url.lastPathComponent.hasPrefix(".") {
                guard let input = parse(url: url, speaker: speaker) else {
                    rejected.append("\(speaker)/\(url.lastPathComponent)")
                    continue
                }
                if let names = config.names, !names.contains(input.stem) { continue }
                perSpeaker.append(input)
            }
            if let limit = config.limitPerSpeaker { perSpeaker = Array(perSpeaker.prefix(limit)) }
            inputs += perSpeaker
        }
        return (inputs, rejected)
    }
}
