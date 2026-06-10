import CoreML
import Foundation

/// 「セルをタップしてから音を出す準備（＝3モデルのロード）に何 ms 待たされるか」を
/// 本番グリッド（精度3 × computeUnits 4 = 12 セル, shape=fixed262）で計測するハーネス。
///
/// ## なぜ「初回」と「キャッシュ後」を分けるか
///
/// `.mlmodelc` はビルド時にコンパイル済みだが、`MLModel(contentsOf:configuration:)` は
/// 各 computeUnits 向けに**初回だけ特殊化（specialize / プラン作成）**を走らせる。これが
/// 初回ロードの主成分。2 回目以降は OS のコンパイル済みキャッシュに当たるので速い。
/// ユーザー体感では「各セルを初めて触った瞬間だけ遅く、戻ってきたら速い」。
/// したがって 1 セルにつき **初回ロード** と **キャッシュ後ロード** の 2 点を採る。
///
/// ## 計測の前提（真の初回を採るには）
///
/// OS のコンパイルキャッシュはプロセスをまたいで残る。よって、あるセルを一度でも
/// ロード済み（例: 先に「合成実行」した）だと、その後のスイープでは「初回」も実は
/// キャッシュ後の値になる。**真の初回が欲しければ、アプリを起動した直後・合成を一度も
/// 走らせる前にこのスイープを実行すること。**
///
/// ## 例外捕捉
///
/// `MLModel(contentsOf:)` が Swift の `throws` で投げるエラーは捕捉できるが、E5RT 系の
/// プロセス即死は捕捉できない。`VocoderStabilityTester` と同様、結果は 1 セルごとに即時
/// CSV 追記し、途中で落ちても「どこまで測れたか」が残るようにする。
final class ModelLoadTimingTester {

    // MARK: - 設定

    /// 本番グリッド。shape は本番の fixed262 に固定する。
    static let precisions: [ModelPrecision] = ModelPrecision.allCases
    static let computeUnitOptions: [ComputeUnitOption] = ComputeUnitOption.allCases
    static let shapeMode: ShapeModeOption = .fixed262

    // MARK: - 結果

    /// 1 モデルぶんのロード時間（初回 / キャッシュ後）
    struct ModelLoad {
        let resourceName: String
        let firstMs: Double?
        let cachedMs: Double?
    }

    /// 1 セル（精度 × computeUnits）ぶんの計測結果
    struct Result {
        let precision: String
        let computeUnits: String
        let shapeMode: String
        let status: String        // "success" | "model_missing" | "load_failed"
        let encoder: ModelLoad?
        let decoder: ModelLoad?
        let hifigan: ModelLoad?
        let error: String?

        /// 3 モデル合計の初回ロード時間（ユーザーが初回タップで待つ総時間）
        var totalFirstMs: Double? {
            guard let e = encoder?.firstMs, let d = decoder?.firstMs, let h = hifigan?.firstMs else { return nil }
            return e + d + h
        }

        /// 3 モデル合計のキャッシュ後ロード時間
        var totalCachedMs: Double? {
            guard let e = encoder?.cachedMs, let d = decoder?.cachedMs, let h = hifigan?.cachedMs else { return nil }
            return e + d + h
        }
    }

    // MARK: - 実行

    /// 12 セルを順にロード計測し、結果配列と CSV パスを返す。
    func runAll(onProgress: @MainActor @escaping (String) -> Void) async -> (results: [Result], csvURL: URL?) {
        let dir: URL
        do {
            dir = try Self.resultDirectory()
        } catch {
            await MainActor.run { onProgress("結果ディレクトリ作成失敗: \(error.localizedDescription)") }
            return ([], nil)
        }

        let timestamp = Self.timestamp()
        let csvURL = dir.appendingPathComponent("load_timing_\(timestamp).csv")
        let crumbURL = dir.appendingPathComponent("load_timing_\(timestamp)_current.txt")

        let header = "precision,compute_units,shape_mode,status,"
            + "encoder_first_ms,decoder_first_ms,hifigan_first_ms,total_first_ms,"
            + "encoder_cached_ms,decoder_cached_ms,hifigan_cached_ms,total_cached_ms,error\n"
        try? header.write(to: csvURL, atomically: true, encoding: .utf8)
        print("[LoadTimingTester] CSV path: \(csvURL.path)")
        print("[CSV] \(header)", terminator: "")

        var results: [Result] = []
        let totalCases = Self.precisions.count * Self.computeUnitOptions.count
        var caseIndex = 0

        for precision in Self.precisions {
            for cu in Self.computeUnitOptions {
                caseIndex += 1
                let label = "\(precision.rawValue) / \(cu.rawValue) / \(Self.shapeMode.rawValue)"
                await MainActor.run { onProgress("(\(caseIndex)/\(totalCases)) \(label)") }
                print("[LoadTimingTester] (\(caseIndex)/\(totalCases)) \(label): start")
                try? "\(label)\n".write(to: crumbURL, atomically: true, encoding: .utf8)

                let r = Self.measureCell(precision: precision, computeUnit: cu)
                results.append(r)
                Self.append(result: r, to: csvURL)
                let firstPart = r.totalFirstMs.map { String(format: ", first=%.1fms", $0) } ?? ""
                let cachedPart = r.totalCachedMs.map { String(format: ", cached=%.1fms", $0) } ?? ""
                let errorPart = r.error.map { ", error=\($0)" } ?? ""
                print("[LoadTimingTester]   → status=\(r.status)" + firstPart + cachedPart + errorPart)
            }
        }

        try? FileManager.default.removeItem(at: crumbURL)
        let doneCount = results.count
        let csvName = csvURL.lastPathComponent
        await MainActor.run { onProgress("ロード計測完了 (\(doneCount) 件) → \(csvName)") }
        return (results, csvURL)
    }

    /// 1 セル分: Encoder / Decoder / HiFi-GAN を「初回 → キャッシュ後」の順に 2 回ロードして計時する。
    private static func measureCell(precision: ModelPrecision, computeUnit: ComputeUnitOption) -> Result {
        let encoderName = shapeMode.transformerEncoderResourceName(for: precision)
        let decoderName = shapeMode.transformerDecoderResourceName(for: precision)
        guard let hifiganName = shapeMode.hifiganResourceName(for: precision) else {
            return Result(
                precision: precision.rawValue, computeUnits: computeUnit.rawValue,
                shapeMode: shapeMode.rawValue, status: "model_missing",
                encoder: nil, decoder: nil, hifigan: nil,
                error: "HiFiGAN \(precision.suffix)_\(shapeMode.rawValue) が Bundle にない"
            )
        }
        guard let encoderURL = Bundle.main.url(forResource: encoderName, withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: decoderName, withExtension: "mlmodelc"),
              let hifiganURL = Bundle.main.url(forResource: hifiganName, withExtension: "mlmodelc") else {
            return Result(
                precision: precision.rawValue, computeUnits: computeUnit.rawValue,
                shapeMode: shapeMode.rawValue, status: "model_missing",
                encoder: nil, decoder: nil, hifigan: nil,
                error: "Encoder/Decoder/HiFiGAN のいずれかが Bundle にない (\(encoderName) / \(decoderName) / \(hifiganName))"
            )
        }

        let units = computeUnit.mlComputeUnits
        do {
            // 初回ロード（特殊化を含む） → 直後にもう一度ロード（OS キャッシュに当たる）。
            // 各モデルを 2 回連続でロードして first / cached を採る。
            let encoder = try measureModel(url: encoderURL, name: encoderName, units: units)
            let decoder = try measureModel(url: decoderURL, name: decoderName, units: units)
            let hifigan = try measureModel(url: hifiganURL, name: hifiganName, units: units)
            return Result(
                precision: precision.rawValue, computeUnits: computeUnit.rawValue,
                shapeMode: shapeMode.rawValue, status: "success",
                encoder: encoder, decoder: decoder, hifigan: hifigan, error: nil
            )
        } catch {
            return Result(
                precision: precision.rawValue, computeUnits: computeUnit.rawValue,
                shapeMode: shapeMode.rawValue, status: "load_failed",
                encoder: nil, decoder: nil, hifigan: nil,
                error: error.localizedDescription
            )
        }
    }

    /// 同じモデルを 2 回ロードし、(初回 ms, キャッシュ後 ms) を返す。
    /// 1 回目のインスタンスは 2 回目を測る前に解放し、純粋な再ロードコストを測る。
    private static func measureModel(url: URL, name: String, units: MLComputeUnits) throws -> ModelLoad {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units

        let t0 = CFAbsoluteTimeGetCurrent()
        var model: MLModel? = try MLModel(contentsOf: url, configuration: cfg)
        let firstMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        model = nil   // 1 回目を解放してから 2 回目を測る
        _ = model

        let cfg2 = MLModelConfiguration()
        cfg2.computeUnits = units
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = try MLModel(contentsOf: url, configuration: cfg2)
        let cachedMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        return ModelLoad(resourceName: name, firstMs: firstMs, cachedMs: cachedMs)
    }

    // MARK: - 出力

    static func resultDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("load_timing", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f.string(from: Date())
    }

    /// 1 セル分を CSV に追記する（fsync 付き + stdout ミラー）。
    static func append(result r: Result, to url: URL) {
        let row = csvRow(r)
        print("[CSV] \(row)", terminator: "")
        guard let data = row.data(using: .utf8) else { return }
        do {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
        } catch {
            print("[LoadTimingTester] CSV 追記失敗: \(error.localizedDescription)")
        }
    }

    static func csvRow(_ r: Result) -> String {
        func d(_ x: Double?) -> String { x.map { String(format: "%.3f", $0) } ?? "" }
        func esc(_ s: String?) -> String {
            guard let s, !s.isEmpty else { return "" }
            let escaped = s.replacingOccurrences(of: "\"", with: "\"\"")
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "\r", with: " ")
            return "\"\(escaped)\""
        }
        return [
            r.precision,
            r.computeUnits,
            r.shapeMode,
            r.status,
            d(r.encoder?.firstMs),
            d(r.decoder?.firstMs),
            d(r.hifigan?.firstMs),
            d(r.totalFirstMs),
            d(r.encoder?.cachedMs),
            d(r.decoder?.cachedMs),
            d(r.hifigan?.cachedMs),
            d(r.totalCachedMs),
            esc(r.error),
        ].joined(separator: ",") + "\n"
    }
}
