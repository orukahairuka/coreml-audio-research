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
/// したがって 1 セルにつき **初回ロード** と **キャッシュ後ロード** の 2 点を採る。
///
/// ## クラッシュ／ハング耐性（実機で観測した事象への対応）
///
/// 実機で `Float16 × cpuAndNE` のロードが返らず、最終的に **OS にプロセスごと kill** された。
/// ANE 経路のロードはハングを超えてアプリを落とすことがある。そこで:
///
/// 1. **再開可能**: CSV は固定名 (`load_timing.csv`)。再起動すると既に記録済みのセルを
///    スキップし、クラッシュ直前に書いた crumb (`load_timing_current.txt`) のセルを
///    `crashed_on_load` として記録してから続きを測る。→ 落ちたら**再起動するだけ**で進む。
/// 2. **ウォッチドッグ**: 1 セルのロードが `timeoutSec` を超えたら `load_timeout` として
///    打ち切り、次セルへ（プロセスを巻き込まないハング向け）。
/// 3. **順序**: 安全な `cpuOnly → cpuAndGPU` を先に、ANE を使う `cpuAndNE → all` を後に
///    回し、落ちる前に確実なデータを確定させる。
final class ModelLoadTimingTester {

    // MARK: - 設定

    /// 本番グリッド。shape は本番の fixed262 に固定する。
    static let precisions: [ModelPrecision] = ModelPrecision.allCases
    /// 安全側 → ANE 側の順に並べる（落ちる前に安全セルを確定させるため）。
    static let computeUnitOptions: [ComputeUnitOption] = [.cpuOnly, .cpuAndGPU, .cpuAndNE, .all]
    static let shapeMode: ShapeModeOption = .fixed262

    /// 1 セルのロードがこれを超えたら `load_timeout` として打ち切る。
    static let timeoutSec: Double = 90

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
        /// "success" | "model_missing" | "load_failed" | "load_timeout" | "crashed_on_load"
        let status: String
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
    /// 既に CSV に記録済みのセルはスキップし、クラッシュ痕跡のセルは `crashed_on_load` で記録する。
    func runAll(onProgress: @MainActor @escaping (String) -> Void) async -> (results: [Result], csvURL: URL?) {
        let dir: URL
        do {
            dir = try Self.resultDirectory()
        } catch {
            await MainActor.run { onProgress("結果ディレクトリ作成失敗: \(error.localizedDescription)") }
            return ([], nil)
        }

        let csvURL = dir.appendingPathComponent("load_timing.csv")
        let crumbURL = dir.appendingPathComponent("load_timing_current.txt")

        // 初回のみヘッダを書く（再開時は追記）。
        if !FileManager.default.fileExists(atPath: csvURL.path) {
            let header = "precision,compute_units,shape_mode,status,"
                + "encoder_first_ms,decoder_first_ms,hifigan_first_ms,total_first_ms,"
                + "encoder_cached_ms,decoder_cached_ms,hifigan_cached_ms,total_cached_ms,error\n"
            try? header.write(to: csvURL, atomically: true, encoding: .utf8)
            print("[LoadTimingTester] CSV path: \(csvURL.path)")
        }

        // 既に記録済みのセル集合を CSV から復元する。
        var done = Self.recordedCells(in: csvURL)

        // クラッシュ痕跡: 前回ここで落ちた、というセルを crashed_on_load で記録する。
        if let crashed = Self.readCrumbCell(at: crumbURL), !done.contains(Self.key(crashed.precision, crashed.computeUnits)) {
            let r = Result(
                precision: crashed.precision, computeUnits: crashed.computeUnits,
                shapeMode: Self.shapeMode.rawValue, status: "crashed_on_load",
                encoder: nil, decoder: nil, hifigan: nil,
                error: "前回このセルのロード中にプロセスが終了（ハング/クラッシュ）"
            )
            Self.append(result: r, to: csvURL)
            done.insert(Self.key(crashed.precision, crashed.computeUnits))
            print("[LoadTimingTester] crash recovered: \(crashed.precision)/\(crashed.computeUnits) → crashed_on_load")
        }

        var results: [Result] = []
        let totalCases = Self.precisions.count * Self.computeUnitOptions.count
        var caseIndex = 0

        for precision in Self.precisions {
            for cu in Self.computeUnitOptions {
                caseIndex += 1
                if done.contains(Self.key(precision.rawValue, cu.rawValue)) {
                    print("[LoadTimingTester] (\(caseIndex)/\(totalCases)) \(precision.rawValue)/\(cu.rawValue): skip (記録済み)")
                    continue
                }

                let label = "\(precision.rawValue) / \(cu.rawValue) / \(Self.shapeMode.rawValue)"
                await MainActor.run { onProgress("(\(caseIndex)/\(totalCases)) \(label)") }
                print("[LoadTimingTester] (\(caseIndex)/\(totalCases)) \(label): start")

                // クラッシュ時の手がかりとして「いま測っているセル」を残す。
                try? "\(precision.rawValue) / \(cu.rawValue) / \(Self.shapeMode.rawValue)\n"
                    .write(to: crumbURL, atomically: true, encoding: .utf8)

                let r = await measureCellWithWatchdog(precision: precision, computeUnit: cu)
                results.append(r)
                Self.append(result: r, to: csvURL)
                done.insert(Self.key(precision.rawValue, cu.rawValue))

                let firstPart = r.totalFirstMs.map { String(format: ", first=%.1fms", $0) } ?? ""
                let cachedPart = r.totalCachedMs.map { String(format: ", cached=%.1fms", $0) } ?? ""
                let errorPart = r.error.map { ", error=\($0)" } ?? ""
                print("[LoadTimingTester]   → status=\(r.status)" + firstPart + cachedPart + errorPart)

                // セルが正常に記録できたので crumb は消す（途中終了でないことの証拠）。
                try? FileManager.default.removeItem(at: crumbURL)
            }
        }

        let allDone = done.count >= totalCases
        let msg = allDone ? "ロード計測完了 (\(done.count)/\(totalCases))"
                          : "途中まで記録 (\(done.count)/\(totalCases))。再起動で続きから"
        await MainActor.run { onProgress(msg) }
        return (results, csvURL)
    }

    /// 1 セルをウォッチドッグ付きで計測する。`timeoutSec` 超過で `load_timeout` を返す。
    ///
    /// `withTaskGroup` を使わない理由: タスクグループはクロージャ末尾で**残りの子タスク
    /// 全部の完了を暗黙に待つ**（バリア）。`MLModel(contentsOf:)` は同期ブロッキングで
    /// `Task.isCancelled` を見ないため `cancelAll()` でも止まらず、ロードが本当にハングすると
    /// この関数自体が返れなくなる（＝ウォッチドッグが本来守りたいハング時に効かない）。
    /// そこでロード本体を専用スレッドに逃がし、タイムアウトと「先に終わった方で **1 度だけ**
    /// resume する」unstructured な race にする。ハングしてもタイムアウト側で確実に返り、
    /// 取り残したロードスレッドは abandon する（次セルへ進む）。
    ///
    /// 既知の限界: abandon したスレッドが ANE 資源を掴んだまま生き残り、後から
    /// プロセスを kill すると、次セルが `crashed_on_load` と誤記録され得る（稀）。
    /// その場合 `load_timeout` を出した**次の**セルの `crashed_on_load` は疑ってよい。
    private func measureCellWithWatchdog(precision: ModelPrecision, computeUnit: ComputeUnitOption) async -> Result {
        let timeoutSec = Self.timeoutSec
        let shapeMode = Self.shapeMode.rawValue
        let precisionName = precision.rawValue
        let computeUnitName = computeUnit.rawValue
        let gate = SingleResume<Result>()

        return await withCheckedContinuation { continuation in
            // ロード本体（同期ブロッキング）を専用スレッドで走らせる。
            Thread.detachNewThread {
                let r = Self.measureCell(precision: precision, computeUnit: computeUnit)
                gate.resume(continuation, with: r)
            }
            // ウォッチドッグ。ロードが返らなくてもここで必ず resume される。
            DispatchQueue.global().asyncAfter(deadline: .now() + timeoutSec) {
                let timeout = Result(
                    precision: precisionName, computeUnits: computeUnitName,
                    shapeMode: shapeMode, status: "load_timeout",
                    encoder: nil, decoder: nil, hifigan: nil,
                    error: "ロードが \(Int(timeoutSec))s 以内に返らなかった"
                )
                gate.resume(continuation, with: timeout)
            }
        }
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

    // MARK: - 再開サポート

    static func key(_ precision: String, _ computeUnits: String) -> String { "\(precision)|\(computeUnits)" }

    /// CSV を読み、既に記録済みの (precision, computeUnits) キー集合を返す。
    private static func recordedCells(in csvURL: URL) -> Set<String> {
        guard let text = try? String(contentsOf: csvURL, encoding: .utf8) else { return [] }
        var set: Set<String> = []
        for line in text.split(separator: "\n").dropFirst() {   // ヘッダを除く
            let cols = line.split(separator: ",", omittingEmptySubsequences: false)
            if cols.count >= 2 {
                set.insert(key(String(cols[0]), String(cols[1])))
            }
        }
        return set
    }

    /// crumb ファイル（"Float16 / cpuAndNE / fixed262"）から precision と computeUnits を取り出す。
    private static func readCrumbCell(at crumbURL: URL) -> (precision: String, computeUnits: String)? {
        guard let text = try? String(contentsOf: crumbURL, encoding: .utf8) else { return nil }
        let parts = text.split(separator: "/").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard parts.count >= 2 else { return nil }
        return (parts[0], parts[1])
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

/// continuation を「最初の 1 回だけ」resume するスレッドセーフなガード。
/// ロード本体スレッドとウォッチドッグが競争するため、二重 resume（クラッシュ要因）を防ぐ。
private final class SingleResume<Value> {
    private let lock = NSLock()
    private var resumed = false

    func resume(_ continuation: CheckedContinuation<Value, Never>, with value: Value) {
        lock.lock()
        if resumed {
            lock.unlock()
            return
        }
        resumed = true
        lock.unlock()
        continuation.resume(returning: value)
    }
}
