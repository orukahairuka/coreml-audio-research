import CoreML
import Foundation

/// HiFi-GAN を「精度 × shape mode × computeUnits」の組み合わせで網羅的に実行し、
/// それぞれの可否・出力統計・推論時間を CSV で記録するための検証ハーネス。
///
/// 目的は「実機でどの組み合わせなら安定して動くか」を切り分けることなので、
/// `cpuOnly` / `cpuAndGPU` / `all` をまたいで同じ入力を流し、結果を比較できるようにする。
///
/// ## 例外捕捉について
///
/// `model.prediction(...)` や `MLModel(contentsOf:)` が **Swift の `throws` で投げる
/// エラー** は `do/try/catch` で確実に捕捉できる。一方、Core ML / Metal / BNNS 内部で
/// `NSException` 等のメカニズムでクラッシュする場合 (E5RT 系のメモリポートエラー、
/// `MTLAssert` の abort、Metal command buffer の致命的失敗) は **Swift から完全には
/// 捕捉できない** ため、テスト中にプロセスごと落ちる可能性がある。
///
/// その対策として、結果は組み合わせごとに即時 CSV に追記している。クラッシュ時には
/// 「何を試している途中で落ちたか」が `currentCombination` ファイルに残る。
final class VocoderStabilityTester {

    // MARK: - 設定

    struct Variant {
        let resourceName: String   // Bundle 内の .mlmodelc 名（拡張子なし）
        let precision: String      // "Float32" | "Float16" | "Int8"
        let shapeMode: String      // "range1" | "range16" | "range16_384" | "fixed262" | "range16_1000" (legacy int8)
    }

    /// `convert_hifigan.py --all-variants` で生成される 7 種類 + 既存 legacy Int8 = 8 モデル。
    /// Bundle に追加されていないものは "model_missing" として記録する。
    /// Int8 は今のところ legacy 命名 (RangeDim 16-1000) のみ存在するので、Int8 + GPU 経路 (.cpuAndGPU / .all) で
    /// 実機が落ちないか / 出力が壊れないかを Float32/Float16 と同じ手順で確認する。
    static let variants: [Variant] = [
        .init(resourceName: "HiFiGAN_Generator_float32_range1",      precision: "Float32", shapeMode: "range1"),
        .init(resourceName: "HiFiGAN_Generator_float32_range16",     precision: "Float32", shapeMode: "range16"),
        .init(resourceName: "HiFiGAN_Generator_float32_range16_384", precision: "Float32", shapeMode: "range16_384"),
        .init(resourceName: "HiFiGAN_Generator_float32_fixed262",    precision: "Float32", shapeMode: "fixed262"),
        .init(resourceName: "HiFiGAN_Generator_float16_range16",     precision: "Float16", shapeMode: "range16"),
        .init(resourceName: "HiFiGAN_Generator_float16_range16_384", precision: "Float16", shapeMode: "range16_384"),
        .init(resourceName: "HiFiGAN_Generator_float16_fixed262",    precision: "Float16", shapeMode: "fixed262"),
        .init(resourceName: "HiFiGAN_Generator_int8",                precision: "Int8",    shapeMode: "range16_1000"),
    ]

    static let computeUnitOptions: [(label: String, units: MLComputeUnits)] = [
        ("cpuOnly",   .cpuOnly),
        ("cpuAndGPU", .cpuAndGPU),
        ("all",       .all),
    ]

    /// 入力 mel のフレーム数。実合成パイプラインで観測される代表値 (262) を使う。
    /// 全モデルがこの T を受理できる (fixed262 はそのもの、RangeDim 系は 16〜1000 内)。
    static let inputT: Int = 262
    static let nMels: Int = 256

    // MARK: - 結果

    struct Result {
        let modelName: String
        let precision: String
        let shapeMode: String
        let computeUnits: String
        let inputT: Int
        let status: String        // "success" | "model_missing" | "load_failed" | "predict_failed"
        let predictMs: Double?
        let stats: WaveformStats?
        let error: String?
    }

    struct WaveformStats {
        let min: Float
        let max: Float
        let mean: Float
        let rms: Float
        let nanCount: Int
        let infCount: Int
        let ratioOver099: Float
        let ratioUnderMinus099: Float
    }

    // MARK: - 実行

    /// すべての組み合わせを順に実行し、結果配列と CSV パスを返す。
    /// 進捗は `onProgress(message)` でメインスレッドに報告する。
    func runAll(onProgress: @MainActor @escaping (String) -> Void) async -> (results: [Result], csvURL: URL?) {
        let dir: URL
        do {
            dir = try Self.resultDirectory()
        } catch {
            await MainActor.run { onProgress("結果ディレクトリ作成失敗: \(error.localizedDescription)") }
            return ([], nil)
        }

        let timestamp = Self.timestamp()
        let csvURL = dir.appendingPathComponent("vocoder_stability_\(timestamp).csv")
        let crumbURL = dir.appendingPathComponent("vocoder_stability_\(timestamp)_current.txt")

        // CSV ヘッダを書く
        let header = "precision,shape_mode,compute_units,input_T,status,min,max,mean,rms,nan_count,inf_count,ratio_over_099,ratio_under_minus_099,predict_ms,error\n"
        try? header.write(to: csvURL, atomically: true, encoding: .utf8)

        // クラッシュで Files.app からも取り出せない事態に備えて、CSV のフルパスと
        // ヘッダを最初に print しておく。Xcode デバッグ中なら確実にコンソールに残る。
        print("[StabilityTester] CSV path: \(csvURL.path)")
        print("[CSV] \(header)", terminator: "")

        // 共通入力 (再現性のため決定的乱数で生成)
        let melInput = Self.makeDeterministicMel(t: Self.inputT, nMels: Self.nMels)

        var results: [Result] = []
        let totalCases = Self.variants.count * Self.computeUnitOptions.count
        var caseIndex = 0

        for variant in Self.variants {
            // Bundle にない場合は computeUnits ごとに model_missing を 3 行記録して次へ
            guard let modelURL = Bundle.main.url(
                forResource: variant.resourceName, withExtension: "mlmodelc"
            ) else {
                for cu in Self.computeUnitOptions {
                    caseIndex += 1
                    let r = Result(
                        modelName: variant.resourceName,
                        precision: variant.precision,
                        shapeMode: variant.shapeMode,
                        computeUnits: cu.label,
                        inputT: Self.inputT,
                        status: "model_missing",
                        predictMs: nil,
                        stats: nil,
                        error: "Bundle に \(variant.resourceName).mlmodelc が見つかりません"
                    )
                    results.append(r)
                    Self.append(result: r, to: csvURL)
                    print("[StabilityTester] (\(caseIndex)/\(totalCases)) \(variant.resourceName) / \(cu.label): model_missing")
                }
                continue
            }

            for cu in Self.computeUnitOptions {
                caseIndex += 1
                let progressIndex = caseIndex
                let label = "\(variant.precision) / \(variant.shapeMode) / \(cu.label)"
                let message = "(\(progressIndex)/\(totalCases)) \(label)"
                await MainActor.run { onProgress(message) }
                print("[StabilityTester] (\(progressIndex)/\(totalCases)) \(variant.resourceName) / \(cu.label): start")

                // クラッシュ時の手がかりとして「いま試している組み合わせ」を残す
                try? "\(label)\n".write(to: crumbURL, atomically: true, encoding: .utf8)

                let r = await runOne(
                    variant: variant,
                    modelURL: modelURL,
                    computeUnits: cu,
                    melInput: melInput
                )
                results.append(r)
                Self.append(result: r, to: csvURL)
                print("[StabilityTester]   → status=\(r.status)" + (r.error.map { ", error=\($0)" } ?? "")
                      + (r.predictMs.map { String(format: ", predictMs=%.1f", $0) } ?? "")
                      + (r.stats.map { ", stats: \(formatStats($0))" } ?? ""))
            }
        }

        // 走り終わったら現在進行ファイルは消しておく（クラッシュしなかった証拠）
        try? FileManager.default.removeItem(at: crumbURL)

        let finalCount = results.count
        let csvName = csvURL.lastPathComponent
        await MainActor.run { onProgress("テスト完了 (\(finalCount) 件) → \(csvName)") }
        return (results, csvURL)
    }

    /// 1 ケース分を実行。NSException を除く例外は捕捉する。
    private func runOne(
        variant: Variant,
        modelURL: URL,
        computeUnits: (label: String, units: MLComputeUnits),
        melInput: MLMultiArray
    ) async -> Result {
        // 1) ロード
        let model: MLModel
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits.units
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
        } catch {
            return Result(
                modelName: variant.resourceName,
                precision: variant.precision,
                shapeMode: variant.shapeMode,
                computeUnits: computeUnits.label,
                inputT: Self.inputT,
                status: "load_failed",
                predictMs: nil,
                stats: nil,
                error: error.localizedDescription
            )
        }

        // 2) 推論
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: melInput)
            ])
            let t0 = CFAbsoluteTimeGetCurrent()
            let output = try await model.prediction(from: provider)
            let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            guard let firstName = output.featureNames.first,
                  let array = output.featureValue(for: firstName)?.multiArrayValue else {
                return Result(
                    modelName: variant.resourceName,
                    precision: variant.precision,
                    shapeMode: variant.shapeMode,
                    computeUnits: computeUnits.label,
                    inputT: Self.inputT,
                    status: "predict_failed",
                    predictMs: predictMs,
                    stats: nil,
                    error: "出力 multiArray を取り出せませんでした"
                )
            }

            let stats = Self.computeStats(from: array)
            return Result(
                modelName: variant.resourceName,
                precision: variant.precision,
                shapeMode: variant.shapeMode,
                computeUnits: computeUnits.label,
                inputT: Self.inputT,
                status: "success",
                predictMs: predictMs,
                stats: stats,
                error: nil
            )
        } catch {
            return Result(
                modelName: variant.resourceName,
                precision: variant.precision,
                shapeMode: variant.shapeMode,
                computeUnits: computeUnits.label,
                inputT: Self.inputT,
                status: "predict_failed",
                predictMs: nil,
                stats: nil,
                error: error.localizedDescription
            )
        }
    }

    // MARK: - 出力差分テスト (Float16 fixed262 cpuAndGPU vs all)

    /// 同じ入力 mel を `.cpuAndGPU` と `.all` の 2 経路で流し、
    /// 出力波形の差分 (max / mean / rms / allclose 相当) を CSV と .npy で保存する。
    ///
    /// 実機テストで `.all` 側のみ振幅が大きく出る現象が観測されたため、
    /// 「ANE 経路で実際にどれくらい数値がずれているか」を定量化する目的。
    /// ここでは GPU/ANE のどちらが「正しい」かは決めず、両者の差を素直に記録する。
    func runFloat16Fixed262DiffTest(onProgress: @MainActor @escaping (String) -> Void) async -> URL? {
        let dir: URL
        do {
            dir = try Self.resultDirectory()
        } catch {
            await MainActor.run { onProgress("結果ディレクトリ作成失敗: \(error.localizedDescription)") }
            return nil
        }

        let timestamp = Self.timestamp()
        let csvURL = dir.appendingPathComponent("vocoder_diff_f16_fixed262_\(timestamp).csv")

        // ヘッダ
        let header = "pair,sample_count,max_abs_diff,mean_abs_diff,rms_diff,"
            + "allclose_atol_1e_2,allclose_atol_1e_3,allclose_atol_1e_4,"
            + "gpu_predict_ms,all_predict_ms,"
            + "gpu_min,gpu_max,gpu_rms,all_min,all_max,all_rms,error\n"
        try? header.write(to: csvURL, atomically: true, encoding: .utf8)
        print("[DiffTest] CSV path: \(csvURL.path)")
        print("[CSV] \(header)", terminator: "")

        let resourceName = "HiFiGAN_Generator_float16_fixed262"
        guard let modelURL = Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") else {
            let row = "Float16_fixed262_cpuAndGPU_vs_all,,,,,,,,,,,,,,,,\"\(resourceName).mlmodelc が Bundle にない\"\n"
            Self.appendRaw(row, to: csvURL)
            await MainActor.run { onProgress("\(resourceName).mlmodelc が見つかりません") }
            return csvURL
        }

        let mel = Self.makeDeterministicMel(t: Self.inputT, nMels: Self.nMels)

        // .cpuAndGPU 側を実行
        await MainActor.run { onProgress("cpuAndGPU 推論中…") }
        let gpu = await runOneCollectingWaveform(modelURL: modelURL, computeUnits: .cpuAndGPU, melInput: mel)
        if let err = gpu.error {
            let row = "Float16_fixed262_cpuAndGPU_vs_all,,,,,,,,,,,,,,,,\"cpuAndGPU 失敗: \(Self.csvEscape(err))\"\n"
            Self.appendRaw(row, to: csvURL)
            await MainActor.run { onProgress("cpuAndGPU 失敗: \(err)") }
            return csvURL
        }

        // .all 側を実行
        await MainActor.run { onProgress("all 推論中…") }
        let all = await runOneCollectingWaveform(modelURL: modelURL, computeUnits: .all, melInput: mel)
        if let err = all.error {
            let row = "Float16_fixed262_cpuAndGPU_vs_all,,,,,,,,,,,,,,,,\"all 失敗: \(Self.csvEscape(err))\"\n"
            Self.appendRaw(row, to: csvURL)
            await MainActor.run { onProgress("all 失敗: \(err)") }
            return csvURL
        }

        guard let gpuWave = gpu.waveform, let allWave = all.waveform else { return csvURL }

        // 波形を .npy で保存 (numpy 側で squeeze() すれば 1D)
        let gpuNpy  = dir.appendingPathComponent("vocoder_diff_f16_fixed262_\(timestamp)_cpuAndGPU.npy")
        let allNpy  = dir.appendingPathComponent("vocoder_diff_f16_fixed262_\(timestamp)_all.npy")
        let diffNpy = dir.appendingPathComponent("vocoder_diff_f16_fixed262_\(timestamp)_diff.npy")
        try? NpyWriter.writeFloat32(gpuWave, rows: 1, cols: gpuWave.count, to: gpuNpy)
        try? NpyWriter.writeFloat32(allWave, rows: 1, cols: allWave.count, to: allNpy)
        let diff = zip(gpuWave, allWave).map { $0 - $1 }
        try? NpyWriter.writeFloat32(diff, rows: 1, cols: diff.count, to: diffNpy)

        // 差分メトリクス
        let n = min(gpuWave.count, allWave.count)
        var maxAbs: Float = 0
        var sumAbs: Double = 0
        var sumSq: Double = 0
        for i in 0..<n {
            let d = abs(gpuWave[i] - allWave[i])
            if d > maxAbs { maxAbs = d }
            sumAbs += Double(d)
            sumSq += Double(d) * Double(d)
        }
        let meanAbs = n > 0 ? Float(sumAbs / Double(n)) : .nan
        let rmsDiff = n > 0 ? Float((sumSq / Double(n)).squareRoot()) : .nan
        // np.allclose 相当: |a-b| <= atol + rtol*|b|, rtol=0 にして atol だけで判定
        func allclose(atol: Float) -> Bool { return maxAbs <= atol }

        let gpuStats = Self.computeStats(from: gpuWave)
        let allStats = Self.computeStats(from: allWave)

        let row = [
            "Float16_fixed262_cpuAndGPU_vs_all",
            String(n),
            String(format: "%.6e", maxAbs),
            String(format: "%.6e", meanAbs),
            String(format: "%.6e", rmsDiff),
            allclose(atol: 1e-2) ? "True" : "False",
            allclose(atol: 1e-3) ? "True" : "False",
            allclose(atol: 1e-4) ? "True" : "False",
            String(format: "%.3f", gpu.predictMs ?? .nan),
            String(format: "%.3f", all.predictMs ?? .nan),
            String(format: "%.6f", gpuStats.min),
            String(format: "%.6f", gpuStats.max),
            String(format: "%.6f", gpuStats.rms),
            String(format: "%.6f", allStats.min),
            String(format: "%.6f", allStats.max),
            String(format: "%.6f", allStats.rms),
            "",
        ].joined(separator: ",") + "\n"
        Self.appendRaw(row, to: csvURL)

        let summary = String(
            format: "diff: max=%.4e mean=%.4e rms=%.4e allclose(1e-3)=%@",
            maxAbs, meanAbs, rmsDiff, allclose(atol: 1e-3) ? "True" : "False"
        )
        await MainActor.run { onProgress("差分テスト完了 — \(summary)") }
        print("[DiffTest] \(summary)")
        print("[DiffTest] gpu stats: min=\(gpuStats.min) max=\(gpuStats.max) rms=\(gpuStats.rms)")
        print("[DiffTest] all stats: min=\(allStats.min) max=\(allStats.max) rms=\(allStats.rms)")
        print("[DiffTest] saved: \(gpuNpy.lastPathComponent), \(allNpy.lastPathComponent), \(diffNpy.lastPathComponent)")

        return csvURL
    }

    /// 推論を実行して **波形 [Float] を保持して返す** バリアント。
    /// `runOne` は統計だけ返すので、差分テスト用に用意した。
    private func runOneCollectingWaveform(
        modelURL: URL,
        computeUnits: MLComputeUnits,
        melInput: MLMultiArray
    ) async -> (waveform: [Float]?, predictMs: Double?, error: String?) {
        let model: MLModel
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
        } catch {
            return (nil, nil, "load_failed: \(error.localizedDescription)")
        }

        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: melInput)
            ])
            let t0 = CFAbsoluteTimeGetCurrent()
            let output = try await model.prediction(from: provider)
            let predictMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            guard let firstName = output.featureNames.first,
                  let array = output.featureValue(for: firstName)?.multiArrayValue else {
                return (nil, predictMs, "predict_failed: 出力 multiArray を取り出せませんでした")
            }

            let count = array.count
            var wave = [Float](repeating: 0, count: count)
            for i in 0..<count { wave[i] = array[i].floatValue }
            return (wave, predictMs, nil)
        } catch {
            return (nil, nil, "predict_failed: \(error.localizedDescription)")
        }
    }

    /// `[Float]` から min/max/mean/rms を計算する小さなヘルパ
    static func computeStats(from wave: [Float]) -> (min: Float, max: Float, mean: Float, rms: Float) {
        if wave.isEmpty { return (.nan, .nan, .nan, .nan) }
        var lo: Float =  .infinity
        var hi: Float = -.infinity
        var sum: Double = 0
        var sumSq: Double = 0
        for v in wave {
            if v < lo { lo = v }
            if v > hi { hi = v }
            sum += Double(v)
            sumSq += Double(v) * Double(v)
        }
        let n = Double(wave.count)
        return (lo, hi, Float(sum / n), Float((sumSq / n).squareRoot()))
    }

    /// CSV 行をそのままファイル末尾に追記する (synchronize 付き)
    static func appendRaw(_ row: String, to url: URL) {
        print("[CSV] \(row)", terminator: "")
        guard let data = row.data(using: .utf8) else { return }
        do {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
        } catch {
            print("[DiffTest] CSV 追記失敗: \(error.localizedDescription)")
        }
    }

    static func csvEscape(_ s: String) -> String {
        return s.replacingOccurrences(of: "\"", with: "\"\"")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
    }

    // MARK: - 入力生成

    /// 決定的な擬似乱数で (1, nMels, T) の mel を作る。
    /// 全組み合わせで同じ値を流すため、線形合同法で seed=1 から生成する。
    static func makeDeterministicMel(t: Int, nMels: Int) -> MLMultiArray {
        let array = try! MLMultiArray(
            shape: [1, nMels as NSNumber, t as NSNumber],
            dataType: .float32
        )
        // 簡易 LCG → ボックス・ミューラーで N(0,1) 化
        var state: UInt64 = 1
        func next01() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            // 上位 53bit を [0,1) に
            let bits = (state >> 11) & 0x1F_FFFF_FFFF_FFFF
            return Double(bits) / Double(1 << 53)
        }
        var i = 0
        let total = nMels * t
        while i < total {
            let u1 = max(next01(), 1e-12)
            let u2 = next01()
            let r = (-2.0 * log(u1)).squareRoot()
            let z0 = Float(r * cos(2.0 * .pi * u2))
            let z1 = Float(r * sin(2.0 * .pi * u2))
            array[i] = NSNumber(value: z0)
            if i + 1 < total {
                array[i + 1] = NSNumber(value: z1)
            }
            i += 2
        }
        return array
    }

    // MARK: - 統計

    /// 出力波形の統計を一括計算する（min / max / mean / rms / NaN / Inf / ±0.99 比率）。
    static func computeStats(from array: MLMultiArray) -> WaveformStats {
        let count = array.count
        var minV: Float =  .infinity
        var maxV: Float = -.infinity
        var sum: Double = 0
        var sumSq: Double = 0
        var nanC = 0
        var infC = 0
        var over = 0
        var under = 0
        for i in 0..<count {
            let v = array[i].floatValue
            if v.isNaN { nanC += 1; continue }
            if v.isInfinite { infC += 1; continue }
            if v < minV { minV = v }
            if v > maxV { maxV = v }
            sum += Double(v)
            sumSq += Double(v) * Double(v)
            if v >=  0.99 { over += 1 }
            if v <= -0.99 { under += 1 }
        }
        let valid = count - nanC - infC
        let mean = valid > 0 ? Float(sum / Double(valid)) : .nan
        let rms = valid > 0 ? Float((sumSq / Double(valid)).squareRoot()) : .nan
        let denom = max(count, 1)
        return WaveformStats(
            min: (minV.isFinite ? minV : .nan),
            max: (maxV.isFinite ? maxV : .nan),
            mean: mean,
            rms: rms,
            nanCount: nanC,
            infCount: infC,
            ratioOver099: Float(over) / Float(denom),
            ratioUnderMinus099: Float(under) / Float(denom)
        )
    }

    private func formatStats(_ s: WaveformStats) -> String {
        return String(
            format: "min=%.4f max=%.4f mean=%.4e rms=%.4f nan=%d inf=%d over=%.3f under=%.3f",
            s.min, s.max, s.mean, s.rms, s.nanCount, s.infCount, s.ratioOver099, s.ratioUnderMinus099
        )
    }

    // MARK: - 出力

    static func resultDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("stability", isDirectory: true)
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

    /// 1 行ずつ追記する。
    ///
    /// クラッシュ耐性のため、毎回 open → write → **synchronize (fsync)** → close する。
    /// `close()` だけだとバッファ止まりで、Metal/BNNS の abort で即死した際に直前の
    /// 行が失われる可能性がある。`synchronize()` でディスクまで書き出してから返す。
    ///
    /// また、Xcode デバッグ実行中なら print() の方が確実に拾えるため、CSV と同一の
    /// 行を `[CSV]` プレフィックス付きで stdout にもミラーする。実機で Files.app から
    /// CSV を取り出せない場合 (権限問題、デバッガ無し起動など) でも、デバッグ実行で
    /// あればここから値を回収できる。
    static func append(result: Result, to url: URL) {
        let row = csvRow(result)
        print("[CSV] \(row)", terminator: "")
        guard let data = row.data(using: .utf8) else { return }
        do {
            let handle = try FileHandle(forWritingTo: url)
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.synchronize()
            try handle.close()
        } catch {
            print("[StabilityTester] CSV 追記失敗: \(error.localizedDescription)")
        }
    }

    static func csvRow(_ r: Result) -> String {
        func n(_ x: Float?) -> String { x.map { String(format: "%.6f", $0) } ?? "" }
        func d(_ x: Double?) -> String { x.map { String(format: "%.3f", $0) } ?? "" }
        func i(_ x: Int?) -> String { x.map(String.init) ?? "" }
        // CSV エスケープ: ダブルクォート / カンマ / 改行を含む可能性があるエラーメッセージを保護する
        func esc(_ s: String?) -> String {
            guard let s, !s.isEmpty else { return "" }
            let escaped = s.replacingOccurrences(of: "\"", with: "\"\"")
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "\r", with: " ")
            return "\"\(escaped)\""
        }

        let s = r.stats
        return [
            r.precision,
            r.shapeMode,
            r.computeUnits,
            String(r.inputT),
            r.status,
            n(s?.min),
            n(s?.max),
            n(s?.mean),
            n(s?.rms),
            i(s?.nanCount),
            i(s?.infCount),
            n(s?.ratioOver099),
            n(s?.ratioUnderMinus099),
            d(r.predictMs),
            esc(r.error),
        ].joined(separator: ",") + "\n"
    }
}
