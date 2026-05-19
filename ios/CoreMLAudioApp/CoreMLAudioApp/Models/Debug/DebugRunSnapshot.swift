import CoreML
import CryptoKit
import Foundation

/// 1 回の合成パイプライン実行を `Documents/Result/debug/<runId>/` にまるごと書き出す。
///
/// F32 × cpuAndGPU の「auto-test 時だけ quiet」問題の調査専用コード。
/// `CMLA_DEBUG_SNAPSHOT=1` が launchEnvironment に入っていないと一切動かないので、
/// 通常の使用には影響しない。本番出力 (`Documents/Result/{mel,timing}/`) は別経路。
///
/// 各 run は以下を書き出す:
/// - `mel_normalized.npy` ([T, nMels] float32) — Encoder にそのまま渡される正規化済みメル
/// - `encoder_output.npy` ([T, hidden] float32) — Encoder の出力 memory (転置して 2D 化)
/// - `postnet_output.npy` ([T, nMels] float32) — Decoder postnet 出力 (HiFi-GAN への入力)
/// - `waveform.npy` ([1, N] float32) — HiFi-GAN 出力 (デエンファシス前)
/// - `summary.json` — shape, sha256, min/max/mean/rms, 環境情報
///
/// 命名は `<timestamp>_<label>_<precision>_<computeUnit>/`。`label` は launchEnvironment の
/// `CMLA_DEBUG_RUN_LABEL` から拾う (例: `freshFirstRun1`, `withSleep10`).
struct DebugRunSnapshot {

    let runDir: URL
    let runId: String

    private init(runDir: URL, runId: String) {
        self.runDir = runDir
        self.runId = runId
    }

    static func isEnabled() -> Bool {
        let env = ProcessInfo.processInfo.environment
        return env["CMLA_DEBUG_SNAPSHOT"] == "1"
    }

    static func currentLabel() -> String? {
        ProcessInfo.processInfo.environment["CMLA_DEBUG_RUN_LABEL"]
    }

    /// debug 有効時に新規 run ディレクトリを作って返す。無効時は nil。
    static func makeIfEnabled(precision: String, computeUnit: String) -> DebugRunSnapshot? {
        guard isEnabled() else { return nil }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let baseDir = docs
            .appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("debug", isDirectory: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let ts = formatter.string(from: Date())
        let label = currentLabel().map { "_\($0)" } ?? ""
        let runId = "\(ts)\(label)_\(precision)_\(computeUnit)"
        let runDir = baseDir.appendingPathComponent(runId, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        } catch {
            print("[DebugRunSnapshot] runDir create failed: \(error.localizedDescription)")
            return nil
        }
        return DebugRunSnapshot(runDir: runDir, runId: runId)
    }

    // MARK: - Writers

    /// [rows × cols] float32 を .npy + 統計 + sha256 で記録。
    func writeFloat2D(name: String, data: [Float], rows: Int, cols: Int) {
        let url = runDir.appendingPathComponent("\(name).npy")
        do {
            try NpyWriter.writeFloat32(data, rows: rows, cols: cols, to: url)
        } catch {
            print("[DebugRunSnapshot] writeFloat2D \(name) failed: \(error.localizedDescription)")
        }
        let stats = ArrayStats.compute(from: data)
        let hash = sha256(of: data)
        appendSummaryLine(
            "\(name): shape=[\(rows),\(cols)] min=\(stats.min) max=\(stats.max) mean=\(stats.mean) sha256=\(hash)"
        )
    }

    /// 1D float32 を [1, N] .npy で記録。
    func writeFloat1D(name: String, data: [Float]) {
        let url = runDir.appendingPathComponent("\(name).npy")
        do {
            try NpyWriter.writeFloat32(data, rows: 1, cols: data.count, to: url)
        } catch {
            print("[DebugRunSnapshot] writeFloat1D \(name) failed: \(error.localizedDescription)")
        }
        let stats = ArrayStats.compute(from: data)
        var sumSq: Double = 0
        for v in data where v.isFinite { sumSq += Double(v) * Double(v) }
        let rms = data.isEmpty ? Float.nan : Float((sumSq / Double(data.count)).squareRoot())
        let hash = sha256(of: data)
        appendSummaryLine(
            "\(name): shape=[1,\(data.count)] min=\(stats.min) max=\(stats.max) mean=\(stats.mean) rms=\(rms) sha256=\(hash)"
        )
    }

    /// MLMultiArray を [outerDim, innerDim] の 2D float32 として書き出す。
    /// 元の rank は問わず、count = outerDim × innerDim を満たせばよい。
    func writeMlArray(name: String, array: MLMultiArray, outerDim: Int, innerDim: Int) {
        let count = outerDim * innerDim
        guard array.count >= count else {
            print("[DebugRunSnapshot] writeMlArray \(name): size mismatch (have \(array.count), need \(count))")
            return
        }
        var buf = [Float](repeating: 0, count: count)
        for i in 0..<count {
            buf[i] = array[i].floatValue
        }
        writeFloat2D(name: name, data: buf, rows: outerDim, cols: innerDim)
    }

    /// 任意の 1 行の補助情報 (例: "stage=encoder predictMs=15.3") を summary に追記。
    func appendSummaryLine(_ line: String) {
        let url = runDir.appendingPathComponent("summary.txt")
        let entry = line + "\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            if let data = entry.data(using: .utf8) {
                try? handle.write(contentsOf: data)
            }
        } else {
            try? entry.data(using: .utf8)?.write(to: url, options: .atomic)
        }
    }

    /// "実行コンテキスト" を summary 冒頭に積む。誰がいつ何の設定で走らせたか。
    func writeContext(precision: String, computeUnit: String, shapeMode: String, inputURL: URL) {
        let env = ProcessInfo.processInfo.environment
        let testRunner = env["XCTestBundlePath"] != nil || env["XCInjectBundleInto"] != nil
        let label = env["CMLA_DEBUG_RUN_LABEL"] ?? "(none)"
        let pid = ProcessInfo.processInfo.processIdentifier
        appendSummaryLine("runId=\(runId)")
        appendSummaryLine("precision=\(precision) computeUnit=\(computeUnit) shapeMode=\(shapeMode)")
        appendSummaryLine("inputURL=\(inputURL.lastPathComponent)")
        appendSummaryLine("label=\(label) pid=\(pid) underXCTestRunner=\(testRunner)")
        appendSummaryLine("processName=\(ProcessInfo.processInfo.processName)")
        appendSummaryLine("uptime=\(ProcessInfo.processInfo.systemUptime)")
    }

    // MARK: - Decoder step CSV

    /// Decoder ループの各 step ごとに 1 行追記する。
    /// CSV: `step,mel_sha,mel_min,mel_max,mel_mean,post_sha,post_min,post_max,post_mean`
    ///
    /// 262 step × 数十バイトなので 1 ラン 30KB 程度。
    /// quiet run と loud run の CSV を diff すれば、どの step から sha が分岐するか即わかる。
    func appendDecoderStep(
        step: Int,
        melOut: MLMultiArray,
        postnetOut: MLMultiArray?
    ) {
        let url = runDir.appendingPathComponent("decoder_steps.csv")
        if !FileManager.default.fileExists(atPath: url.path) {
            let header = "step,mel_sha,mel_min,mel_max,mel_mean,post_sha,post_min,post_max,post_mean\n"
            try? header.data(using: .utf8)?.write(to: url, options: .atomic)
        }

        let melStats = ArrayStats.compute(from: melOut)
        let melSha = sha256(of: melOut)
        let (postSha, postStats): (String, ArrayStats) = {
            if let p = postnetOut {
                return (sha256(of: p), ArrayStats.compute(from: p))
            }
            return ("-", ArrayStats(min: 0, max: 0, mean: 0, hasNaN: false, hasInf: false))
        }()

        let row = String(
            format: "%d,%@,%.6e,%.6e,%.6e,%@,%.6e,%.6e,%.6e\n",
            step,
            String(melSha.prefix(16)),
            melStats.min, melStats.max, melStats.mean,
            String(postSha.prefix(16)),
            postStats.min, postStats.max, postStats.mean
        )
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            if let data = row.data(using: .utf8) {
                try? handle.write(contentsOf: data)
            }
        }
    }

    // MARK: - sha256

    private func sha256(of data: [Float]) -> String {
        var hasher = SHA256()
        data.withUnsafeBufferPointer { buf in
            if let base = buf.baseAddress {
                let bytes = UnsafeRawBufferPointer(start: base, count: buf.count * MemoryLayout<Float>.size)
                hasher.update(bufferPointer: bytes)
            }
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func sha256(of array: MLMultiArray) -> String {
        // MLMultiArray は dataPointer から生バイトで読み取れるが、stride/dtype が変動するので
        // float に正規化してから sha を取る (どの run でも同じ計算)。
        var buf = [Float](repeating: 0, count: array.count)
        for i in 0..<array.count {
            buf[i] = array[i].floatValue
        }
        return sha256(of: buf)
    }
}
