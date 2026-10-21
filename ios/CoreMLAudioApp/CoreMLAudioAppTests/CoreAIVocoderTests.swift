import Testing
import Foundation
@testable import CoreMLAudioApp
#if canImport(CoreAI)
import CoreAI
#endif

/// Phase 2 MVP — HiFi-GAN の **Core AI 実機推論**を、cpu / gpu / neuralEngine で
/// 切り替えて回し、出力波形の統計（min/max/rms/clipping/nan/inf）と PyTorch 参照
/// との PSNR を CSV に記録する。
///
/// 目的: CoreML で観測した HiFi-GAN×Neural Engine の破綻（rms 約4.25倍→clip）が
/// 別ランタイム Core AI でも再現するかの切り分け。UI は作らず固定入力で検証する。
///
/// 前提:
/// - iOS 27 実機（iPhone(3)@27.0）。シミュレータ / macOS 26 では Core AI 推論不可。
/// - `AIModels/HiFiGAN_Generator_float32_f262.aimodel` がアプリバンドルに**ディレクトリ
///   として**含まれていること（`.aimodel` は Xcode が package 認識しないため、フォルダ
///   参照で同梱されているか要確認。されていない場合はこのテストが modelNotFound で落ちる）。
/// - `AIModels/reference/hifigan_mel_float32_f262.npy`（入力）と
///   `hifigan_pt_out_float32_f262.npy`（PyTorch 参照出力）が同梱されていること。
///
/// 出力: `Documents/Result/coreai/hifigan_compute_units.csv` と各波形の .npy。
#if canImport(CoreAI)
struct CoreAIVocoderTests {

    private static let modelBaseName = "HiFiGAN_Generator_float32_f262"
    private static let melName = "hifigan_mel_float32_f262"
    private static let refOutName = "hifigan_pt_out_float32_f262"
    private static let nMels = 256
    private static let frames = 262
    private static let hopSize = 256

    @Test
    func hifiganAcrossComputeUnits() async throws {
        // --- リソース解決 ---
        let modelURL = try locateResource(name: Self.modelBaseName, ext: "aimodel")
        let melURL = try locateResource(name: Self.melName, ext: "npy")
        let refURL = try locateResource(name: Self.refOutName, ext: "npy")

        let mel = try MinimalNpyReader.readFloat32(contentsOf: melURL)
        let reference = try MinimalNpyReader.readFloat32(contentsOf: refURL)
        #expect(mel.count == Self.nMels * Self.frames, "mel 要素数が想定外: \(mel.count)")
        // 参照長が一致しないと psnr() が短い方に黙って切り詰め、妥当そうな偽 PSNR を出す。
        // #expect は失敗を記録するだけでテストを継続し偽 PSNR が CSV に書かれてしまうため、
        // #require で PSNR 計算前に中断する。
        try #require(reference.count == Self.frames * Self.hopSize,
                     "参照出力の要素数が想定外: \(reference.count) (期待 \(Self.frames * Self.hopSize))")
        print("[CoreAIVocoderTests] mel=\(mel.count), reference=\(reference.count)")

        // --- 出力先 ---
        let outDir = try makeOutputDir()
        var csvLines: [String] = [csvHeader]

        // --- compute unit ごとに実行 ---
        for unit in CoreAIComputeUnit.allCases {
            do {
                let (runner, loadMs) = try await CoreAIVocoderRunner.load(
                    modelURL: modelURL,
                    computeUnit: unit,
                    hopSize: Self.hopSize
                )
                print("[CoreAIVocoderTests] \(unit.label): loaded in \(String(format: "%.1f", loadMs)) ms")

                // 2 回実行して初回（コンパイル/ウォームアップ込み）と 2 回目を分離
                for runIndex in 1...2 {
                    let (waveform, predictMs) = try await runner.run(
                        mel: mel, nMels: Self.nMels, frames: Self.frames
                    )
                    let stats = WaveformStatistics(waveform: waveform)
                    let psnr = WaveformStatistics.psnr(reference: reference, test: waveform)

                    csvLines.append(csvRow(
                        unit: unit, runIndex: runIndex, status: "ok",
                        loadMs: runIndex == 1 ? loadMs : 0,
                        predictMs: predictMs, psnr: psnr, stats: stats
                    ))

                    print(String(
                        format: "[CoreAIVocoderTests] %@ run%d: %.1f ms, rms=%.4e, over0.99=%.4f, nan=%d, inf=%d, psnr=%.1f dB",
                        unit.label, runIndex, predictMs, stats.rms,
                        stats.ratioOver099, stats.nanCount, stats.infCount, psnr
                    ))

                    // 2 回目の波形を npy 保存（オフライン MCD / 追加 PSNR 用）
                    if runIndex == 2 {
                        let wavURL = outDir.appendingPathComponent("coreai_hifigan_\(unit.label).npy")
                        try NpyWriter.writeFloat32(waveform, rows: 1, cols: waveform.count, to: wavURL)
                    }

                    // 検証は「実機で通る」ことが主目的。clipping/破綻は記録対象であって失敗ではない。
                    #expect(waveform.count == Self.frames * Self.hopSize)
                }
            } catch {
                csvLines.append(csvRow(
                    unit: unit, runIndex: 0, status: "error: \(error)",
                    loadMs: 0, predictMs: 0, psnr: .nan,
                    stats: WaveformStatistics(waveform: [])
                ))
                print("[CoreAIVocoderTests] \(unit.label): ERROR \(error)")
            }
        }

        // --- CSV 書き出し ---
        let csvURL = outDir.appendingPathComponent("hifigan_compute_units.csv")
        try csvLines.joined(separator: "\n").appending("\n")
            .write(to: csvURL, atomically: true, encoding: .utf8)
        print("[CoreAIVocoderTests] CSV: \(csvURL.path)")
    }

    // MARK: - CSV

    private var csvHeader: String {
        // export_precision は .aimodel の export 精度（常に float32）。
        // runtime_precision は実行時の実効精度（NE は fp16）。両者を混同しないため別列。
        "model,export_precision,runtime_precision,compute_unit,coreml_equiv,run_index,status,load_ms,predict_ms,psnr_db,"
            + WaveformStatistics.csvHeader
    }

    private func csvRow(
        unit: CoreAIComputeUnit, runIndex: Int, status: String,
        loadMs: Double, predictMs: Double, psnr: Double, stats: WaveformStatistics
    ) -> String {
        // status はエラー文をそのまま入れることがあり、カンマ/改行が混じると列がずれる。
        // 一番見たい失敗行の CSV が壊れないよう、区切り文字を無害化してから書く。
        let safeStatus = status
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: ",", with: ";")
        return [
            "HiFiGAN_Generator", "float32", unit.runtimePrecisionNote,
            unit.label, unit.coreMLEquivalentNote,
            String(runIndex), safeStatus,
            String(format: "%.3f", loadMs),
            String(format: "%.3f", predictMs),
            String(format: "%.3f", psnr),
            stats.csvRow,
        ].joined(separator: ",")
    }

    // MARK: - リソース解決

    /// バンドルから name.ext を探す。直下で見つからなければ再帰探索する
    /// （同期グループのサブフォルダ flatten 対策）。
    private func locateResource(name: String, ext: String) throws -> URL {
        let bundles = [Bundle.main, Bundle(for: BundleToken.self)]
        for bundle in bundles {
            if let url = bundle.url(forResource: name, withExtension: ext) {
                return url
            }
            if let root = bundle.resourceURL,
               let found = Self.findRecursively(in: root, fileName: "\(name).\(ext)") {
                return found
            }
        }
        throw ResourceError.notFound("\(name).\(ext)")
    }

    private static func findRecursively(in root: URL, fileName: String) -> URL? {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return nil
        }
        for case let url as URL in enumerator where url.lastPathComponent == fileName {
            return url
        }
        return nil
    }

    private func makeOutputDir() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result/coreai", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    enum ResourceError: Error {
        case notFound(String)
    }
}

/// `Bundle(for:)` 用のトークン。
private final class BundleToken {}
#endif
