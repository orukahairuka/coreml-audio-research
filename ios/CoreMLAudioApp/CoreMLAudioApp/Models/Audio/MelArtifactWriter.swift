import Foundation
import ImageIO
import UniformTypeIdentifiers

/// メルスペクトログラムの値 (.npy) と画像 (.png) をペアで書き出す
///
/// 保存先は `Documents/Result/mel/`。ファイル名は呼び出し側が baseName で指定する。
/// 固定名での上書き保存（バッチ取得用）を想定。
enum MelArtifactWriter {

    enum WriteError: LocalizedError {
        case directoryCreationFailed(URL, underlying: Error)
        case pngEncodingFailed

        var errorDescription: String? {
            switch self {
            case .directoryCreationFailed(let url, let underlying):
                return "保存先の作成に失敗: \(url.path) (\(underlying.localizedDescription))"
            case .pngEncodingFailed:
                return "PNG エンコードに失敗"
            }
        }
    }

    /// 出力ディレクトリ (`Documents/Result/mel/`)。無ければ作る
    static func resultDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("mel", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            } catch {
                throw WriteError.directoryCreationFailed(dir, underlying: error)
            }
        }
        return dir
    }

    /// メル配列から `.npy` と `.png` を書き出す
    ///
    /// - Parameters:
    ///   - melData: [frameCount × nMels] 配列、dB スケール (-80〜0)
    ///   - frameCount: 時間フレーム数
    ///   - nMels: メル次元数
    ///   - baseName: 拡張子抜きのファイル名 (例: `"output_mel_Float16_cpuAndGPU"`)
    static func write(
        melData: [Float],
        frameCount: Int,
        nMels: Int,
        baseName: String
    ) throws {
        let dir = try resultDirectory()
        let npyURL = dir.appendingPathComponent("\(baseName).npy")
        let pngURL = dir.appendingPathComponent("\(baseName).png")

        try NpyWriter.writeFloat32(melData, rows: frameCount, cols: nMels, to: npyURL)

        guard let image = MelSpectrogramRenderer.makeImage(melData: melData, frameCount: frameCount, nMels: nMels) else {
            throw WriteError.pngEncodingFailed
        }
        try writePng(image, to: pngURL)
    }

    private static func writePng(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw WriteError.pngEncodingFailed
        }
        CGImageDestinationAddImage(dest, image, nil)
        if !CGImageDestinationFinalize(dest) {
            throw WriteError.pngEncodingFailed
        }
    }
}
