import Foundation

/// 研究検証用の最小 .npy リーダー（float32 / `<f4` のみ対応）。
///
/// 書き出しは既存 `NpyWriter`。読み込みはこれまで無かったので、固定入力
/// （PyTorch 参照 mel / 出力）を実機テストに渡すために最小実装する。
///
/// 対応フォーマット: NumPy v1.0、`descr` が `<f4`（little-endian float32）。
/// shape は読み飛ばし、データ部の全 float32 を 1 次元 [Float] で返す。
enum MinimalNpyReader {

    enum NpyError: Error {
        case badMagic
        case unsupportedVersion(major: UInt8, minor: UInt8)
        case unsupportedDtype(String)
        case fortranOrder
        case truncated
    }

    /// float32 の .npy を [Float]（row-major フラット）として読む。
    static func readFloat32(contentsOf url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return try readFloat32(data: data)
    }

    static func readFloat32(data: Data) throws -> [Float] {
        // magic(6) + version(2) + headerLen(2, uint16 LE)
        guard data.count >= 10 else { throw NpyError.truncated }
        // Data のスライスは元の index を保持するので、先頭からの絶対 offset で参照する。
        let base = data.startIndex
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  // \x93NUMPY
        guard Array(data[base ..< base + 6]) == magic else { throw NpyError.badMagic }

        // version。v1.0 のみ対応（headerLen は uint16=2byte）。v2.0+ は headerLen が uint32 で
        // ここの読み取りがずれるため、黙って誤読せず弾く。
        let major = data[base + 6]
        let minor = data[base + 7]
        guard major == 1 else { throw NpyError.unsupportedVersion(major: major, minor: minor) }

        let headerLen = Int(data[base + 8]) | (Int(data[base + 9]) << 8)
        let headerStart = 10
        let dataStart = headerStart + headerLen
        guard data.count >= dataStart else { throw NpyError.truncated }

        let header = String(decoding: data[base + headerStart ..< base + dataStart], as: UTF8.self)
        guard header.contains("'<f4'") || header.contains("\"<f4\"") else {
            throw NpyError.unsupportedDtype(header)
        }
        // fortran_order: True だと列優先で、生バイトを row-major として読むと転置した値になる。
        // np.save は C-contiguous 配列なら False を書くが、取り違え検出のため明示的に弾く。
        guard !header.contains("'fortran_order': True") else { throw NpyError.fortranOrder }

        // npy `<f4` は little-endian。arm64 も little-endian なので生バイトをそのままコピーできる。
        let stride = MemoryLayout<Float32>.size
        let rawCount = (data.count - dataStart) / stride
        var result = [Float](repeating: 0, count: rawCount)
        result.withUnsafeMutableBytes { dst in
            _ = data.withUnsafeBytes { src in
                memcpy(dst.baseAddress!, src.baseAddress!.advanced(by: dataStart), rawCount * stride)
            }
        }
        return result
    }
}
