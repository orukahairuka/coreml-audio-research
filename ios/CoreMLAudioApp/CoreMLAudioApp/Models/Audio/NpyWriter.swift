import Foundation

/// NumPy の .npy フォーマット (Version 1.0) を書き出す
///
/// フォーマット:
///   - Magic: `\x93NUMPY`
///   - Version: `\x01\x00`
///   - Header length: uint16 little-endian
///   - Header: Python dict 形式の文字列、末尾 `\n`、全体長が 64 バイト境界に揃うようスペースでパディング
///   - Data: dtype に従った生バイト列 (little-endian)
enum NpyWriter {

    enum NpyError: Error {
        case invalidShape
    }

    /// 2D float32 配列を .npy として書き出す
    ///
    /// - Parameters:
    ///   - data: row-major 配列 (長さ = rows × cols)
    ///   - rows: 行数
    ///   - cols: 列数
    ///   - url: 出力先
    static func writeFloat32(_ data: [Float], rows: Int, cols: Int, to url: URL) throws {
        guard rows > 0, cols > 0, data.count == rows * cols else {
            throw NpyError.invalidShape
        }

        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  // \x93NUMPY
        let version: [UInt8] = [0x01, 0x00]

        // (rows, cols) の Python タプル表記。1次元なら (N,) になるが今回は常に 2D
        let shapeStr = "(\(rows), \(cols))"
        var headerStr = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeStr), }"

        // magic(6) + version(2) + headerLen(2) + header + '\n' が 64 バイト境界に揃うようパディング
        let prefixLen = 6 + 2 + 2
        let totalUnpadded = prefixLen + headerStr.count + 1  // +1 は末尾 '\n'
        let padded = ((totalUnpadded + 63) / 64) * 64
        let paddingCount = padded - totalUnpadded
        headerStr += String(repeating: " ", count: paddingCount) + "\n"

        let headerBytes = Array(headerStr.utf8)
        let headerLen = UInt16(headerBytes.count)
        let headerLenBytes: [UInt8] = [
            UInt8(headerLen & 0xFF),
            UInt8((headerLen >> 8) & 0xFF),
        ]

        var out = Data()
        out.append(contentsOf: magic)
        out.append(contentsOf: version)
        out.append(contentsOf: headerLenBytes)
        out.append(contentsOf: headerBytes)

        let byteCount = data.count * MemoryLayout<Float>.size
        data.withUnsafeBufferPointer { buf in
            out.append(Data(bytes: buf.baseAddress!, count: byteCount))
        }

        try out.write(to: url)
    }
}
