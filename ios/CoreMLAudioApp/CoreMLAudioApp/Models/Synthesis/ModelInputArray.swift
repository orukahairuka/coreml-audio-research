import Accelerate
import CoreML

/// Core ML モデルへ渡す入力配列の確保と検証
///
/// `MLMultiArray(shape:dataType:)` は確保した時点で内容を初期化しない（値の設定は利用側の責任）。
/// 固定長モデル（例: fixed262）では配列の先頭にしか値を書き込まないため、
/// 書き込まない部分に何が入っているかは保証されない。
/// ここで確保直後に全要素を 0 で埋め、推論の直前に意図した値になっているかを検証する。
///
/// 書き込んだ部分に NaN/Inf が含まれていても例外にはしない（数を返すだけ）。
/// Decoder の自己回帰出力が不安定な条件では NaN/Inf が入力に戻ってくるが、
/// それは止めずに後段まで流し、HiFi-GAN 出力の NaN/Inf として記録したい観測対象であるため。
enum ModelInputArray {

    enum ValidationError: LocalizedError {
        /// 配列が行優先で連続していない（範囲の計算が成り立たない）
        case notContiguous(name: String)
        /// 0 のはずの部分に 0 以外の値がある
        case nonZeroPadding(name: String)

        var errorDescription: String? {
            switch self {
            case .notContiguous(let name): return "\(name): 入力配列が連続したメモリ配置ではありません。"
            case .nonZeroPadding(let name): return "\(name): 値を書き込まない部分が 0 になっていません。"
            }
        }
    }

    /// 全要素を 0 で埋めた配列を確保する
    static func zeros(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: dataType)
        array.withUnsafeMutableBytes { buffer, _ in
            _ = buffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
        return array
    }

    /// float32 の入力配列を検証する
    /// - Parameters:
    ///   - array: 検証する配列（行優先で連続していること）
    ///   - paddingRanges: 値を書き込まず 0 のままのはずの要素の範囲（線形インデックス）
    ///   - name: エラー表示用の配列名
    /// - Returns: 配列全体に含まれる NaN/Inf の個数（診断用。0 以外でも例外にしない）
    /// - Throws: 連続でない、または `paddingRanges` に 0 以外（NaN/Inf を含む）がある場合
    @discardableResult
    static func validateFloat32(_ array: MLMultiArray, paddingRanges: [Range<Int>], name: String) throws -> Int {
        precondition(array.dataType == .float32, "validateFloat32 は float32 配列のみ対象")
        guard isRowMajorContiguous(array) else { throw ValidationError.notContiguous(name: name) }

        let count = array.count
        return try array.withUnsafeBytes { raw in
            try raw.withMemoryRebound(to: Float.self) { values in
                guard let base = values.baseAddress, values.count >= count else {
                    throw ValidationError.notContiguous(name: name)
                }

                // 0 のはずの部分は二乗和が 0（NaN が混ざれば NaN になり、これも弾く）
                for range in paddingRanges where !range.isEmpty {
                    precondition(range.lowerBound >= 0 && range.upperBound <= count, "paddingRanges が配列の範囲外")
                    var sumSq: Float = 0
                    vDSP_svesq(base + range.lowerBound, 1, &sumSq, vDSP_Length(range.count))
                    guard sumSq == 0 else { throw ValidationError.nonZeroPadding(name: name) }
                }

                // 全要素の和が有限なら NaN/Inf は含まない（値は正規化済みメルなので和の桁あふれは起きない）。
                // 有限でないときだけ 1 要素ずつ数える
                var sum: Float = 0
                vDSP_sve(base, 1, &sum, vDSP_Length(count))
                if sum.isFinite { return 0 }
                var nonFiniteCount = 0
                for i in 0..<count where !base[i].isFinite { nonFiniteCount += 1 }
                return nonFiniteCount
            }
        }
    }

    /// strides が行優先の連続配置と一致するか
    private static func isRowMajorContiguous(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        var expected = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            if shape[axis] > 1 && strides[axis] != expected { return false }
            expected *= shape[axis]
        }
        return true
    }
}
