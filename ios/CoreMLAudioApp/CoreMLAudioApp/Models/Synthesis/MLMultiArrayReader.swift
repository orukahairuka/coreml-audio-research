import CoreML

/// `MLMultiArray` の値を、添字で 1 要素ずつ読むのと同じ結果になるようメモリから直接読む
///
/// Float16 モデルの出力は `float16` 型で、行の間に隙間がある配置（strides）になることがあるので、
/// 型と strides を見て読む。対応していない型は添字で読む。
enum MLMultiArrayReader {

    /// [1, 行, 列] の配列の `rows` の行を、行優先で `destination` に書き込む
    static func readRows(
        _ array: MLMultiArray, rows: Range<Int>, columns: Int, into destination: UnsafeMutablePointer<Float>
    ) {
        precondition(array.shape.count == 3, "[1, 行, 列] の配列のみ対象")
        precondition(rows.lowerBound >= 0 && rows.upperBound <= array.shape[1].intValue, "行が配列の範囲外")
        precondition(columns <= array.shape[2].intValue, "列が配列の範囲外")
        let rowStride = array.strides[1].intValue
        let columnStride = array.strides[2].intValue

        func copy<Scalar>(_ type: Scalar.Type, convert: (Scalar) -> Float) {
            array.withUnsafeBytes({ raw in
                let values = raw.bindMemory(to: Scalar.self)
                var output = destination
                for row in rows {
                    var index = row * rowStride
                    for _ in 0..<columns {
                        output.pointee = convert(values[index])
                        output += 1
                        index += columnStride
                    }
                }
            })
        }

        switch array.dataType {
        case .float32: copy(Float.self, convert: { $0 })
        case .float16: copy(Float16.self, convert: { Float($0) })
        case .double: copy(Double.self, convert: { Float($0) })
        default:
            var output = destination
            for row in rows {
                for column in 0..<columns {
                    output.pointee = array[[0, row as NSNumber, column as NSNumber]].floatValue
                    output += 1
                }
            }
        }
    }

    /// [1, 行, 列] の配列の `rows` の行を、行優先の配列にして返す
    static func rows(_ array: MLMultiArray, rows: Range<Int>, columns: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: rows.count * columns, initializingWith: { buffer, count in
            readRows(array, rows: rows, columns: columns, into: buffer.baseAddress!)
            count = rows.count * columns
        })
    }
}

extension MLModel {
    /// 同期版の `prediction(from:)` を呼ぶ（async の文脈からは async 版が選ばれるので分けている）
    func predictSynchronously(from input: MLFeatureProvider) throws -> MLFeatureProvider {
        try prediction(from: input)
    }
}
