import CoreML
import Foundation
import Testing
@testable import CoreMLAudioApp

/// 高速化した読み書きが、添字で 1 要素ずつ扱う従来の方法と同じ結果になることを確かめる
struct SynthesisOptimizationTests {

    /// 行の間に隙間のある [1, rows, columns] の配列を作る
    private func makeArray<Scalar>(
        rows: Int, columns: Int, rowGap: Int, dataType: MLMultiArrayDataType, value: (Int) -> Scalar
    ) throws -> MLMultiArray {
        let rowStride = columns + rowGap
        let pointer = UnsafeMutablePointer<Scalar>.allocate(capacity: rows * rowStride)
        for i in 0..<(rows * rowStride) { pointer.initialize(to: value(0)) ; pointer[i] = value(i) }
        return try MLMultiArray(
            dataPointer: pointer, shape: [1, rows as NSNumber, columns as NSNumber], dataType: dataType,
            strides: [(rows * rowStride) as NSNumber, rowStride as NSNumber, 1],
            deallocator: { $0.deallocate() }
        )
    }

    private func elementRows(_ array: MLMultiArray, rows: Range<Int>, columns: Int) -> [Float] {
        rows.flatMap({ row in
            (0..<columns).map({ array[[0, row as NSNumber, $0 as NSNumber]].floatValue })
        })
    }

    @Test(arguments: [0, 3])
    func readRowsMatchesElementAccessForFloat16(rowGap: Int) throws {
        let array: MLMultiArray = try makeArray(
            rows: 7, columns: 5, rowGap: rowGap, dataType: .float16, value: { Float16($0) * 0.125 - 3.5 }
        )
        #expect(MLMultiArrayReader.rows(array, rows: 2..<6, columns: 5) == elementRows(array, rows: 2..<6, columns: 5))
    }

    @Test(arguments: [0, 3])
    func readRowsMatchesElementAccessForFloat32(rowGap: Int) throws {
        let array: MLMultiArray = try makeArray(
            rows: 7, columns: 5, rowGap: rowGap, dataType: .float32, value: { Float($0) * 0.1 - 0.7 }
        )
        #expect(MLMultiArrayReader.rows(array, rows: 0..<7, columns: 5) == elementRows(array, rows: 0..<7, columns: 5))
    }

    @Test func readRowsReadsSingleRowWaveform() throws {
        let array: MLMultiArray = try makeArray(
            rows: 1, columns: 16, rowGap: 0, dataType: .float16, value: { Float16($0) / 8 }
        )
        let expected = (0..<16).map({ array[$0].floatValue })
        #expect(MLMultiArrayReader.rows(array, rows: 0..<1, columns: 16) == expected)
    }

    @Test func parseAcceptsKnownNamesAndRejectsUnknown() {
        #expect(SynthesisOptimizations.parse("none") == [])
        #expect(SynthesisOptimizations.parse("vocoderTranspose,reuseDecoderPos") == [.vocoderTranspose, .reuseDecoderPos])
        #expect(SynthesisOptimizations.parse("unknown") == nil)
        #expect(SynthesisOptimizations([.vocoderTranspose, .syncPrediction]).label == "vocoderTranspose+syncPrediction")
        #expect(SynthesisOptimizations([]).label == "none")
    }
}
