import Testing
@testable import CoreMLAudioApp
import CoreML

/// モデル入力配列の 0 初期化と検証（`ModelInputArray`）のテスト。
/// 固定長モデルの入力で、値を書き込まない部分が 0 になることを保証するためのもの。
struct ModelInputArrayTests {

    // MARK: - zeros

    @Test func zerosFillsEveryFloat32Element() throws {
        let array = try ModelInputArray.zeros(shape: [1, 262, 256], dataType: .float32)
        for i in 0..<array.count {
            #expect(array[i].floatValue == 0)
        }
    }

    @Test func zerosFillsEveryInt32Element() throws {
        let array = try ModelInputArray.zeros(shape: [1, 262], dataType: .int32)
        for i in 0..<array.count {
            #expect(array[i].int32Value == 0)
        }
    }

    // MARK: - copyFloat32 / Decoder 入力の書き込み

    /// 配列全体の生バイト列
    private func bytes(of array: MLMultiArray) -> [UInt8] {
        array.withUnsafeBytes { raw in Array(raw.prefix(array.count * MemoryLayout<Float>.size)) }
    }

    /// 生成済みフレームの値（通常の値に加えて NaN・Inf・-0.0・非正規化数も混ぜる）
    private func sampleFrames(count: Int) -> [Float] {
        var values = (0..<count).map { i in Float(sin(Double(i) * 0.37)) * 4 - 1.5 }
        let specials: [Float] = [.nan, .infinity, -.infinity, -0.0, .leastNonzeroMagnitude, .greatestFiniteMagnitude]
        for (k, v) in specials.enumerated() where k * 97 < count {
            values[k * 97] = v
        }
        return values
    }

    /// 従来の 1 要素ずつの書き込みと、まとめてコピーする書き込みで、入力配列のバイト列が一致する
    @Test(arguments: [0, 1, 137, 261, 262])
    func bulkWriteMatchesElementWrite(frames: Int) throws {
        let nMels = 256
        let data = sampleFrames(count: 262 * nMels)
        let count = frames * nMels
        let element = try ModelInputArray.zeros(shape: [1, 262, nMels], dataType: .float32)
        let bulk = try ModelInputArray.zeros(shape: [1, 262, nMels], dataType: .float32)
        try DecoderRunner.writeInput(data, count: count, into: element, method: .element)
        try DecoderRunner.writeInput(data, count: count, into: bulk, method: .bulk)
        #expect(bytes(of: element) == bytes(of: bulk))
        // 書き込まない部分は 0 のまま（固定長モデルの未生成部分）
        try ModelInputArray.validateFloat32(bulk, paddingRanges: [count..<(262 * nMels)], name: "test")
    }

    @Test func copyRejectsTooManyElements() throws {
        let array = try ModelInputArray.zeros(shape: [1, 2, 3], dataType: .float32)
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.copyFloat32([Float](repeating: 1, count: 7), count: 7, into: array, name: "test")
        }
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.copyFloat32([1, 2], count: 3, into: array, name: "test")
        }
    }

    @Test func copyRejectsNonFloat32() throws {
        let array = try ModelInputArray.zeros(shape: [1, 4], dataType: .int32)
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.copyFloat32([1, 2], count: 2, into: array, name: "test")
        }
    }

    // MARK: - validateFloat32

    @Test func validatePassesWhenPaddingIsZero() throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        for i in 0..<6 { array[i] = NSNumber(value: Float(i) + 0.5) }  // 先頭 2 フレームだけ書き込む
        try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
    }

    @Test func validateRejectsNonZeroPadding() throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        array[10] = 1e-6
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
        }
    }

    /// 書き込み済みの部分の NaN/Inf は不安定な条件の観測対象なので、例外にせず個数を返す
    @Test(arguments: [Float.nan, Float.infinity, -Float.infinity])
    func validateCountsNonFiniteContentWithoutThrowing(value: Float) throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        array[2] = NSNumber(value: value)  // 書き込み済みの部分に混入
        array[5] = NSNumber(value: value)
        let nonFiniteCount = try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
        #expect(nonFiniteCount == 2)
    }

    @Test func validateReturnsZeroForFiniteContent() throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        for i in 0..<6 { array[i] = NSNumber(value: Float(i) - 2.5) }
        let nonFiniteCount = try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
        #expect(nonFiniteCount == 0)
    }

    @Test func validateRejectsNaNInPadding() throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        array[8] = NSNumber(value: Float.nan)
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
        }
    }

    /// HiFi-GAN の入力と同じ [1, nMels, T] 配置で、帯域ごとの末尾を padding として検証できること
    @Test func validateChecksEachMelRowTail() throws {
        let nMels = 3, inputT = 5, contentT = 2
        let array = try ModelInputArray.zeros(shape: [1, nMels, inputT], dataType: .float32)
        for m in 0..<nMels {
            for t in 0..<contentT {
                array[[0, m as NSNumber, t as NSNumber]] = 1
            }
        }
        let ranges = (0..<nMels).map { m in (m * inputT + contentT)..<((m + 1) * inputT) }
        try ModelInputArray.validateFloat32(array, paddingRanges: ranges, name: "test")

        array[[0, 2, 4]] = 0.25
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.validateFloat32(array, paddingRanges: ranges, name: "test")
        }
    }
}
