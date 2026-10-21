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

    @Test(arguments: [Float.nan, Float.infinity, -Float.infinity])
    func validateRejectsNonFiniteValues(value: Float) throws {
        let array = try ModelInputArray.zeros(shape: [1, 4, 3], dataType: .float32)
        array[2] = NSNumber(value: value)  // 書き込み済みの部分に混入
        #expect(throws: ModelInputArray.ValidationError.self) {
            try ModelInputArray.validateFloat32(array, paddingRanges: [6..<12], name: "test")
        }
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
