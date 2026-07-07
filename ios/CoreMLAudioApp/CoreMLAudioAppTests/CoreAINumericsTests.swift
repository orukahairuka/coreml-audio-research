import Testing
import Foundation
@testable import CoreMLAudioApp

/// Core AI 経路の**実機非依存**な純粋関数（PSNR 計算・npy 読み込み）の回帰テスト。
///
/// 実機テスト `CoreAIVocoderTests` は iOS 27 + `.aimodel` 同梱が必要で CI では回せない。
/// ここは Core AI に依存しないロジックだけを対象にし、破綻検出の要である
/// 「non-finite が混じったら PSNR を nan にする」挙動と、npy パースの前提を固定する。
struct CoreAINumericsTests {

    // MARK: - WaveformStatistics.psnr

    @Test
    func psnrIdenticalIsInfinity() {
        let signal: [Float] = [0.1, -0.2, 0.3, -0.4]
        #expect(WaveformStatistics.psnr(reference: signal, test: signal) == .infinity)
    }

    @Test
    func psnrKnownValue() {
        // peak=1, rmse=0.1 → 20*log10(1/0.1)=20 dB
        let reference: [Float] = [1, -1]
        let test: [Float] = [0.9, -0.9]
        let psnr = WaveformStatistics.psnr(reference: reference, test: test)
        #expect(abs(psnr - 20.0) < 1e-4)
    }

    @Test
    func psnrTruncatesToShorterLength() {
        // 長さが違う場合は短い方に合わせる（前の 20 dB ケースと一致するはず）。
        let reference: [Float] = [1, -1, 999]
        let test: [Float] = [0.9, -0.9]
        let psnr = WaveformStatistics.psnr(reference: reference, test: test)
        #expect(abs(psnr - 20.0) < 1e-4)
    }

    @Test
    func psnrNonFiniteInReferenceIsNaN() {
        // 参照に inf が 1 つでも混じったら、有限サンプルで妥当そうな値を出さず nan を返す。
        let reference: [Float] = [1, .infinity, -1]
        let test: [Float] = [0.9, 0.0, -0.9]
        #expect(WaveformStatistics.psnr(reference: reference, test: test).isNaN)
    }

    @Test
    func psnrNonFiniteInTestIsNaN() {
        let reference: [Float] = [1, 0.0, -1]
        let test: [Float] = [0.9, .nan, -0.9]
        #expect(WaveformStatistics.psnr(reference: reference, test: test).isNaN)
    }

    @Test
    func psnrEmptyIsNaN() {
        #expect(WaveformStatistics.psnr(reference: [], test: []).isNaN)
    }

    // MARK: - WaveformStatistics（統計量）

    @Test
    func statsSeparatesNonFiniteFromFinite() {
        let stats = WaveformStatistics(waveform: [1.0, -1.0, .nan, .infinity, 0.5])
        #expect(stats.count == 5)
        #expect(stats.finiteCount == 3)
        #expect(stats.nanCount == 1)
        #expect(stats.infCount == 1)
        // clipping 指標は全要素数を分母にする（documented）
        #expect(abs(stats.ratioOver099 - 1.0 / 5.0) < 1e-9)
        #expect(abs(stats.ratioUnderMinus099 - 1.0 / 5.0) < 1e-9)
    }

    @Test
    func statsAllNonFiniteYieldsNaNAggregates() {
        let stats = WaveformStatistics(waveform: [.nan, .infinity])
        #expect(stats.finiteCount == 0)
        #expect(stats.rms.isNaN)
        #expect(stats.mean.isNaN)
    }

    // MARK: - MinimalNpyReader

    @Test
    func npyRoundTripThroughWriter() throws {
        // NpyWriter が書いた v1.0 <f4 を MinimalNpyReader が同じ値列で読み戻せる。
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("minimalnpy_roundtrip_\(UUID().uuidString).npy")
        try NpyWriter.writeFloat32(values, rows: 2, cols: 3, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let read = try MinimalNpyReader.readFloat32(contentsOf: url)
        #expect(read == values)
    }

    @Test
    func npyRejectsBadMagic() {
        let bogus = Data(repeating: 0, count: 128)
        #expect(throws: MinimalNpyReader.NpyError.self) {
            _ = try MinimalNpyReader.readFloat32(data: bogus)
        }
    }

    @Test
    func npyRejectsFortranOrder() {
        // fortran_order: True を弾く（黙って転置値を読まない）。
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }"
        let data = makeNpy(headerBody: header, floats: [1.0])
        #expect(throws: MinimalNpyReader.NpyError.self) {
            _ = try MinimalNpyReader.readFloat32(data: data)
        }
    }

    @Test
    func npyRejectsVersion2() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }"
        let data = makeNpy(headerBody: header, floats: [1.0], major: 2)
        #expect(throws: MinimalNpyReader.NpyError.self) {
            _ = try MinimalNpyReader.readFloat32(data: data)
        }
    }

    @Test
    func npyRejectsNonFloat32Dtype() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"
        let data = makeNpy(headerBody: header, floats: [1.0])
        #expect(throws: MinimalNpyReader.NpyError.self) {
            _ = try MinimalNpyReader.readFloat32(data: data)
        }
    }

    // MARK: - Helpers

    /// テスト用に v1.0 npy バイト列を組み立てる（header は 64 バイト境界に padding）。
    private func makeNpy(headerBody: String, floats: [Float], major: UInt8 = 1) -> Data {
        var header = headerBody
        let prefixLen = 6 + 2 + 2
        let totalUnpadded = prefixLen + header.count + 1
        let padded = ((totalUnpadded + 63) / 64) * 64
        header += String(repeating: " ", count: padded - totalUnpadded) + "\n"

        let headerBytes = Array(header.utf8)
        let headerLen = UInt16(headerBytes.count)

        var out = Data()
        out.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])  // \x93NUMPY
        out.append(contentsOf: [major, 0x00])
        out.append(contentsOf: [UInt8(headerLen & 0xFF), UInt8((headerLen >> 8) & 0xFF)])
        out.append(contentsOf: headerBytes)
        floats.withUnsafeBufferPointer { buf in
            out.append(Data(bytes: buf.baseAddress!, count: buf.count * MemoryLayout<Float>.size))
        }
        return out
    }
}
