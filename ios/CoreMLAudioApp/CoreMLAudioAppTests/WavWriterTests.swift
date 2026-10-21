import Testing
@testable import CoreMLAudioApp
import AVFoundation
import Foundation

/// `WavWriter`（AVAudioFile による Float32 → Int16 の保存）のテスト。
///
/// 範囲 [-1, 1] を超えた値が保存時にどう扱われるか（クリップかラップアラウンドか）を、
/// 書き出した WAV のデータ部を Int16 のまま読み直して確かめる。
struct WavWriterTests {

    /// 書き出した WAV の "data" チャンクを Int16（little-endian）の配列として読む
    private func readInt16Samples(url: URL) throws -> [Int16] {
        let data = try Data(contentsOf: url)
        let bytes = [UInt8](data)
        var offset = 12  // "RIFF" + size + "WAVE"
        while offset + 8 <= bytes.count {
            let id = String(bytes: bytes[offset..<(offset + 4)], encoding: .ascii) ?? ""
            let size = Int(bytes[offset + 4]) | Int(bytes[offset + 5]) << 8
                | Int(bytes[offset + 6]) << 16 | Int(bytes[offset + 7]) << 24
            let body = offset + 8
            if id == "data" {
                var samples: [Int16] = []
                var i = body
                while i + 1 < body + size && i + 1 < bytes.count {
                    samples.append(Int16(bitPattern: UInt16(bytes[i]) | UInt16(bytes[i + 1]) << 8))
                    i += 2
                }
                return samples
            }
            offset = body + size + (size % 2)
        }
        Issue.record("data チャンクが見つからない")
        return []
    }

    private func writeAndRead(_ waveform: [Float]) throws -> [Int16] {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("wavwriter_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try WavWriter.writeInt16(waveform: waveform, sampleRate: 22050, to: url)
        return try readInt16Samples(url: url)
    }

    @Test func inRangeValuesKeepSign() throws {
        let samples = try writeAndRead([0, 0.5, -0.5, 0.99, -0.99])
        #expect(samples.count == 5)
        #expect(samples[0] == 0)
        #expect(samples[1] > 16000 && samples[1] < 16500)
        #expect(samples[2] < -16000 && samples[2] > -16500)
    }

    /// 範囲を超えた値はクリップされ、符号が反転するラップアラウンドは起きない
    @Test func outOfRangeValuesAreClippedNotWrapped() throws {
        let input: [Float] = [1.0, -1.0, 1.0001, -1.0001, 1.5, -1.5, 3.0, -3.0, 100, -100]
        let samples = try writeAndRead(input)
        print("[WavWriterTests] input=\(input) int16=\(samples)")
        #expect(samples.count == input.count)
        for (x, s) in zip(input, samples) {
            if x > 0 {
                #expect(s >= 32766, "x=\(x) → \(s)")
            } else {
                #expect(s <= -32767, "x=\(x) → \(s)")
            }
        }
        #expect(samples[4] == Int16.max)
        #expect(samples[5] == Int16.min)
    }
}
