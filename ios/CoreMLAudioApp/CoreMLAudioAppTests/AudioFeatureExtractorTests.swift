import Testing
@testable import CoreMLAudioApp
import Foundation

struct AudioFeatureExtractorTests {

    // MARK: - Hz ↔ Mel ラウンドトリップ

    @Test(arguments: [0.0, 440.0, 1000.0, 8000.0, 11025.0] as [Float])
    func hzToMelRoundTrip(hz: Float) {
        let mel = AudioFeatureExtractor.hzToMel(hz)
        let recovered = AudioFeatureExtractor.melToHz(mel)
        #expect(abs(recovered - hz) < 0.01, "Hz→Mel→Hz が元に戻らない: \(hz) → \(recovered)")
    }

    @Test func hzToMelKnownValues() {
        // 0 Hz → 0 mel
        #expect(AudioFeatureExtractor.hzToMel(0) == 0)
        // 1000 Hz ≈ 1000 mel (O'Shaughnessy 式の特性)
        let mel1000 = AudioFeatureExtractor.hzToMel(1000)
        #expect(abs(mel1000 - 999.985) < 0.1)
    }

    // MARK: - プリエンファシス ↔ デエンファシス ラウンドトリップ

    @Test func preemphasisDeemphasisRoundTrip() {
        let original: [Float] = [0.5, -0.3, 0.8, -0.1, 0.6, 0.2, -0.7, 0.4]
        let emphasized = AudioFeatureExtractor.applyPreemphasis(original)
        let recovered = AudioFeatureExtractor.applyDeemphasis(emphasized)

        for i in 0..<original.count {
            #expect(abs(recovered[i] - original[i]) < 1e-5,
                    "index \(i): \(original[i]) → \(recovered[i])")
        }
    }

    @Test func preemphasisDeemphasisSingleElement() {
        let single: [Float] = [0.5]
        #expect(AudioFeatureExtractor.applyPreemphasis(single) == single)
        #expect(AudioFeatureExtractor.applyDeemphasis(single) == single)
    }

    @Test func preemphasisDeemphasisEmpty() {
        let empty: [Float] = []
        #expect(AudioFeatureExtractor.applyPreemphasis(empty) == empty)
        #expect(AudioFeatureExtractor.applyDeemphasis(empty) == empty)
    }

    // MARK: - normalizeToDB ↔ denormalizeToDisplayDb ラウンドトリップ

    @Test func normalizeDeormalizeRoundTrip() {
        // normalizeToDB は (mel, frameCount) タプルを受けてクリッピングするので、
        // クリッピングされない範囲の値でラウンドトリップを検証する
        // normalized = (20*log10(value) - refDB + maxDB) / maxDB
        // クリッピングなし条件: 1e-8 < normalized < 1.0
        // → refDB - maxDB < db < 0  → -80 < db < 0
        // → 10^(-80/20) < value < 10^(0/20) → 1e-4 < value < 1.0
        let values: [Float] = [0.001, 0.01, 0.1, 0.5, 1.0]
        let normalized = AudioFeatureExtractor.normalizeToDB((mel: values, frameCount: 1))

        let denormalized = AudioFeatureExtractor.denormalizeToDisplayDb(normalized)

        // denormalize は dB スケールに戻す（power_to_db 相当の -80〜0 レンジ）
        // normalizeToDB: db = 20*log10(value), normalized = (db - refDB + maxDB) / maxDB
        // denormalize:   db_recovered = normalized * maxDB + refDB - maxDB
        // → db_recovered = db であることを確認
        for i in 0..<values.count {
            let expectedDb = 20.0 * log10(max(1e-5, values[i]))
            #expect(abs(denormalized[i] - expectedDb) < 0.01,
                    "index \(i): expected dB=\(expectedDb), got \(denormalized[i])")
        }
    }

    @Test func normalizeToDBClipping() {
        // 非常に小さい値 → 1e-8 にクリップされる
        let tiny: [Float] = [1e-10]
        let normalized = AudioFeatureExtractor.normalizeToDB((mel: tiny, frameCount: 1))
        #expect(normalized[0] >= 1e-8)

        // 非常に大きい値 → 1.0 にクリップされる
        let large: [Float] = [1000.0]
        let normalizedLarge = AudioFeatureExtractor.normalizeToDB((mel: large, frameCount: 1))
        #expect(normalizedLarge[0] <= 1.0)
    }

    // MARK: - preprocess: オンセット検出

    @Test func preprocessOnsetDetection() {
        let sampleRate = 22050
        // 先頭 0.5 秒を無音、その後に信号を入れる
        let silenceCount = sampleRate / 2
        let signalCount = sampleRate / 2
        var samples = [Float](repeating: 0, count: silenceCount + signalCount)
        for i in silenceCount..<(silenceCount + signalCount) {
            samples[i] = 0.5 * sin(Float(i) * 2 * .pi * 440 / Float(sampleRate))
        }

        let result = AudioFeatureExtractor.preprocess(samples)

        // 無音部分がトリムされるので、結果は元より短くなるはず
        #expect(result.count < samples.count)
        // 大幅に短くなっているはず（無音分がカットされる）
        #expect(result.count <= signalCount + AudioFeatureExtractor.onsetWindowLength)
    }

    @Test func preprocessNoSilence() {
        // 全体が信号の場合、ほぼそのまま返る（フェードアウト以外は変わらない）
        let count = 4096
        var samples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            samples[i] = 0.5 * sin(Float(i) * 2 * .pi * 440 / 22050)
        }

        let result = AudioFeatureExtractor.preprocess(samples)
        #expect(result.count == count)
    }

    // MARK: - preprocess: フェードアウト

    @Test func preprocessFadeOut() {
        // 一定振幅の信号を入力して、末尾15%がフェードアウトされることを確認
        let count = 1000
        // オンセット検出を通過するため十分な振幅の信号
        let samples = [Float](repeating: 0.5, count: count)

        let result = AudioFeatureExtractor.preprocess(samples)

        let fadeOutSamples = Int(Float(result.count) * AudioFeatureExtractor.fadeOutRatio)
        let fadeStart = result.count - fadeOutSamples

        // フェードアウト開始直前は元の振幅のまま
        #expect(abs(result[fadeStart - 1] - 0.5) < 1e-5)
        // 最後のサンプルはほぼゼロ
        #expect(abs(result[result.count - 1]) < 0.01)
        // フェードアウト中間は元の振幅より小さい
        let mid = fadeStart + fadeOutSamples / 2
        #expect(result[mid] < 0.5)
        #expect(result[mid] > 0)
    }

    // MARK: - STFT: 正弦波の周波数ピーク検出

    @Test func stftSinePeakDetection() {
        let sampleRate: Float = 22050
        let freq: Float = 1000  // 1000 Hz の正弦波
        let duration: Float = 0.1  // 100ms
        let sampleCount = Int(sampleRate * duration)

        var signal = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            signal[i] = sin(2 * .pi * freq * Float(i) / sampleRate)
        }

        let (magnitudes, frameCount) = AudioFeatureExtractor.stft(signal)
        #expect(frameCount > 0)

        let binCount = AudioFeatureExtractor.nFFT / 2 + 1
        let freqResolution = sampleRate / Float(AudioFeatureExtractor.nFFT)
        let expectedBin = Int(round(freq / freqResolution))

        // 中間フレームでピーク位置を確認
        let midFrame = frameCount / 2
        let offset = midFrame * binCount
        var maxBin = 0
        var maxValue: Float = 0
        for k in 0..<binCount {
            if magnitudes[offset + k] > maxValue {
                maxValue = magnitudes[offset + k]
                maxBin = k
            }
        }

        // ピークが期待するビン付近（±1）にあること
        #expect(abs(maxBin - expectedBin) <= 1,
                "1000Hz のピークが bin \(expectedBin) 付近に来るはず、実際は bin \(maxBin)")
    }
}
