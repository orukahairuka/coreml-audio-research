import Testing
@testable import CoreMLAudioApp
import Foundation

struct AudioFeatureExtractorTests {

    // MARK: - Hz ↔ Mel ラウンドトリップ

    @Test(arguments: [0.0, 440.0, 1000.0, 8000.0, 11025.0] as [Double])
    func hzToMelRoundTrip(hz: Double) {
        let mel = AudioFeatureExtractor.hzToMel(hz)
        let recovered = AudioFeatureExtractor.melToHz(mel)
        #expect(abs(recovered - hz) < 1e-6, "Hz→Mel→Hz が元に戻らない: \(hz) → \(recovered)")
    }

    @Test func hzToMelKnownValues() {
        // Slaney 尺度: 1000 Hz で 15、6400 Hz で 15 + 27
        #expect(AudioFeatureExtractor.hzToMel(0) == 0)
        #expect(abs(AudioFeatureExtractor.hzToMel(1000) - 15) < 1e-9)
        #expect(abs(AudioFeatureExtractor.hzToMel(6400) - 42) < 1e-9)
    }

    // MARK: - メルフィルタバンク

    /// librosa.filters.mel(sr=22050, n_fft=1024, n_mels=256) の値（librosa 0.11.0）
    @Test(arguments: [
        (0, 1, 0.026015037670731544, 0.026015037670731544),
        (64, 39, 0.06674706935882568, 0.06674706935882568),
        (128, 93, 0.0282394140958786, 0.044648677110672),
        (255, 505, 0.00666936906054616, 0.046399421989917755),
    ] as [(Int, Int, Double, Double)])
    func melFilterbankMatchesLibrosa(band: Int, peakBin: Int, peak: Double, sum: Double) {
        let binCount = AudioFeatureExtractor.nFFT / 2 + 1
        let filterbank = AudioFeatureExtractor.createMelFilterbank(
            sampleRate: 22050, nFFT: 1024, nMels: 256, fMin: 0, fMax: 11025
        )
        let row = Array(filterbank[(band * binCount)..<((band + 1) * binCount)])
        #expect(abs(Double(row[peakBin]) - peak) < 1e-7)
        #expect(abs(Double(row.reduce(0, +)) - sum) < 1e-6)
    }

    /// librosa と同じく、256 帯域のどのフィルタにも 1 つ以上の FFT ビンが入る
    @Test func melFilterbankHasNoEmptyBand() {
        let binCount = AudioFeatureExtractor.nFFT / 2 + 1
        let filterbank = AudioFeatureExtractor.createMelFilterbank(
            sampleRate: 22050, nFFT: 1024, nMels: 256, fMin: 0, fMax: 11025
        )
        for band in 0..<256 {
            #expect(filterbank[(band * binCount)..<((band + 1) * binCount)].contains(where: { $0 > 0 }), "帯域 \(band)")
        }
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

        let fadeOutSamples = Int(Double(result.count) * AudioFeatureExtractor.fadeOutRatio)
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

    /// np.linspace(1.0, 0.0, n) と同じく、フェードの先頭の係数は 1、末尾の係数は 0
    @Test func preprocessFadeOutMatchesLinspace() {
        let samples = [Float](repeating: 0.5, count: 1000)
        let result = AudioFeatureExtractor.preprocess(samples)
        // 末尾 150 サンプルがフェードの区間
        #expect(result[849] == 0.5)
        #expect(result[850] == 0.5)
        #expect(result[999] == 0)
        #expect(abs(result[851] - 0.5 * Float(1.0 - 1.0 / 149.0)) < 1e-6)
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

    /// フレーム数は librosa.stft（center=True）と同じ len / hop + 1
    @Test(arguments: [(0, 1), (255, 1), (256, 2), (1000, 4), (67939, 266), (75520, 296)] as [(Int, Int)])
    func stftFrameCountMatchesCenteredLibrosa(sampleCount: Int, frames: Int) {
        let (_, frameCount) = AudioFeatureExtractor.stft([Float](repeating: 0, count: sampleCount))
        #expect(frameCount == frames)
    }

    /// フレーム 0 の窓の中心は信号の先頭にある（先頭のインパルスの振幅は窓の最大値 1 のまま）
    @Test func stftFirstFrameIsCenteredOnFirstSample() {
        var signal = [Float](repeating: 0, count: 1024)
        signal[0] = 1
        let (magnitudes, _) = AudioFeatureExtractor.stft(signal)
        #expect(abs(magnitudes[0] - 1) < 1e-5)
        #expect(abs(magnitudes[100] - 1) < 1e-5)
    }

    /// 振幅 1 の正弦波がビンの中心にあるとき、librosa.stft の |X[k]| は Σw[n] / 2 = 256 になる
    @Test func stftMagnitudeMatchesLibrosaScale() {
        let bin = 32
        let signal = (0..<4096).map({ sin(2 * Float.pi * Float(bin) * Float($0) / 1024) })
        let (magnitudes, frameCount) = AudioFeatureExtractor.stft(signal)
        let binCount = AudioFeatureExtractor.nFFT / 2 + 1
        let value = magnitudes[(frameCount / 2) * binCount + bin]
        #expect(abs(value - 256) < 0.01, "実際は \(value)")
    }
}
