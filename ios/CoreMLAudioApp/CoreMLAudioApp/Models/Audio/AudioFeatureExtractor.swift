import Accelerate
import AVFoundation

/// PronounSE の get_spectrograms と同等のメルスペクトログラム抽出を行う
struct AudioFeatureExtractor {

    // hyperparams.py と一致させる
    static let sampleRate: Double = 22050
    static let nFFT = 1024
    static let hopLength = 256
    static let winLength = 1024
    static let nMels = 256
    static let maxDB: Float = 100
    static let refDB: Float = 20
    static let preemphasisCoeff: Float = 0.97

    // PronounSE/Transformer/utils.py の preprocess() デフォルト引数と一致させる
    /// 音の開始を判定するエネルギーの閾値。窓内の二乗和がこの値を超えたら「音が始まった」とみなす
    static let onsetThreshold: Float = 0.08
    /// オンセット検出時に窓をずらす幅（サンプル数）
    static let onsetShift = 256
    /// オンセット検出時にエネルギーを測る窓の幅（サンプル数）
    static let onsetWindowLength = 1024
    /// フェードアウトする割合。末尾 15% の区間で音量を下げる
    static let fadeOutRatio: Double = 0.15

    /// wav ファイルからメルスペクトログラムを抽出する
    /// - Returns: (mel: [T, nMels], frameCount: Int)
    static func extractMelSpectrogram(
        from url: URL, optimizations: SynthesisOptimizations = .adopted
    ) throws -> (mel: [Float], frameCount: Int) {
        extractMelSpectrogram(fromSamples: try loadAudio(from: url), optimizations: optimizations)
    }

    /// 読み込み済み（22050 Hz・モノラル）の波形からメルスペクトログラムを抽出する
    /// - Returns: (mel: [T, nMels], frameCount: Int)
    /// - Parameter optimizations: 高速化の指定（結果は変わらない）
    static func extractMelSpectrogram(
        fromSamples samples: [Float], optimizations: SynthesisOptimizations = .adopted
    ) -> (mel: [Float], frameCount: Int) {
        let preprocessed = preprocess(samples)
        let emphasized = applyPreemphasis(preprocessed)
        let magnitude = stft(emphasized)
        let melSpec = optimizations.contains(.sparseMelFilterbank)
            ? applyMelFilterbankSparse(magnitude) : applyMelFilterbank(magnitude)
        let normalized = normalizeToDB(melSpec)
        let frameCount = normalized.count / nMels
        return (normalized, frameCount)
    }

    // MARK: - Audio Loading

    /// 入力のサンプリング周波数ごとのリサンプリングの方法
    enum ResamplingMethod: String {
        /// 22.05 kHz の入力はそのまま使う
        case none
        /// 44.1 kHz の入力は soxr_hq を再現した FIR を使う（公開実装と一致する）
        case soxrHQDecimator
        /// それ以外の周波数は AVAudioConverter を使う（公開実装とは一致しない）
        case audioConverter
    }

    static func resamplingMethod(forSourceRate rate: Double) -> ResamplingMethod {
        if rate == sampleRate { return .none }
        if rate == SoxrHQDecimator.sourceRate { return .soxrHQDecimator }
        return .audioConverter
    }

    /// 音声ファイルを 22050 Hz・モノラルの波形として読み込む
    static func loadAudio(from url: URL) throws -> [Float] {
        let source = try readMonoSource(from: url)
        return try resample(source.samples, sourceRate: source.sampleRate)
    }

    /// 音声ファイルを元のサンプリング周波数のまま読み込み、モノラルにする
    /// （複数チャンネルは librosa.to_mono と同じく平均する）
    static func readMonoSource(from url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw FeatureExtractionError.bufferCreationFailed
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else {
            throw FeatureExtractionError.bufferCreationFailed
        }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(format.channelCount)
        if channelCount == 1 {
            return (Array(UnsafeBufferPointer(start: channels[0], count: frameCount)), format.sampleRate)
        }
        var mono = [Float](repeating: 0, count: frameCount)
        for channel in 0..<channelCount {
            for i in 0..<frameCount {
                mono[i] += channels[channel][i]
            }
        }
        let divisor = Float(channelCount)
        for i in 0..<frameCount {
            mono[i] /= divisor
        }
        return (mono, format.sampleRate)
    }

    /// モノラルの波形を 22050 Hz に変換する
    static func resample(_ samples: [Float], sourceRate: Double) throws -> [Float] {
        switch resamplingMethod(forSourceRate: sourceRate) {
        case .none: return samples
        case .soxrHQDecimator: return SoxrHQDecimator.decimate(samples)
        case .audioConverter: return try convertWithAudioConverter(samples, sourceRate: sourceRate)
        }
    }

    private static func convertWithAudioConverter(_ samples: [Float], sourceRate: Double) throws -> [Float] {
        guard !samples.isEmpty else { return [] }
        guard let sourceFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sourceRate, channels: 1, interleaved: false),
              let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw FeatureExtractionError.invalidFormat
        }
        let frameCount = AVAudioFrameCount(samples.count)
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount),
              let sourceChannel = sourceBuffer.floatChannelData?[0] else {
            throw FeatureExtractionError.bufferCreationFailed
        }
        sourceBuffer.frameLength = frameCount
        samples.withUnsafeBufferPointer({ pointer in
            sourceChannel.update(from: pointer.baseAddress!, count: samples.count)
        })
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * sampleRate / sourceRate)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
            throw FeatureExtractionError.bufferCreationFailed
        }

        var error: NSError?
        var hasData = true
        converter.convert(to: outputBuffer, error: &error, withInputFrom: { _, outStatus in
            if hasData {
                hasData = false
                outStatus.pointee = .haveData
                return sourceBuffer
            }
            outStatus.pointee = .endOfStream
            return nil
        })
        if let error { throw error }

        guard let channelData = outputBuffer.floatChannelData?[0] else {
            throw FeatureExtractionError.bufferCreationFailed
        }
        return Array(UnsafeBufferPointer(start: channelData, count: Int(outputBuffer.frameLength)))
    }

    enum FeatureExtractionError: LocalizedError {
        case invalidFormat
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case .invalidFormat: return "オーディオフォーマットの作成に失敗しました"
            case .bufferCreationFailed: return "オーディオバッファの作成に失敗しました"
            }
        }
    }

    // MARK: - Preprocessing

    /// onset 検出 + fade out (PronounSE/Transformer/utils.py の preprocess と同等)
    static func preprocess(_ y: [Float]) -> [Float] {
        // onset 検出, 音がはじまる位置を見つける
        var onsetIndex = 0
        let sq = y.map { $0 * $0 }
        for i in stride(from: 0, to: sq.count - onsetWindowLength, by: onsetShift) {
            let sum = sq[i..<(i + onsetWindowLength)].reduce(0, +)
            if sum > onsetThreshold {
                onsetIndex = i
                break
            }
        }

        var trimmed = Array(y[onsetIndex...])

        // fade out（np.linspace(1.0, 0.0, n) を掛ける。公開実装と同じく倍精度で計算する）
        let fadeOutSamples = Int(Double(trimmed.count) * fadeOutRatio)
        if fadeOutSamples > 0 {
            let fadeStart = trimmed.count - fadeOutSamples
            let step = fadeOutSamples > 1 ? -1.0 / Double(fadeOutSamples - 1) : 0.0
            for i in 0..<fadeOutSamples {
                let factor = i == fadeOutSamples - 1 && fadeOutSamples > 1 ? 0.0 : 1.0 + Double(i) * step
                trimmed[fadeStart + i] = Float(Double(trimmed[fadeStart + i]) * factor)
            }
        }

        return trimmed
    }

    /// プリエンファシスフィルタ: y[n] = x[n] - coeff * x[n-1]
    static func applyPreemphasis(_ y: [Float]) -> [Float] {
        guard y.count > 1 else { return y }
        var result = [Float](repeating: 0, count: y.count)
        result[0] = y[0]
        for i in 1..<y.count {
            result[i] = y[i] - preemphasisCoeff * y[i - 1]
        }
        return result
    }

    // MARK: - STFT

    /// STFT を実行し magnitude spectrogram を返す
    /// - Returns: [T, 1 + nFFT/2] の magnitude (行優先)
    static func stft(_ signal: [Float]) -> (magnitudes: [Float], frameCount: Int) {
        let fftSize = nFFT           // FFT のサイズ（1024）
        let halfFFT = fftSize / 2    // FFT の半分（512）
        let binCount = halfFFT + 1   // 周波数ビンの数（513）。FFT 結果は対称なので半分+1 で十分

        // Hann 窓: 切り出し区間の端を滑らかにゼロに落とし、ブツ切れによるノイズを防ぐ。
        // librosa の窓（最大値 1 の周期 Hann 窓）と同じにするため DENORM を使う（NORM は 1.633 倍になる）
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_DENORM))

        // librosa.stft の center=True, pad_mode="constant" と同じく、信号の両端を nFFT / 2 ずつゼロで埋める。
        // フレーム t の窓の中心が元の信号の t * hopLength サンプル目になる
        let edge = [Float](repeating: 0, count: halfFFT)
        let padded = edge + signal + edge

        // hopLength ずつずらして何フレーム分析できるか（= signal.count / hopLength + 1）
        let frameCount = (padded.count - fftSize) / hopLength + 1

        // Accelerate の FFT エンジンを準備
        let log2n = vDSP_Length(log2(Float(fftSize)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return ([], 0)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }  // 関数終了時に解放

        var magnitudes = [Float](repeating: 0, count: frameCount * binCount)  // 全フレームの振幅スペクトル
        var windowedFrame = [Float](repeating: 0, count: fftSize)  // 窓関数適用済みの1フレーム分
        var realPart = [Float](repeating: 0, count: halfFFT)  // FFT 結果の実数部
        var imagPart = [Float](repeating: 0, count: halfFFT)  // FFT 結果の虚数部

        for frame in 0..<frameCount {
            let start = frame * hopLength  // このフレームの開始位置
            let end = min(start + fftSize, padded.count)
            let available = end - start    // 実際に使えるサンプル数

            // 窓関数を掛けて切り出し。足りない部分はゼロ
            for i in 0..<fftSize {
                windowedFrame[i] = i < available ? padded[start + i] * window[i] : 0
            }

            // Accelerate の FFT を実行（ポインタ操作が必要）
            windowedFrame.withUnsafeMutableBufferPointer { bufPtr in
                realPart.withUnsafeMutableBufferPointer { realPtr in
                    imagPart.withUnsafeMutableBufferPointer { imagPtr in
                        // 実数部と虚数部を分離した形式に変換
                        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                        bufPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfFFT) { complexPtr in
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfFFT))
                        }
                        // FFT 実行
                        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                        // magnitude（振幅）計算。librosa.stft と同じ |X[k]| にする。
                        // vDSP_fft_zrip の出力は DFT の 2 倍なので、すべてのビンを 2 で割る
                        let offset = frame * binCount
                        // DC 成分（0Hz、信号の平均値に相当）
                        magnitudes[offset] = abs(splitComplex.realp[0]) / 2.0
                        // Nyquist 成分（サンプルレートの半分の周波数）
                        magnitudes[offset + halfFFT] = abs(splitComplex.imagp[0]) / 2.0
                        // その他の周波数
                        for i in 1..<halfFFT {
                            let re = splitComplex.realp[i]
                            let im = splitComplex.imagp[i]
                            magnitudes[offset + i] = sqrt(re * re + im * im) / 2.0
                        }
                    }
                }
            }
        }

        return (magnitudes, frameCount)
    }

    // MARK: - Mel Filterbank

    /// メルフィルタバンクを適用する
    /// - Returns: [T, nMels] のメルスペクトログラム (行優先)
    static func applyMelFilterbank(_ stftResult: (magnitudes: [Float], frameCount: Int)) -> (mel: [Float], frameCount: Int) {
        let (magnitudes, frameCount) = stftResult
        let binCount = nFFT / 2 + 1

        // メルフィルタバンク行列を生成 [nMels, binCount]
        let filterbank = createMelFilterbank(
            sampleRate: Float(sampleRate),
            nFFT: nFFT,
            nMels: nMels,
            fMin: 0,
            fMax: Float(sampleRate / 2)
        )

        // 行列積: mel = filterbank @ magnitude.T → [nMels, T] → transpose → [T, nMels]
        var mel = [Float](repeating: 0, count: frameCount * nMels)

        for t in 0..<frameCount {
            for m in 0..<nMels {
                var sum: Float = 0
                let filterOffset = m * binCount
                let magOffset = t * binCount
                for k in 0..<binCount {
                    sum += filterbank[filterOffset + k] * magnitudes[magOffset + k]
                }
                mel[t * nMels + m] = sum
            }
        }

        return (mel, frameCount)
    }

    /// メルフィルタバンクと、帯域ごとの 0 でない FFT ビンの範囲（一度だけ作って再利用する）
    private static let sparseFilterbank: (weights: [Float], ranges: [Range<Int>]) = {
        let binCount = nFFT / 2 + 1
        let weights = createMelFilterbank(
            sampleRate: Float(sampleRate), nFFT: nFFT, nMels: nMels, fMin: 0, fMax: Float(sampleRate / 2)
        )
        let ranges = (0..<nMels).map({ band -> Range<Int> in
            let row = weights[(band * binCount)..<((band + 1) * binCount)]
            guard let first = row.firstIndex(where: { $0 != 0 }), let last = row.lastIndex(where: { $0 != 0 }) else {
                return 0..<0
            }
            return (first - band * binCount)..<(last - band * binCount + 1)
        })
        return (weights, ranges)
    }()

    /// `applyMelFilterbank` と同じ順序で、0 でない範囲だけを足し合わせる
    /// （0 の項を足しても和は変わらないので、結果はビット単位で同じになる）
    static func applyMelFilterbankSparse(_ stftResult: (magnitudes: [Float], frameCount: Int)) -> (mel: [Float], frameCount: Int) {
        let (magnitudes, frameCount) = stftResult
        let binCount = nFFT / 2 + 1
        let (weights, ranges) = sparseFilterbank
        var mel = [Float](repeating: 0, count: frameCount * nMels)
        mel.withUnsafeMutableBufferPointer({ output in
            weights.withUnsafeBufferPointer({ filter in
                magnitudes.withUnsafeBufferPointer({ spectrum in
                    for t in 0..<frameCount {
                        let frame = t * binCount
                        for m in 0..<nMels {
                            let row = m * binCount
                            var sum: Float = 0
                            for k in ranges[m] {
                                sum += filter[row + k] * spectrum[frame + k]
                            }
                            output[t * nMels + m] = sum
                        }
                    }
                })
            })
        })
        return (mel, frameCount)
    }

    /// Hz → メル変換（Slaney 尺度。librosa.hz_to_mel の htk=False と同じ）
    /// 1000 Hz までは線形、それより上は対数
    static func hzToMel(_ hz: Double) -> Double {
        if hz < 1000.0 { return 3.0 * hz / 200.0 }
        return 15.0 + 27.0 * log10(hz / 1000.0) / log10(6.4)
    }

    /// メル → Hz 変換（Slaney 尺度。librosa.mel_to_hz の htk=False と同じ）
    static func melToHz(_ mel: Double) -> Double {
        if mel < 15.0 { return 200.0 * mel / 3.0 }
        return 1000.0 * pow(6.4, (mel - 15.0) / 27.0)
    }

    /// デエンファシスフィルタ: y[n] = x[n] + coeff * y[n-1]
    static func applyDeemphasis(_ signal: [Float]) -> [Float] {
        guard signal.count > 1 else { return signal }
        var result = signal
        for i in 1..<result.count {
            result[i] = result[i] + preemphasisCoeff * result[i - 1]
        }
        return result
    }

    /// メルフィルタバンク行列を生成する（librosa.filters.mel の htk=False, norm="slaney" と同じ）
    ///
    /// 帯域 m の三角フィルタの両端を f₋, f₊ とすると、面積が 1 になるよう 2 / (f₊ - f₋) を掛ける
    static func createMelFilterbank(sampleRate: Float, nFFT: Int, nMels: Int, fMin: Float, fMax: Float) -> [Float] {
        let binCount = nFFT / 2 + 1
        let rate = Double(sampleRate)
        let melMin = hzToMel(Double(fMin))
        let melMax = hzToMel(Double(fMax))

        // nMels + 2 個の等間隔メル周波数を Hz にしたもの
        let edges = (0..<(nMels + 2)).map({ index in
            melToHz(melMin + Double(index) * (melMax - melMin) / Double(nMels + 1))
        })

        var filterbank = [Float](repeating: 0, count: nMels * binCount)
        for m in 0..<nMels {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2])
            let areaNorm = 2.0 / (right - left)
            for k in 0..<binCount {
                let hz = Double(k) * rate / Double(nFFT)
                let weight = max(0.0, min((hz - left) / (center - left), (right - hz) / (right - center)))
                filterbank[m * binCount + k] = Float(weight * areaNorm)
            }
        }
        return filterbank
    }

    // MARK: - Normalization

    /// dB 変換 + 正規化 (get_spectrograms と同等)
    static func normalizeToDB(_ melResult: (mel: [Float], frameCount: Int)) -> [Float] {
        let (mel, _) = melResult
        return mel.map { value in
            // to decibel
            let db = 20.0 * log10(max(1e-5, value))
            // normalize
            let normalized = (db - refDB + maxDB) / maxDB
            return min(1.0, max(1e-8, normalized))
        }
    }

    // MARK: - Display Mel Spectrogram

    /// 可視化用メルスペクトログラムを計算する (librosa.power_to_db 相当, -80〜0 dB)
    static func melSpectrogramForDisplay(from url: URL) throws -> (mel: [Float], frameCount: Int) {
        let samples = try loadAudio(from: url)
        let preprocessed = preprocess(samples)
        let emphasized = applyPreemphasis(preprocessed)
        let magnitude = stft(emphasized)
        let melSpec = applyMelFilterbank(magnitude)
        return powerToDb(melSpec)
    }

    /// power → dB 変換 (librosa.power_to_db(S, ref=np.max) 相当)
    /// 戻り値は -80〜0 dB レンジ
    static func powerToDb(_ melResult: (mel: [Float], frameCount: Int)) -> (mel: [Float], frameCount: Int) {
        let (mel, frameCount) = melResult
        // ref = max(mel)
        let refValue = mel.max() ?? 1e-10
        let refLog = 10.0 * log10(max(1e-10, refValue))
        let minDb: Float = -80.0
        let result = mel.map { value in
            let db = 10.0 * log10(max(1e-10, value)) - refLog
            return max(minDb, db)
        }
        return (result, frameCount)
    }

    /// CoreML Decoder の出力メル (正規化済み [0〜1]) を可視化用 dB スケールに逆変換する
    /// normalizeToDB の逆: db = normalized * maxDB + refDB - maxDB
    /// → power_to_db 相当の -80〜0 レンジに変換
    static func denormalizeToDisplayDb(_ normalized: [Float]) -> [Float] {
        return normalized.map { value in
            // normalizeToDB の逆変換: db = value * maxDB + refDB - maxDB
            let db = value * maxDB + refDB - maxDB
            return max(-80.0, min(0.0, db))
        }
    }
}
