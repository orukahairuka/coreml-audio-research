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
    static let fadeOutRatio: Float = 0.15

    /// wav ファイルからメルスペクトログラムを抽出する
    /// - Returns: (mel: [T, nMels], frameCount: Int)
    static func extractMelSpectrogram(from url: URL) throws -> (mel: [Float], frameCount: Int) {
        let samples = try loadAudio(from: url)
        let preprocessed = preprocess(samples)
        let emphasized = applyPreemphasis(preprocessed)
        let magnitude = stft(emphasized)
        let melSpec = applyMelFilterbank(magnitude)
        let normalized = normalizeToDB(melSpec)
        let frameCount = normalized.count / nMels
        return (normalized, frameCount)
    }

    // MARK: - Audio Loading

    static func loadAudio(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
            throw FeatureExtractionError.invalidFormat
        }
        let originalFormat = file.processingFormat

        let frameCount = AVAudioFrameCount(file.length)
        guard let originalBuffer = AVAudioPCMBuffer(pcmFormat: originalFormat, frameCapacity: frameCount) else {
            throw FeatureExtractionError.bufferCreationFailed
        }
        try file.read(into: originalBuffer)

        // リサンプリング不要の場合
        if originalFormat.sampleRate == sampleRate && originalFormat.channelCount == 1 {
            guard let channelData = originalBuffer.floatChannelData?[0] else {
                throw FeatureExtractionError.bufferCreationFailed
            }
            return Array(UnsafeBufferPointer(start: channelData, count: Int(originalBuffer.frameLength)))
        }

        // リサンプリング
        guard let converter = AVAudioConverter(from: originalFormat, to: format) else {
            throw FeatureExtractionError.invalidFormat
        }
        let ratio = sampleRate / originalFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputFrameCount) else {
            throw FeatureExtractionError.bufferCreationFailed
        }

        var error: NSError?
        var hasData = true
        converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if hasData {
                hasData = false
                outStatus.pointee = .haveData
                return originalBuffer
            }
            outStatus.pointee = .endOfStream
            return nil
        }
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

        // fade out
        let fadeOutSamples = Int(Float(trimmed.count) * fadeOutRatio)
        if fadeOutSamples > 0 {
            let fadeStart = trimmed.count - fadeOutSamples
            for i in 0..<fadeOutSamples {
                let factor = 1.0 - Float(i) / Float(fadeOutSamples)
                trimmed[fadeStart + i] *= factor
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

        // Hann 窓: 切り出し区間の端を滑らかにゼロに落とし、ブツ切れによるノイズを防ぐ
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))

        // 信号が FFT サイズより短い場合、末尾をゼロで埋める
        let padded = signal + [Float](repeating: 0, count: max(0, fftSize - signal.count))

        // hopLength ずつずらして何フレーム分析できるか
        let frameCount = max(0, (padded.count - fftSize) / hopLength) + 1

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

                        // magnitude（振幅）計算
                        let offset = frame * binCount
                        // DC 成分（0Hz、信号の平均値に相当）
                        magnitudes[offset] = abs(splitComplex.realp[0]) / Float(fftSize)
                        // Nyquist 成分（サンプルレートの半分の周波数）
                        magnitudes[offset + halfFFT] = abs(splitComplex.imagp[0]) / Float(fftSize)
                        // その他の周波数。2.0 は実数信号の対称性による補正係数
                        for i in 1..<halfFFT {
                            let re = splitComplex.realp[i]
                            let im = splitComplex.imagp[i]
                            magnitudes[offset + i] = sqrt(re * re + im * im) * 2.0 / Float(fftSize)
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

    /// Hz → メル変換
    static func hzToMel(_ hz: Float) -> Float {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }

    /// メル → Hz 変換
    static func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
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

    /// メルフィルタバンク行列を生成する (librosa.filters.mel 相当)
    static func createMelFilterbank(sampleRate: Float, nFFT: Int, nMels: Int, fMin: Float, fMax: Float) -> [Float] {
        let binCount = nFFT / 2 + 1

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // nMels + 2 個の等間隔メル周波数
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }

        // メル → Hz → FFT bin
        let fftFreqs = melPoints.map { melToHz($0) }
        let bins = fftFreqs.map { $0 * Float(nFFT) / sampleRate }

        // 三角フィルタ
        var filterbank = [Float](repeating: 0, count: nMels * binCount)
        for m in 0..<nMels {
            let left = bins[m]
            let center = bins[m + 1]
            let right = bins[m + 2]

            for k in 0..<binCount {
                let freq = Float(k)
                if freq >= left && freq <= center && center > left {
                    filterbank[m * binCount + k] = (freq - left) / (center - left)
                } else if freq > center && freq <= right && right > center {
                    filterbank[m * binCount + k] = (right - freq) / (right - center)
                }
            }
        }

        // Slaney 正規化
        for m in 0..<nMels {
            let bandwidth = 2.0 * (Self.melToHz(melPoints[m + 2]) - Self.melToHz(melPoints[m])) / sampleRate
            if bandwidth > 0 {
                for k in 0..<binCount {
                    filterbank[m * binCount + k] /= bandwidth
                }
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
