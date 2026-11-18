import Foundation

/// 前処理の各段階の出力（公開実装と段階ごとに比べるためのもの）
struct PreprocessStages {
    /// 読み込みとリサンプリングの後の波形
    let load: [Float]
    /// オンセット検出とフェードアウトの後の波形
    let preprocessed: [Float]
    /// プリエンファシスの後の波形
    let emphasized: [Float]
    /// 振幅スペクトログラム [T × 513]
    let magnitude: [Float]
    /// メルスペクトログラム（dB 変換の前）[T × 256]
    let mel: [Float]
    /// 正規化済みメルスペクトログラム [T × 256]
    let normalized: [Float]
    let frameCount: Int

    /// 修正後の前処理（`AudioFeatureExtractor`）
    static func current(url: URL) throws -> PreprocessStages {
        typealias E = AudioFeatureExtractor
        let load = try E.loadAudio(from: url)
        let preprocessed = E.preprocess(load)
        let emphasized = E.applyPreemphasis(preprocessed)
        let magnitude = E.stft(emphasized)
        let mel = E.applyMelFilterbank(magnitude)
        return PreprocessStages(
            load: load, preprocessed: preprocessed, emphasized: emphasized,
            magnitude: magnitude.magnitudes, mel: mel.mel, normalized: E.normalizeToDB(mel),
            frameCount: mel.frameCount
        )
    }

    /// 修正前の前処理（`LegacyAudioFeatureExtractor`）
    static func legacy(url: URL) throws -> PreprocessStages {
        typealias E = LegacyAudioFeatureExtractor
        let load = try E.loadAudio(from: url)
        let preprocessed = E.preprocess(load)
        let emphasized = E.applyPreemphasis(preprocessed)
        let magnitude = E.stft(emphasized)
        let mel = E.applyMelFilterbank(magnitude)
        return PreprocessStages(
            load: load, preprocessed: preprocessed, emphasized: emphasized,
            magnitude: magnitude.magnitudes, mel: mel.mel, normalized: E.normalizeToDB(mel),
            frameCount: mel.frameCount
        )
    }

    /// 保存するファイル名の段階の名前と、[行 × 列] の配列
    var arrays: [(name: String, values: [Float], rows: Int, cols: Int)] {
        [
            ("load", load, 1, load.count),
            ("prep", preprocessed, 1, preprocessed.count),
            ("emph", emphasized, 1, emphasized.count),
            ("mag", magnitude, frameCount, AudioFeatureExtractor.nFFT / 2 + 1),
            ("mel", mel, frameCount, AudioFeatureExtractor.nMels),
            ("norm", normalized, frameCount, AudioFeatureExtractor.nMels),
        ]
    }
}
