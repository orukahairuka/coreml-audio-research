import AVFoundation

/// Float 波形を 16 ビット PCM の WAV として保存する
///
/// アプリの再生（`AudioPlayer`）と卒論の計測モードで同じ書き出し経路を使うため、
/// AVAudioFile による Float32 → Int16 の変換をここにまとめる。
/// 範囲 [-1, 1] を超えた値がどう Int16 に変換されるか（クリップかラップアラウンドか）は
/// `WavWriterTests` で確認している。
enum WavWriter {

    enum WriteError: LocalizedError {
        case formatCreationFailed
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case .formatCreationFailed: return "オーディオフォーマットの作成に失敗しました。"
            case .bufferCreationFailed: return "オーディオバッファの作成に失敗しました。"
            }
        }
    }

    /// 波形を Float32 の PCM バッファに詰める（再生にも使う）
    static func makeBuffer(waveform: [Float], sampleRate: Double) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw WriteError.formatCreationFailed
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(waveform.count)) else {
            throw WriteError.bufferCreationFailed
        }
        buffer.frameLength = AVAudioFrameCount(waveform.count)

        guard let channelData = buffer.floatChannelData?[0] else {
            throw WriteError.bufferCreationFailed
        }
        for i in 0..<waveform.count {
            channelData[i] = waveform[i]
        }
        return buffer
    }

    /// バッファを 16 ビット PCM の WAV として `url` に書き出す
    static func writeInt16(buffer: AVAudioPCMBuffer, to url: URL) throws {
        let wavSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: buffer.format.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        // do スコープで outputFile を閉じる（閉じるまでヘッダが確定しない）
        do {
            let outputFile = try AVAudioFile(
                forWriting: url,
                settings: wavSettings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            try outputFile.write(from: buffer)
        }
    }

    /// 波形を 16 ビット PCM の WAV として `url` に書き出す
    static func writeInt16(waveform: [Float], sampleRate: Double, to url: URL) throws {
        let buffer = try makeBuffer(waveform: waveform, sampleRate: sampleRate)
        try writeInt16(buffer: buffer, to: url)
    }
}
