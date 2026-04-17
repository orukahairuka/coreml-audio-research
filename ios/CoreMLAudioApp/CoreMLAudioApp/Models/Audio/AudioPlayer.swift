import AVFoundation

/// 波形データの再生を管理する
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {

    private var player: AVAudioPlayer?
    var onPlaybackFinished: (() -> Void)?

    var isPlaying: Bool {
        player?.isPlaying ?? false
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onPlaybackFinished?()
    }

    /// Float 波形配列から WAV を生成して再生する
    func play(waveform: [Float], sampleRate: Double) throws {
        #if os(iOS)
        try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)
        #endif

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw PlaybackError.formatCreationFailed
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(waveform.count)) else {
            throw PlaybackError.bufferCreationFailed
        }
        buffer.frameLength = AVAudioFrameCount(waveform.count)

        guard let channelData = buffer.floatChannelData?[0] else {
            throw PlaybackError.bufferCreationFailed
        }
        for i in 0..<waveform.count {
            channelData[i] = waveform[i]
        }

        // WAV ファイルとして Result/ に書き出す (Int16 PCM)
        let resultDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Result", isDirectory: true)
        if !FileManager.default.fileExists(atPath: resultDir.path) {
            try FileManager.default.createDirectory(at: resultDir, withIntermediateDirectories: true)
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let outputURL = resultDir.appendingPathComponent("output_\(formatter.string(from: Date())).wav")

        let wavSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        // do スコープで outputFile を閉じてから AVAudioPlayer で読み込む
        do {
            let outputFile = try AVAudioFile(
                forWriting: outputURL,
                settings: wavSettings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            try outputFile.write(from: buffer)
        }

        let newPlayer = try AVAudioPlayer(contentsOf: outputURL)
        newPlayer.delegate = self
        newPlayer.volume = 1.0
        newPlayer.prepareToPlay()
        newPlayer.play()
        player = newPlayer
    }

    func stop() {
        player?.stop()
    }

    enum PlaybackError: LocalizedError {
        case formatCreationFailed
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case .formatCreationFailed: return "オーディオフォーマットの作成に失敗しました。"
            case .bufferCreationFailed: return "オーディオバッファの作成に失敗しました。"
            }
        }
    }
}
