import AVFoundation

/// 波形データの再生を管理する
final class AudioPlayer {

    private var player: AVAudioPlayer?

    var isPlaying: Bool {
        player?.isPlaying ?? false
    }

    /// Float 波形配列から WAV を生成して再生する
    func play(waveform: [Float], sampleRate: Double) throws {
        #if os(iOS)
        try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)
        #endif

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(waveform.count))!
        buffer.frameLength = AVAudioFrameCount(waveform.count)

        let channelData = buffer.floatChannelData![0]
        for i in 0..<waveform.count {
            channelData[i] = waveform[i]
        }

        // WAV ファイルとして書き出す (Int16 PCM)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("output.wav")
        try? FileManager.default.removeItem(at: tempURL)

        let wavSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let outputFile = try AVAudioFile(
                forWriting: tempURL,
                settings: wavSettings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            try outputFile.write(from: buffer)
        }

        let newPlayer = try AVAudioPlayer(contentsOf: tempURL)
        newPlayer.volume = 1.0
        newPlayer.prepareToPlay()
        newPlayer.play()
        player = newPlayer
    }

    func stop() {
        player?.stop()
    }
}
