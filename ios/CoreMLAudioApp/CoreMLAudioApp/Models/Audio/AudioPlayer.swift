import AVFoundation

/// 波形データの再生を管理する
final class AudioPlayer: NSObject, AVAudioPlayerDelegate {

    private var player: AVAudioPlayer?
    var onPlaybackFinished: (() -> Void)?

    var isPlaying: Bool {
        player?.isPlaying ?? false
    }

    override init() {
        super.init()
        #if os(iOS)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleInterruption(_:)),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
        #endif
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onPlaybackFinished?()
    }

    #if os(iOS)
    /// 再生中に電話などで AVAudioSession が中断されると didFinishPlaying が呼ばれず、
    /// playOutputAndAwaitCompletion の continuation が永久に resume されずハングする。
    /// 中断開始を「再生終了」とみなして完了コールバックを発火し、待機側を解放する。
    @objc private func handleInterruption(_ notification: Notification) {
        guard let info = notification.userInfo,
              let raw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              AVAudioSession.InterruptionType(rawValue: raw) == .began else { return }
        player?.stop()
        onPlaybackFinished?()
    }
    #endif

    /// Float 波形配列から WAV を生成して再生する
    ///
    /// - Parameter baseName: 拡張子抜きのファイル名 (例: `"Float16_cpuAndGPU"`)。
    ///   nil の場合はタイムスタンプ命名 (`output_yyyyMMdd_HHmmss.wav`) で保存する。
    func play(waveform: [Float], sampleRate: Double, baseName: String? = nil) throws {
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
        let fileName: String
        if let baseName = baseName {
            fileName = "output_\(baseName).wav"
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss"
            fileName = "output_\(formatter.string(from: Date())).wav"
        }
        let outputURL = resultDir.appendingPathComponent(fileName)

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
        // play() は再生開始に失敗すると throw せず false を返す。ここで弾かないと
        // didFinishPlaying が永久に来ず、呼び出し側の継続が宙吊りになる。
        guard newPlayer.play() else {
            throw PlaybackError.playbackStartFailed
        }
        player = newPlayer
    }

    /// 再生を止める。stop() では AVAudioPlayer の delegate (didFinishPlaying) が呼ばれないので、
    /// 待機側 (continuation) を解放するため明示的に完了コールバックを呼ぶ。
    func stop() {
        player?.stop()
        onPlaybackFinished?()
    }

    enum PlaybackError: LocalizedError {
        case formatCreationFailed
        case bufferCreationFailed
        case playbackStartFailed

        var errorDescription: String? {
            switch self {
            case .formatCreationFailed: return "オーディオフォーマットの作成に失敗しました。"
            case .bufferCreationFailed: return "オーディオバッファの作成に失敗しました。"
            case .playbackStartFailed: return "再生を開始できませんでした。"
            }
        }
    }
}
