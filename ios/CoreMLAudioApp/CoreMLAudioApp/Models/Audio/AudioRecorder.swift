import AVFoundation

/// マイクからの音声録音を管理する
final class AudioRecorder: NSObject, AVAudioRecorderDelegate {

    /// 録音の最大時間（秒）。CoreML モデルの入力上限 1000 フレーム ≒ 11.6 秒に対応
    static let maxDuration: TimeInterval = 11.0

    private var recorder: AVAudioRecorder?
    private var timer: Timer?

    /// 録音完了時に呼ばれる。成功時は URL、失敗時は nil
    var onRecordingFinished: ((URL?) -> Void)?
    var onTimeUpdate: ((TimeInterval) -> Void)?

    private(set) var isRecording = false

    // MARK: - Recording

    func startRecording() throws {
        guard !isRecording else { return }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)

        let url = Self.newRecordingURL()
        try Self.ensureRecordingsDirectory()

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: AudioFeatureExtractor.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        let newRecorder = try AVAudioRecorder(url: url, settings: settings)
        newRecorder.delegate = self
        newRecorder.prepareToRecord()
        guard newRecorder.record(forDuration: Self.maxDuration) else {
            try? FileManager.default.removeItem(at: url)
            throw RecordingError.startFailed
        }

        recorder = newRecorder
        isRecording = true

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true, block: { [weak self] _ in
            guard let self, let recorder = self.recorder else { return }
            self.onTimeUpdate?(recorder.currentTime)
        })
    }

    func stopRecording() {
        recorder?.stop()
    }

    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        timer?.invalidate()
        timer = nil
        isRecording = false

        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default)
        try? session.setActive(true)

        if flag {
            onRecordingFinished?(recorder.url)
        } else {
            try? FileManager.default.removeItem(at: recorder.url)
            onRecordingFinished?(nil)
        }
    }

    // MARK: - File Management

    static var recordingsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Recordings", isDirectory: true)
    }

    static func ensureRecordingsDirectory() throws {
        let dir = recordingsDirectory
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    static func recordings() -> [URL] {
        let dir = recordingsDirectory
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        return files
            .filter { $0.pathExtension == "wav" }
            .sorted(by: { ($0.lastPathComponent) > ($1.lastPathComponent) })
    }

    static func deleteRecording(at url: URL) throws {
        try FileManager.default.removeItem(at: url)
    }

    // MARK: - Private

    private static func newRecordingURL() -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let name = "recording_\(formatter.string(from: Date())).wav"
        return recordingsDirectory.appendingPathComponent(name)
    }

    enum RecordingError: LocalizedError {
        case startFailed

        var errorDescription: String? {
            switch self {
            case .startFailed: return "録音を開始できませんでした。"
            }
        }
    }
}
