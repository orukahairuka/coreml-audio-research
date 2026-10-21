import CryptoKit
import Foundation

/// 卒論第4章の計測結果の書き出し（CSV・npy・wav・crumb）と、行に添える端末の情報
///
/// CSV は 1 回の実行を 1 行として追記し、書くたびにファイルを閉じる。
/// 途中でプロセスが落ちても、それまでの行は残る。
final class ThesisRecorder {

    let config: ThesisBenchConfig
    let directory: URL

    init(config: ThesisBenchConfig) throws {
        self.config = config
        self.directory = config.sessionDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: - 端末の情報

    /// utsname の machine（例: `iPhone14,5`）。シミュレータでは `arm64` になる
    static var deviceModel: String {
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        let bytes = mirror.children.compactMap({ $0.value as? Int8 }).prefix(while: { $0 != 0 })
        return String(decoding: bytes.map({ UInt8(bitPattern: $0) }), as: UTF8.self)
    }

    /// iOS のバージョン（例: `27.0.1`）
    static var osVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// ビルド番号を含む表記（例: `Version 27.0 (Build 24A...)`）
    static var osVersionString: String {
        ProcessInfo.processInfo.operatingSystemVersionString
    }

    static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }

    /// Float32 の生バイト列（little-endian）の SHA-256
    static func sha256(_ values: [Float]) -> String {
        var hasher = SHA256()
        values.withUnsafeBufferPointer { buf in
            if let base = buf.baseAddress {
                hasher.update(bufferPointer: UnsafeRawBufferPointer(start: base, count: buf.count * MemoryLayout<Float>.size))
            }
        }
        return hasher.finalize().map({ String(format: "%02x", $0) }).joined()
    }

    // MARK: - ログ

    /// コンソールに 1 行出す。devicectl --console で PC に届くよう、すぐ flush する
    static func log(_ message: String) {
        print(message)
        fflush(stdout)
    }

    // MARK: - CSV

    /// `name` の CSV に 1 行追記する。ファイルが無ければ見出しを先に書く
    func appendRow(csvName: String, header: [String], values: [String: String]) {
        let url = directory.appendingPathComponent(csvName)
        var text = ""
        if !FileManager.default.fileExists(atPath: url.path) {
            text += header.joined(separator: ",") + "\n"
        }
        text += header.map({ Self.csvField(values[$0] ?? "") }).joined(separator: ",") + "\n"
        let data = Data(text.utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
            try? handle.synchronize()
        } else {
            try? data.write(to: url)
        }
    }

    /// カンマ・改行・引用符を含む値は引用符で囲む
    static func csvField(_ value: String) -> String {
        guard value.contains(",") || value.contains("\n") || value.contains("\"") else { return value }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    /// すべての行に共通の列
    func commonValues() -> [String: String] {
        [
            "session_id": config.sessionID,
            "launch_id": config.launchID,
            "mode": config.mode.rawValue,
            "device_model": Self.deviceModel,
            "os_version": Self.osVersion,
            "os_version_string": Self.osVersionString,
            "precision": config.precision?.rawValue ?? "",
            "compute_units": config.computeUnit?.rawValue ?? "",
            "shape_mode": ThesisBenchConfig.shapeMode.rawValue,
        ]
    }

    static let commonHeader = [
        "session_id", "launch_id", "mode", "device_model", "os_version", "os_version_string",
        "precision", "compute_units", "shape_mode",
    ]

    // MARK: - crumb

    /// 今どこを実行しているかを残す。プロセスが落ちたとき、PC 側がこれを読んで記録する
    func writeCrumb(_ text: String) {
        let url = directory.appendingPathComponent("crumb_\(config.launchID).txt")
        try? (text + "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    func removeCrumb() {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent("crumb_\(config.launchID).txt"))
    }

    // MARK: - npy / wav

    /// `runs/<launch>/` の下のパス（CSV にはセッションのディレクトリからの相対パスを書く）
    func runFileURL(_ fileName: String) throws -> (url: URL, relativePath: String) {
        let relative = "runs/\(config.launchID)/\(fileName)"
        let url = directory.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        return (url, relative)
    }

    /// [rows × cols] の float32 を npy で保存する
    func writeNpy(_ values: [Float], rows: Int, cols: Int, fileName: String) throws {
        guard rows > 0, cols > 0 else { return }
        let (url, _) = try runFileURL(fileName)
        try NpyWriter.writeFloat32(values, rows: rows, cols: cols, to: url)
    }

    func writeWav(_ waveform: [Float], sampleRate: Double, fileName: String) throws {
        guard !waveform.isEmpty else { return }
        let (url, _) = try runFileURL(fileName)
        try WavWriter.writeInt16(waveform: waveform, sampleRate: sampleRate, to: url)
    }
}
