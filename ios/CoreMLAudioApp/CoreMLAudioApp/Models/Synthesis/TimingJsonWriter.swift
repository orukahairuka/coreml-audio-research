import Foundation

/// `TimingInfo` を JSON ファイルとして `Documents/Result/timing/` に書き出す
///
/// ファイル名は `timing_<Precision>_<ComputeUnit>.json`。固定名で上書き保存する想定。
enum TimingJsonWriter {

    enum WriteError: LocalizedError {
        case directoryCreationFailed(URL, underlying: Error)

        var errorDescription: String? {
            switch self {
            case .directoryCreationFailed(let url, let underlying):
                return "保存先の作成に失敗: \(url.path) (\(underlying.localizedDescription))"
            }
        }
    }

    /// 書き出し用に `TimingInfo` を JSON 化するための DTO
    private struct TimingRecord: Codable {
        let precision: String
        let computeUnit: String
        let encoderMs: Double
        let decoderTotalMs: Double
        let decoderStepCount: Int
        let decoderAvgPerStepMs: Double
        let hifiganMs: Double
        let totalPredictMs: Double
        let outputDurationMs: Double
        let realTimeFactor: Double
        let modelSizeBytes: Int64
    }

    /// 出力ディレクトリ (`Documents/Result/timing/`)。無ければ作る
    static func resultDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("Result", isDirectory: true)
            .appendingPathComponent("timing", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            } catch {
                throw WriteError.directoryCreationFailed(dir, underlying: error)
            }
        }
        return dir
    }

    /// `TimingInfo` を JSON として書き出す
    ///
    /// - Parameters:
    ///   - timing: 計測結果
    ///   - precision: 精度設定 (rawValue を JSON に保存)
    ///   - computeUnit: 計算デバイス設定 (rawValue を JSON に保存)
    static func write(
        timing: TimingInfo,
        precision: ModelPrecision,
        computeUnit: ComputeUnitOption
    ) throws {
        let dir = try resultDirectory()
        let fileName = "timing_\(precision.rawValue)_\(computeUnit.rawValue).json"
        let url = dir.appendingPathComponent(fileName)

        let record = TimingRecord(
            precision: precision.rawValue,
            computeUnit: computeUnit.rawValue,
            encoderMs: timing.encoderMs,
            decoderTotalMs: timing.decoderTotalMs,
            decoderStepCount: timing.decoderStepCount,
            decoderAvgPerStepMs: timing.decoderAvgPerStepMs,
            hifiganMs: timing.hifiganMs,
            totalPredictMs: timing.totalPredictMs,
            outputDurationMs: timing.outputDurationMs,
            realTimeFactor: timing.realTimeFactor,
            modelSizeBytes: timing.modelSizeBytes
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(record)
        try data.write(to: url)
    }
}
