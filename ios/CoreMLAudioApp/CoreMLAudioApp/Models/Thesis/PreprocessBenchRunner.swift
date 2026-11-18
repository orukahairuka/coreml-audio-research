import Darwin
import Foundation

/// 修正前と修正後の前処理を実機で比べる計測モード（`CMLA_THESIS_BENCH=preprocess`）
///
/// モデルは読み込まない。入力ごとに次の順で実行する。
/// 1. 時間: ウォームアップ＋定常時の回数だけ、新旧を回ごとに順番を入れ替えて実行する
/// 2. メモリ: 物理フットプリントを 1 ms ごとに読みながら 3 回実行する（時間の計測とは分ける）
/// 3. 保存: 各段階の出力を npy に保存する
final class PreprocessBenchRunner {

    enum Implementation: String, CaseIterable {
        case legacy, current
    }

    static let csvName = "preprocess_runs.csv"
    static let memoryRuns = 3

    static let header = ThesisRecorder.commonHeader + [
        "input_name", "implementation", "run_index", "phase", "status", "error",
        "thermal_before", "thermal_after", "started_at",
        "source_rate", "resampling_method", "sample_count", "T", "T_prime",
        "read_ms", "resample_ms", "load_ms", "mel_ms", "total_ms",
        "footprint_before_bytes", "footprint_peak_bytes", "footprint_delta_bytes", "mel_sha256",
    ] + ThesisRecorder.versionHeader

    private let config: ThesisBenchConfig
    private let recorder: ThesisRecorder

    init(config: ThesisBenchConfig, recorder: ThesisRecorder) {
        self.config = config
        self.recorder = recorder
    }

    func run() throws {
        let names = try config.inputNames ?? ThesisPipeline.inputInfos().map({ $0.input_name })
        for name in names {
            let url = try ThesisPipeline.inputURL(name: name)
            for runIndex in 0..<(config.warmupCount + config.steadyCount) {
                let phase = runIndex < config.warmupCount ? "warmup" : "steady"
                // 新旧の実行順による偏りを避けるため、回ごとに順番を入れ替える
                let order = runIndex % 2 == 0 ? Implementation.allCases : Implementation.allCases.reversed()
                for implementation in order {
                    measure(implementation, name: name, url: url, runIndex: runIndex, phase: phase, sampleMemory: false)
                }
            }
            for runIndex in 0..<Self.memoryRuns {
                for implementation in Implementation.allCases {
                    measure(implementation, name: name, url: url, runIndex: runIndex, phase: "memory", sampleMemory: true)
                }
            }
            try saveStages(name: name, url: url)
        }
    }

    private func measure(
        _ implementation: Implementation, name: String, url: URL, runIndex: Int, phase: String, sampleMemory: Bool
    ) {
        recorder.writeCrumb("mode=preprocess input=\(name) impl=\(implementation.rawValue) run_index=\(runIndex) phase=\(phase)")
        var values = recorder.commonValues()
        values["input_name"] = name
        values["implementation"] = implementation.rawValue
        values["run_index"] = String(runIndex)
        values["phase"] = phase
        values["thermal_before"] = ThesisRecorder.thermalState
        values["started_at"] = ThesisRecorder.timestamp()

        let sampler = sampleMemory ? FootprintSampler() : nil
        sampler?.start()
        do {
            try autoreleasepool(invoking: {
                let timing = try Self.timedRun(implementation, url: url)
                values.merge(timing, uniquingKeysWith: { _, new in new })
            })
            values["status"] = "success"
        } catch {
            values["status"] = "failed"
            values["error"] = error.localizedDescription
        }
        if let sampler {
            let result = sampler.stop()
            values["footprint_before_bytes"] = String(result.before)
            values["footprint_peak_bytes"] = String(result.peak)
            values["footprint_delta_bytes"] = String(result.peak >= result.before ? result.peak - result.before : 0)
        }
        values["thermal_after"] = ThesisRecorder.thermalState
        recorder.appendRow(csvName: Self.csvName, header: Self.header, values: values)
        ThesisRecorder.log("[thesis] preprocess in=\(name) impl=\(implementation.rawValue) r=\(runIndex) \(phase) "
            + "total_ms=\(values["total_ms"] ?? "-") T=\(values["T"] ?? "-") status=\(values["status"] ?? "")")
    }

    /// 読み込みからメルの正規化までを実行し、CSV の列の値を返す
    private static func timedRun(_ implementation: Implementation, url: URL) throws -> [String: String] {
        var values = [String: String]()
        let mel: [Float]
        let frameCount: Int
        let t0 = CFAbsoluteTimeGetCurrent()
        let tLoaded: CFAbsoluteTime
        switch implementation {
        case .legacy:
            let samples = try LegacyAudioFeatureExtractor.loadAudio(from: url)
            tLoaded = CFAbsoluteTimeGetCurrent()
            (mel, frameCount) = LegacyAudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples)
            values["sample_count"] = String(samples.count)
        case .current:
            let source = try AudioFeatureExtractor.readMonoSource(from: url)
            let tRead = CFAbsoluteTimeGetCurrent()
            let samples = try AudioFeatureExtractor.resample(source.samples, sourceRate: source.sampleRate)
            tLoaded = CFAbsoluteTimeGetCurrent()
            (mel, frameCount) = AudioFeatureExtractor.extractMelSpectrogram(fromSamples: samples)
            values["read_ms"] = ThesisBenchRunner.ms((tRead - t0) * 1000)
            values["resample_ms"] = ThesisBenchRunner.ms((tLoaded - tRead) * 1000)
            values["source_rate"] = String(source.sampleRate)
            values["resampling_method"] = AudioFeatureExtractor.resamplingMethod(forSourceRate: source.sampleRate).rawValue
            values["sample_count"] = String(samples.count)
        }
        let tEnd = CFAbsoluteTimeGetCurrent()
        values["load_ms"] = ThesisBenchRunner.ms((tLoaded - t0) * 1000)
        values["mel_ms"] = ThesisBenchRunner.ms((tEnd - tLoaded) * 1000)
        values["total_ms"] = ThesisBenchRunner.ms((tEnd - t0) * 1000)
        values["T"] = String(frameCount)
        values["T_prime"] = String(effectiveFrames(frameCount))
        values["mel_sha256"] = ThesisRecorder.sha256(mel)
        return values
    }

    /// 合成に使うフレーム数（`ThesisPipeline.synthesize` と同じ計算）
    private static func effectiveFrames(_ frameCount: Int) -> Int {
        switch ThesisBenchConfig.shapeMode.transformerInputPolicy {
        case .fixed(let targetT): return min(frameCount, targetT)
        case .dynamic(let maxT): return min(frameCount, maxT)
        }
    }

    private func saveStages(name: String, url: URL) throws {
        for implementation in Implementation.allCases {
            let stages = implementation == .legacy
                ? try PreprocessStages.legacy(url: url) : try PreprocessStages.current(url: url)
            for array in stages.arrays {
                try recorder.writeNpy(
                    array.values, rows: array.rows, cols: array.cols,
                    fileName: "preprocess_\(implementation.rawValue)_in\(name)_\(array.name).npy"
                )
            }
        }
    }
}

/// 物理フットプリントを別スレッドで 1 ms ごとに読み、最大値を記録する
private final class FootprintSampler {
    private let lock = NSLock()
    private var running = false
    private var peak: UInt64 = 0
    private var before: UInt64 = 0
    private let finished = DispatchSemaphore(value: 0)

    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info, { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count), { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            })
        })
        return status == KERN_SUCCESS ? info.phys_footprint : 0
    }

    func start() {
        before = Self.footprint()
        peak = before
        running = true
        Thread.detachNewThread({ [self] in
            while true {
                lock.lock()
                let keepRunning = running
                peak = max(peak, Self.footprint())
                lock.unlock()
                if !keepRunning { break }
                Thread.sleep(forTimeInterval: 0.001)
            }
            finished.signal()
        })
    }

    func stop() -> (before: UInt64, peak: UInt64) {
        lock.lock()
        running = false
        lock.unlock()
        finished.wait()
        return (before, peak)
    }
}
