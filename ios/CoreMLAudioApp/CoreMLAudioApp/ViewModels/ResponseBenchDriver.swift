import Foundation
import UIKit

/// 通常操作の応答時間の計測モードで、画面の「合成」と同じ `SynthesisViewModel.runSynthesis()` を繰り返し呼ぶ
///
/// - 合成・保存・再生は通常の操作と同じ処理経路を通る（画面の表示と進捗の更新も行われる）
/// - 入力は同梱した卒論の評価入力（`thesis_src_<name>.wav`）を、録音した音声と同じ `.recording(URL)` として選ぶ
/// - 計測前にモデルを読み込んでおき、各回の `loadModels` は読み込み済みの確認だけで戻る
/// - 1 回の合成ごとに `response_runs.csv` に 1 行を書く（書くたびにファイルを閉じる）。
///   CSV の書き出しと SHA-256 の計算は、再生が終わって計測区間を抜けた後に行う
@MainActor
enum ResponseBenchDriver {

    static let csvName = "response_runs.csv"

    static let header = [
        "session_id", "launch_id", "bench_version", "device_model", "os_version", "os_version_string",
        "build_configuration", "precision", "compute_setting", "encdec_compute_units", "hifigan_compute_units",
        "shape_mode", "input_name", "input_order", "T_public", "T_prime_app", "audio_sec",
        "run_index", "phase", "first_after_load", "status", "error",
        "thermal_before", "thermal_after", "started_at", "gap_ms",
        "entry_to_A_ms", "A_to_B_ms", "B_to_C_ms", "C_to_D_ms", "A_to_D_ms",
        "D_to_E_ms", "E_to_F_ms", "F_to_G_ms", "A_to_F_ms", "A_to_G_ms", "G_to_end_ms",
        "encoder_ms", "decoder_total_ms", "hifigan_ms", "synth_ms", "rtf", "output_sha256",
        "optimizations", "progress_updates", "progress_decreases", "progress_max_gap_ms",
        "main_stall_max_ms", "main_stall_over16_count", "main_stall_over33_count", "main_stall_over16_total_ms",
    ]

    /// 計測を最後まで行い、プロセスの終了コードを返す
    static func run(viewModel: SynthesisViewModel) async -> Int32 {
        log("RESPONSE_BENCH_START")
        UIApplication.shared.isIdleTimerDisabled = true
        do {
            let config = try ResponseBenchConfig.fromEnvironment()
            try FileManager.default.createDirectory(at: config.sessionDirectory, withIntermediateDirectories: true)
            try await measure(config: config, viewModel: viewModel)
            log("RESPONSE_BENCH_DONE status=ok")
            return 0
        } catch {
            log("RESPONSE_BENCH_DONE status=failed error=\(error.localizedDescription)")
            return 1
        }
    }

    private static func measure(config: ResponseBenchConfig, viewModel: SynthesisViewModel) async throws {
        viewModel.selectedPrecision = config.precision
        viewModel.selectedComputeUnit = config.setting
        viewModel.selectedShapeMode = ThesisBenchConfig.shapeMode
        viewModel.synthesisOptimizations = config.optimizations

        // モデルを読み込んでおく（計測区間の外）
        let loadStart = DispatchTime.now().uptimeNanoseconds
        try viewModel.loadModelsForCurrentSelection()
        let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000

        let infos = try ThesisPipeline.inputInfos()
        let names = config.inputNames ?? infos.map({ $0.input_name })
        let publicT = Dictionary(uniqueKeysWithValues: infos.map({ ($0.input_name, $0.T) }))

        writeMeta(config: config, inputOrder: names, loadMs: loadMs)
        log(String(format: "RESPONSE_BENCH_LOADED load_ms=%.1f thermal=%@", loadMs, ThesisRecorder.thermalState))

        let probe = ResponseTimeProbe()
        viewModel.responseProbe = probe
        defer { viewModel.responseProbe = nil }

        var firstAfterLoad = true
        for (order, name) in names.enumerated() {
            let url = try ThesisPipeline.inputURL(name: name)
            viewModel.audioSource = .recording(url)
            let total = config.warmupCount + config.steadyCount
            for runIndex in 0..<total {
                let phase = runIndex < config.warmupCount ? "warmup" : "steady"
                let thermalBefore = ThesisRecorder.thermalState
                let startedAt = ThesisRecorder.timestamp()

                await viewModel.runSynthesis()

                let thermalAfter = ThesisRecorder.thermalState
                let row = makeRow(
                    config: config, viewModel: viewModel, probe: probe,
                    name: name, order: order + 1, publicT: publicT[name],
                    runIndex: runIndex, phase: phase, firstAfterLoad: firstAfterLoad,
                    thermalBefore: thermalBefore, thermalAfter: thermalAfter, startedAt: startedAt
                )
                appendRow(directory: config.sessionDirectory, values: row)
                log("RESPONSE_RUN input=\(name) run=\(runIndex) phase=\(phase) status=\(row["status"] ?? "")"
                    + " A_to_B=\(row["A_to_B_ms"] ?? "") A_to_F=\(row["A_to_F_ms"] ?? "") thermal=\(thermalAfter)")
                firstAfterLoad = false

                if config.gapMs > 0 {
                    try await Task.sleep(nanoseconds: UInt64(config.gapMs) * 1_000_000)
                }
            }
        }
    }

    private static func makeRow(
        config: ResponseBenchConfig, viewModel: SynthesisViewModel, probe: ResponseTimeProbe,
        name: String, order: Int, publicT: Int?, runIndex: Int, phase: String, firstAfterLoad: Bool,
        thermalBefore: String, thermalAfter: String, startedAt: String
    ) -> [String: String] {
        func ms(_ from: ResponseTimeProbe.Point, _ to: ResponseTimeProbe.Point) -> String {
            probe.ms(from, to).map({ String(format: "%.3f", $0) }) ?? ""
        }
        let result = viewModel.synthesisResult
        // 再生開始の API から戻るまで（G）が打たれ、エラーが無ければ成功
        let ok = viewModel.errorMessage == nil && result != nil && probe.has(.g)
        var values: [String: String] = [
            "session_id": config.sessionID,
            "launch_id": config.launchID,
            "bench_version": ResponseBenchConfig.benchVersion,
            "device_model": ThesisRecorder.deviceModel,
            "os_version": ThesisRecorder.osVersion,
            "os_version_string": ThesisRecorder.osVersionString,
            "build_configuration": ResponseBenchConfig.buildConfiguration,
            "precision": config.precision.rawValue,
            "compute_setting": config.setting.rawValue,
            "encdec_compute_units": config.setting.encoderDecoderUnit.rawValue,
            "hifigan_compute_units": config.setting.hifiganUnit.rawValue,
            "shape_mode": viewModel.selectedShapeMode.rawValue,
            "input_name": name,
            "input_order": String(order),
            "T_public": publicT.map({ String($0) }) ?? "",
            "run_index": String(runIndex),
            "phase": phase,
            "first_after_load": firstAfterLoad ? "1" : "0",
            "status": ok ? "success" : "failed",
            "error": viewModel.errorMessage ?? "",
            "thermal_before": thermalBefore,
            "thermal_after": thermalAfter,
            "started_at": startedAt,
            "gap_ms": String(config.gapMs),
            "entry_to_A_ms": ms(.entry, .a),
            "A_to_B_ms": ms(.a, .b),
            "B_to_C_ms": ms(.b, .c),
            "C_to_D_ms": ms(.c, .d),
            "A_to_D_ms": ms(.a, .d),
            "D_to_E_ms": ms(.d, .e),
            "E_to_F_ms": ms(.e, .f),
            "F_to_G_ms": ms(.f, .g),
            "A_to_F_ms": ms(.a, .f),
            "A_to_G_ms": ms(.a, .g),
            "G_to_end_ms": ms(.g, .end),
            "optimizations": config.optimizations.label,
            "progress_updates": String(probe.progressUpdates),
            "progress_decreases": String(probe.progressDecreases),
            "progress_max_gap_ms": String(format: "%.3f", probe.progressMaxGapMs),
        ]
        if let stall = probe.stall {
            values["main_stall_max_ms"] = String(format: "%.3f", stall.maxGapMs)
            values["main_stall_over16_count"] = String(stall.over16Count)
            values["main_stall_over33_count"] = String(stall.over33Count)
            values["main_stall_over16_total_ms"] = String(format: "%.3f", stall.over16TotalMs)
        }
        if let result {
            let audioSec = Double(result.outputWaveform.count) / result.sampleRate
            values["T_prime_app"] = String(result.outputFrameCount)
            values["audio_sec"] = String(format: "%.6f", audioSec)
            values["encoder_ms"] = String(format: "%.3f", result.timing.encoderMs)
            values["decoder_total_ms"] = String(format: "%.3f", result.timing.decoderTotalMs)
            values["hifigan_ms"] = String(format: "%.3f", result.timing.hifiganMs)
            let synthMs = result.timing.encoderMs + result.timing.decoderTotalMs + result.timing.hifiganMs
            values["synth_ms"] = String(format: "%.3f", synthMs)
            values["rtf"] = audioSec > 0 ? String(format: "%.6f", synthMs / 1000 / audioSec) : ""
            values["output_sha256"] = ThesisRecorder.sha256(result.outputWaveform)
        }
        return values
    }

    // MARK: - 書き出し

    private static func appendRow(directory: URL, values: [String: String]) {
        let url = directory.appendingPathComponent(csvName)
        var text = ""
        if !FileManager.default.fileExists(atPath: url.path) {
            text += header.joined(separator: ",") + "\n"
        }
        text += header.map({ ThesisRecorder.csvField(values[$0] ?? "") }).joined(separator: ",") + "\n"
        let data = Data(text.utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url)
        }
    }

    private static func writeMeta(config: ResponseBenchConfig, inputOrder: [String], loadMs: Double) {
        let meta: [String: Any] = [
            "session_id": config.sessionID,
            "launch_id": config.launchID,
            "bench_version": ResponseBenchConfig.benchVersion,
            "device_model": ThesisRecorder.deviceModel,
            "os_version_string": ThesisRecorder.osVersionString,
            "build_configuration": ResponseBenchConfig.buildConfiguration,
            "precision": config.precision.rawValue,
            "compute_setting": config.setting.rawValue,
            "shape_mode": ThesisBenchConfig.shapeMode.rawValue,
            "warmup_per_input": config.warmupCount,
            "steady_per_input": config.steadyCount,
            "input_order": inputOrder,
            "gap_ms": config.gapMs,
            "model_load_ms_before_measure": loadMs,
            "thermal_at_start": ThesisRecorder.thermalState,
            "started_at": ThesisRecorder.timestamp(),
            "clock": "DispatchTime.uptimeNanoseconds (mach_absolute_time)",
            "points": [
                "entry": "runSynthesis() の先頭",
                "A": "loadModels から戻った直後（合成処理の開始）",
                "B": "synthesize() から戻った直後",
                "C": "saveMelArtifacts・saveTimingArtifact の完了",
                "D": "audioPlayer.play() を呼ぶ直前",
                "E": "AudioPlayer.play() 内で WAV の保存が終わった直後",
                "F": "AVAudioPlayer.play()（再生開始の API）を呼ぶ直前",
                "G": "AVAudioPlayer.play() から戻った直後",
                "end": "再生終了の通知を MainActor で受けた時刻",
            ],
        ]
        let url = config.sessionDirectory.appendingPathComponent("session_meta_\(config.launchID).json")
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url)
        }
    }

    private static func log(_ message: String) {
        ThesisRecorder.log(message)
    }
}
