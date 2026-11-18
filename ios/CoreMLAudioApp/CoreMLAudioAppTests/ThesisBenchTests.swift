import Testing
@testable import CoreMLAudioApp
import Foundation

/// 卒論第4章の計測モード（設定の読み込み・CSV・ハッシュ）のテスト
struct ThesisBenchTests {

    // MARK: - ThesisBenchConfig

    @Test func configParsesInferEnvironment() throws {
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "infer",
            "CMLA_THESIS_SESSION": "s1",
            "CMLA_THESIS_LAUNCH": "L001",
            "CMLA_THESIS_PRECISION": "Float16",
            "CMLA_THESIS_CU": "cpuAndNE",
            "CMLA_THESIS_STEADY": "3",
            "CMLA_THESIS_INPUTS": "1,11",
        ])
        #expect(config.mode == .infer)
        #expect(config.precision == .float16)
        #expect(config.computeUnit == .cpuAndNE)
        #expect(config.warmupCount == 1)
        #expect(config.steadyCount == 3)
        #expect(config.reloadCount == 5)
        #expect(config.inputNames == ["1", "11"])
        #expect(config.conditionLabel == "Float16_cpuAndNE")
        #expect(config.sessionDirectory.path.hasSuffix("Result/thesis/s1"))
    }

    @Test func configWithoutNewVariablesKeepsLegacyCondition() throws {
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE",
        ])
        #expect(config.isMixedComputeUnits == false)
        #expect(config.hifiganComputeUnit == .cpuAndNE)
        #expect(config.computeUnitsLabel == "cpuAndNE")
        #expect(config.conditionLabel == "Float16_cpuAndNE")
        #expect(config.profileEnabled == false)
    }

    @Test func configParsesMixedComputeUnits() throws {
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE",
            "CMLA_THESIS_CU_HIFI": "cpuOnly", "CMLA_THESIS_PROFILE": "1",
        ])
        #expect(config.isMixedComputeUnits)
        #expect(config.computeUnit == .cpuAndNE)
        #expect(config.hifiganComputeUnit == .cpuOnly)
        #expect(config.computeUnitsLabel == "cpuAndNE+hifi-cpuOnly")
        #expect(config.conditionLabel == "Float16_cpuAndNE+hifi-cpuOnly")
        #expect(config.profileEnabled)
    }

    @Test func configTreatsSameHifiganUnitAsLegacy() throws {
        // 条件 A（3 モデルとも cpuAndNE）を明示しても、従来の Float16_cpuAndNE と同じ行になる
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE",
            "CMLA_THESIS_CU_HIFI": "cpuAndNE",
        ])
        #expect(config.isMixedComputeUnits == false)
        #expect(config.conditionLabel == "Float16_cpuAndNE")
    }

    @Test func configRejectsUnknownNewValues() {
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment([
                "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE",
                "CMLA_THESIS_CU_HIFI": "gpu",
            ])
        }
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment([
                "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE",
                "CMLA_THESIS_PROFILE": "yes",
            ])
        }
    }

    // MARK: - StageProfiler

    @Test func profilerLapsPartitionTheSpan() {
        let profiler = StageProfiler()
        profiler.start()
        for _ in 0..<3 {
            profiler.lap(ProfileStage.decInputFill)
            profiler.lap(ProfileStage.decPredict)
        }
        profiler.lap(ProfileStage.deemph)
        #expect(profiler.counts[ProfileStage.decInputFill] == 3)
        #expect(profiler.counts[ProfileStage.deemph] == 1)
        #expect(abs(profiler.sumMs - profiler.spanMs) < 1e-6)
    }

    @Test func stagesHeaderHasEveryStageOnce() {
        let header = ThesisBenchRunner.stagesHeader
        #expect(Set(header).count == header.count)
        for stage in ProfileStage.all {
            #expect(header.contains("\(stage)_ms"))
        }
    }

    @Test func runHeaderKeepsLegacyColumnsFirst() {
        // 追加した列は末尾だけ。既存の列の並びは変えない
        let legacyTail = Array(ThesisBenchRunner.runHeader.prefix(ThesisBenchRunner.runHeader.count - ThesisRecorder.versionHeader.count))
        #expect(legacyTail.last == "file_prefix")
        #expect(Array(ThesisBenchRunner.runHeader.suffix(ThesisRecorder.versionHeader.count)) == ThesisRecorder.versionHeader)
    }

    @Test func appendRowAvoidsMixingDifferentHeaders() throws {
        let env = [
            "CMLA_THESIS_BENCH": "load", "CMLA_THESIS_SESSION": "unittest_\(UUID().uuidString)",
            "CMLA_THESIS_PRECISION": "Float32", "CMLA_THESIS_CU": "cpuOnly",
        ]
        let recorder = try ThesisRecorder(config: try ThesisBenchConfig.fromEnvironment(env))
        defer { try? FileManager.default.removeItem(at: recorder.directory) }
        recorder.appendRow(csvName: "t.csv", header: ["a", "b"], values: ["a": "1"])
        recorder.appendRow(csvName: "t.csv", header: ["a", "b", "c"], values: ["c": "3"])
        let old = try String(contentsOf: recorder.directory.appendingPathComponent("t.csv"), encoding: .utf8)
        #expect(old == "a,b\n1,\n")
        let new = try String(contentsOf: recorder.directory.appendingPathComponent("t_\(ThesisBenchConfig.benchVersion).csv"), encoding: .utf8)
        #expect(new == "a,b,c\n,,3\n")
    }

    @Test func configStepStatsDefaultsOffAndCanBeEnabled() throws {
        let base = ["CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE"]
        #expect(try ThesisBenchConfig.fromEnvironment(base).decoderStepStats == false)
        var on = base
        on["CMLA_THESIS_STEP_STATS"] = "1"
        #expect(try ThesisBenchConfig.fromEnvironment(on).decoderStepStats)
        var bad = base
        bad["CMLA_THESIS_STEP_STATS"] = "true"
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment(bad)
        }
    }

    @Test func configDecoderInputWriteDefaultsToBulk() throws {
        let base = ["CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuAndNE"]
        #expect(try ThesisBenchConfig.fromEnvironment(base).decoderInputWrite == .bulk)
        var legacy = base
        legacy["CMLA_THESIS_DECODER_INPUT_WRITE"] = "element"
        #expect(try ThesisBenchConfig.fromEnvironment(legacy).decoderInputWrite == .element)
        var bad = base
        bad["CMLA_THESIS_DECODER_INPUT_WRITE"] = "memcpy"
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment(bad)
        }
    }

    @Test func versionHeaderRecordsStepStats() {
        #expect(ThesisRecorder.versionHeader.contains("decoder_step_stats"))
        #expect(ThesisRecorder.versionHeader.contains("decoder_input_write"))
        #expect(ThesisBenchRunner.stagesHeader.contains("dec_input_alloc_ms"))
    }

    @Test func preprocessModeIsParsedAndRecordsBothImplementations() throws {
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "preprocess", "CMLA_THESIS_PRECISION": "Float16", "CMLA_THESIS_CU": "cpuOnly",
        ])
        #expect(config.mode == .preprocess)
        #expect(PreprocessBenchRunner.Implementation.allCases.map({ $0.rawValue }) == ["legacy", "current"])
        #expect(PreprocessBenchRunner.header.contains("resample_ms"))
        #expect(PreprocessBenchRunner.header.contains("footprint_peak_bytes"))
    }

    @Test func planFailureNamesEveryFailedModel() {
        let error = ThesisBenchRunner.RunError.planFailed(models: ["encoder", "hifigan"])
        let message = error.localizedDescription
        #expect(message.contains("encoder"))
        #expect(message.contains("hifigan"))
    }

    @Test func configRequiresConditionExceptThermal() throws {
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment(["CMLA_THESIS_BENCH": "load", "CMLA_THESIS_CU": "all"])
        }
        let thermal = try ThesisBenchConfig.fromEnvironment(["CMLA_THESIS_BENCH": "thermal"])
        #expect(thermal.mode == .thermal)
    }

    @Test func configRejectsUnknownValues() {
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment(["CMLA_THESIS_BENCH": "bench"])
        }
        #expect(throws: ThesisBenchConfig.ConfigError.self) {
            _ = try ThesisBenchConfig.fromEnvironment([
                "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "float16", "CMLA_THESIS_CU": "all",
            ])
        }
    }

    // MARK: - ThesisRecorder

    @Test func csvFieldQuotesOnlyWhenNeeded() {
        #expect(ThesisRecorder.csvField("abc") == "abc")
        #expect(ThesisRecorder.csvField("a,b") == "\"a,b\"")
        #expect(ThesisRecorder.csvField("say \"x\"") == "\"say \"\"x\"\"\"")
    }

    @Test func sha256MatchesKnownFloat32Bytes() {
        // [0.0] の float32 バイト列 00 00 00 00 の SHA-256
        #expect(ThesisRecorder.sha256([0])
            == "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119")
        #expect(ThesisRecorder.sha256([1, 2]) != ThesisRecorder.sha256([2, 1]))
    }

    @Test func appendRowWritesHeaderOnceAndKeepsColumnOrder() throws {
        let env = [
            "CMLA_THESIS_BENCH": "load", "CMLA_THESIS_SESSION": "unittest_\(UUID().uuidString)",
            "CMLA_THESIS_PRECISION": "Float32", "CMLA_THESIS_CU": "cpuOnly",
        ]
        let recorder = try ThesisRecorder(config: try ThesisBenchConfig.fromEnvironment(env))
        defer { try? FileManager.default.removeItem(at: recorder.directory) }
        recorder.appendRow(csvName: "t.csv", header: ["a", "b", "c"], values: ["a": "1", "c": "x,y"])
        recorder.appendRow(csvName: "t.csv", header: ["a", "b", "c"], values: ["b": "2"])
        let text = try String(contentsOf: recorder.directory.appendingPathComponent("t.csv"), encoding: .utf8)
        #expect(text == "a,b,c\n1,,\"x,y\"\n,2,\n")
    }

    @Test func thermalStateIsKnownLabel() {
        #expect(["nominal", "fair", "serious", "critical"].contains(ThesisRecorder.thermalState))
        #expect(!ThesisRecorder.deviceModel.isEmpty)
    }
}
