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
