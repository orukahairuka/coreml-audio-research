import Foundation
import Testing
@testable import CoreMLAudioApp

/// 正式評価の一括合成モード（設定・入力の名前・再開のための CSV）のテスト
struct FormalEvalTests {

    // MARK: - 設定

    @Test func configParsesEnvironment() throws {
        let config = try FormalEvalConfig.fromEnvironment([
            "CMLA_FORMAL_EVAL": "1", "CMLA_FORMAL_SESSION": "smoke_20261122",
            "CMLA_FORMAL_PRECISION": "Float16", "CMLA_FORMAL_CU": "cpuAndGPU",
            "CMLA_FORMAL_SPEAKERS": "m-01, f-01", "CMLA_FORMAL_LIMIT": "2",
        ])
        #expect(config.precision == .float16)
        #expect(config.computeUnit == .cpuAndGPU)
        #expect(config.speakers == ["m-01", "f-01"])
        #expect(config.limitPerSpeaker == 2)
        #expect(config.names == nil)
        #expect(config.sessionDirectory.path.hasSuffix("Result/formal_eval/smoke_20261122"))
        // 正式評価の条件は固定
        #expect(FormalEvalConfig.shapeMode == .range1)
        #expect(FormalEvalConfig.waveformPreprocess == .fullInput)
    }

    /// 実験用の設定は、指定しなければ使われない（従来と同じ動作）
    @Test func experimentalOptionsAreOffByDefault() throws {
        let base: [String: String] = ["CMLA_FORMAL_SESSION": "s", "CMLA_FORMAL_PRECISION": "Float32", "CMLA_FORMAL_CU": "cpuOnly"]
        let plain = try FormalEvalConfig.fromEnvironment(base)
        #expect(plain.usesExperimentalOptions == false)
        #expect(plain.computeUnit(for: "decoder") == .cpuOnly)
        #expect(plain.padDecoderInput == false)

        var env = base
        env["CMLA_FORMAL_CU_ENC"] = "cpuAndGPU"
        env["CMLA_FORMAL_CU_DEC"] = "cpuAndGPU"
        env["CMLA_FORMAL_DEC_INPUT"] = "padT"
        env["CMLA_FORMAL_RELOAD_MB"] = "1000"
        let config = try FormalEvalConfig.fromEnvironment(env)
        #expect(config.usesExperimentalOptions)
        #expect(config.computeUnit(for: "encoder") == .cpuAndGPU)
        #expect(config.computeUnit(for: "decoder") == .cpuAndGPU)
        #expect(config.computeUnit(for: "hifigan") == .cpuOnly)
        #expect(config.padDecoderInput)
        #expect(config.reloadThresholdMB == 1000)

        env["CMLA_FORMAL_DEC_INPUT"] = "pad"
        #expect(throws: FormalEvalConfig.ConfigError.self) { _ = try FormalEvalConfig.fromEnvironment(env) }
    }

    @Test func configRejectsMissingAndUnsafeValues() {
        #expect(throws: FormalEvalConfig.ConfigError.self) {
            _ = try FormalEvalConfig.fromEnvironment(["CMLA_FORMAL_PRECISION": "Float32", "CMLA_FORMAL_CU": "cpuOnly"])
        }
        #expect(throws: FormalEvalConfig.ConfigError.self) {
            _ = try FormalEvalConfig.fromEnvironment([
                "CMLA_FORMAL_SESSION": "../x", "CMLA_FORMAL_PRECISION": "Float32", "CMLA_FORMAL_CU": "cpuOnly",
            ])
        }
        #expect(throws: FormalEvalConfig.ConfigError.self) {
            _ = try FormalEvalConfig.fromEnvironment([
                "CMLA_FORMAL_SESSION": "s", "CMLA_FORMAL_PRECISION": "Float64", "CMLA_FORMAL_CU": "cpuOnly",
            ])
        }
    }

    // MARK: - 入力と出力の名前

    @Test func inputNameKeepsReferenceAndSpeaker() throws {
        let url = URL(fileURLWithPath: "/x/m-01/EXP_FX_Explosion_Generic_7_ST_trimmed_m-01.wav")
        let input = try #require(FormalEvalInput.parse(url: url, speaker: "m-01"))
        #expect(input.reference == "EXP_FX_Explosion_Generic_7_ST_trimmed")
        #expect(input.stem == "EXP_FX_Explosion_Generic_7_ST_trimmed_m-01")
        #expect(input.outputStem == "EXP_FX_Explosion_Generic_7_ST_trimmed_m-01_pronounse_ios")
        // 話者フォルダとファイル名の話者が違うもの、wav でないものは受け付けない
        #expect(FormalEvalInput.parse(url: url, speaker: "m-02") == nil)
        #expect(FormalEvalInput.parse(url: URL(fileURLWithPath: "/x/m-01/a_m-01.npy"), speaker: "m-01") == nil)
        #expect(FormalEvalInput.parse(url: URL(fileURLWithPath: "/x/m-01/_m-01.wav"), speaker: "m-01") == nil)
    }

    @Test func listFiltersSpeakersNamesAndLimit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("formal_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        for spk in ["m-01", "f-01"] {
            let dir = root.appendingPathComponent(spk)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for ref in ["A", "B", "C"] {
                try Data().write(to: dir.appendingPathComponent("\(ref)_\(spk).wav"))
            }
        }
        try Data().write(to: root.appendingPathComponent("m-01/bad_name.wav"))
        let base: [String: String] = ["CMLA_FORMAL_SESSION": "s", "CMLA_FORMAL_PRECISION": "Float32", "CMLA_FORMAL_CU": "cpuOnly"]

        let all = try FormalEvalInput.list(root: root, config: FormalEvalConfig.fromEnvironment(base))
        #expect(all.inputs.map({ $0.stem }) == ["A_f-01", "B_f-01", "C_f-01", "A_m-01", "B_m-01", "C_m-01"])
        #expect(all.rejected == ["m-01/bad_name.wav"])

        var env = base
        env["CMLA_FORMAL_SPEAKERS"] = "m-01"
        env["CMLA_FORMAL_LIMIT"] = "2"
        #expect(try FormalEvalInput.list(root: root, config: FormalEvalConfig.fromEnvironment(env)).inputs.map({ $0.stem })
                == ["A_m-01", "B_m-01"])

        env = base
        env["CMLA_FORMAL_NAMES"] = "C_m-01,B_f-01"
        #expect(try FormalEvalInput.list(root: root, config: FormalEvalConfig.fromEnvironment(env)).inputs.map({ $0.stem })
                == ["B_f-01", "C_m-01"])
    }

    // MARK: - 再開のための CSV

    @Test func csvEscapeRoundTrips() {
        let fields = ["plain", "with,comma", "with \"quote\"", ""]
        let line = fields.map(FormalEvalRunner.escape).joined(separator: ",")
        #expect(FormalEvalRunner.parseCSVLine(line) == fields)
    }

    @Test func completedStemsOnlyCountsOk() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("results_\(UUID().uuidString).csv")
        defer { try? FileManager.default.removeItem(at: url) }
        let header = FormalEvalRunner.csvHeader
        func line(_ values: [String: String]) -> String {
            header.map({ FormalEvalRunner.escape(values[$0] ?? "") }).joined(separator: ",")
        }
        let text = [
            header.joined(separator: ","),
            line(["speaker": "m-01", "reference": "A", "status": "ok"]),
            line(["speaker": "m-01", "reference": "B", "status": "error", "error": "x, y"]),
            line(["speaker": "f-01", "reference": "A", "status": "skipped_existing_output"]),
        ].joined(separator: "\n") + "\n"
        try text.write(to: url, atomically: true, encoding: .utf8)
        #expect(try FormalEvalRunner.completedStems(resultsURL: url) == ["A_m-01"])
        #expect(try FormalEvalRunner.completedStems(
            resultsURL: url.deletingLastPathComponent().appendingPathComponent("none.csv")) == [])
    }
}
