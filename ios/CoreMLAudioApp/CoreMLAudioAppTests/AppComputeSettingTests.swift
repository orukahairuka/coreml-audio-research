import Testing
@testable import CoreMLAudioApp

/// アプリ画面の計算デバイスの設定（`AppComputeSetting`）のテスト
struct AppComputeSettingTests {

    /// 3 モデル同じ指定の 4 種類は、従来の ComputeUnitOption と同じ名前・同じ指定になる
    /// （画面の accessibilityIdentifier と保存するファイル名が変わらない）
    @Test func uniformSettingsMatchComputeUnitOption() {
        for option in ComputeUnitOption.allCases {
            let setting = AppComputeSetting(rawValue: option.rawValue)
            #expect(setting != nil)
            #expect(setting?.encoderDecoderUnit == option)
            #expect(setting?.hifiganUnit == option)
            #expect(setting?.displayName == option.displayName)
        }
    }

    /// 条件 C: Encoder・Decoder は cpuAndNE、HiFi-GAN は cpuOnly
    @Test func hifiganCPUSettingSplitsComputeUnits() throws {
        let setting = AppComputeSetting.cpuAndNEHifiganCPU
        #expect(setting.encoderDecoderUnit == .cpuAndNE)
        #expect(setting.hifiganUnit == .cpuOnly)
    }

    /// 条件 C の表記は、計測モードの CSV の compute_units 列と同じ
    @Test func hifiganCPULabelMatchesThesisBenchLabel() throws {
        let config = try ThesisBenchConfig.fromEnvironment([
            "CMLA_THESIS_BENCH": "infer", "CMLA_THESIS_PRECISION": "Float16",
            "CMLA_THESIS_CU": "cpuAndNE", "CMLA_THESIS_CU_HIFI": "cpuOnly",
        ])
        #expect(AppComputeSetting.cpuAndNEHifiganCPU.rawValue == config.computeUnitsLabel)
    }
}
