//
//  CoreMLAudioAppUITests.swift
//  CoreMLAudioAppUITests
//
//  Created by Sakurai Erika on 2026/03/21.
//

import XCTest

final class CoreMLAudioAppUITests: XCTestCase {

    override func setUpWithError() throws {
        // バッチ取得では1件失敗しても残りを続行したい
        continueAfterFailure = true
    }

    /// 精度3 × 計算デバイス4 = 12通りを全部回して
    /// Documents/Result/mel/ にメル値 (.npy) と画像 (.png) を書き出す。
    ///
    /// 取り出しはテスト終了後にホスト側で:
    ///   scripts/extract_ui_test_results.sh
    @MainActor
    func testCaptureAllCombinations() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "allCombinations"
        app.launch()

        // 精度 × デバイスの組み合わせ。rawValue が accessibilityIdentifier 末尾と一致する。
        let precisions = ["Float32", "Float16", "Int8"]
        let computeUnits = ["cpuOnly", "cpuAndGPU", "cpuAndNE", "all"]

        var failures: [(combo: String, reason: String)] = []

        for precision in precisions {
            for computeUnit in computeUnits {
                let combo = "\(precision) × \(computeUnit)"
                XCTContext.runActivity(named: "Combination: \(combo)") { _ in
                    do {
                        try runOneCombination(app: app, precision: precision, computeUnit: computeUnit)
                    } catch {
                        failures.append((combo, "\(error)"))
                    }
                }
            }
        }

        if !failures.isEmpty {
            let summary = failures.map { "- \($0.combo): \($0.reason)" }.joined(separator: "\n")
            XCTFail("失敗した組み合わせ: \(failures.count) 件\n\(summary)")
        }
    }

    /// Fresh app から F32 × cpuAndGPU を 1 発目に走らせる検証用テスト。
    /// 手動操作と同じ出力 (mode B, rms ~5029) になるかを確認する。
    /// (実験 A: sleep なし baseline)
    @MainActor
    func testFp32GpuFreshFirst() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "freshFirstNoSleep"
        app.launch()
        try runOneCombination(app: app, precision: "Float32", computeUnit: "cpuAndGPU")
    }

    /// 実験 B: launch 直後に 10 秒 sleep を入れた F32 × cpuAndGPU 単発。
    /// sleep なし版と比較して loud に化けるかを見る (launch 直後タイミング問題仮説の検証)。
    @MainActor
    func testFp32GpuFreshFirstWithLaunchSleep10() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "freshFirstSleep10"
        app.launch()
        // 起動直後の AVAudioSession / Metal ウォームアップ / CoreML 遅延初期化が
        // 影響しているかを切り分けるため、10 秒だけ静置してから初操作する。
        Thread.sleep(forTimeInterval: 10)
        try runOneCombination(app: app, precision: "Float32", computeUnit: "cpuAndGPU")
    }

    /// 実験 C: F32 × cpuAndGPU 単発の再現性確認。
    /// アプリを 3 回新規起動して 1 発ずつ実行。md5 / rms / peak の分布を取る。
    @MainActor
    func testFp32GpuFreshFirstRepeat3() throws {
        for i in 1...3 {
            let app = XCUIApplication()
            app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
            app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "freshFirstRepeat\(i)"
            app.launch()
            try runOneCombination(app: app, precision: "Float32", computeUnit: "cpuAndGPU")
            app.terminate()
            // 再 launch までに少し間を取る (terminate 完了待ち)
            Thread.sleep(forTimeInterval: 2)
        }
    }

    /// 実験 D-1: F32 × cpuOnly 単発の再現性確認。
    @MainActor
    func testFp32CpuOnlyFreshFirst() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "freshFirstCpuOnly"
        app.launch()
        try runOneCombination(app: app, precision: "Float32", computeUnit: "cpuOnly")
    }

    /// 実験 D-2: F32 × all 単発の再現性確認。
    @MainActor
    func testFp32AllFreshFirst() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "freshFirstAll"
        app.launch()
        try runOneCombination(app: app, precision: "Float32", computeUnit: "all")
    }

    /// 12 通りを cpuAndGPU から始める順序版。
    /// fp32 × cpuAndGPU を 1 番目に走らせれば mode B (loud) になるかの検証。
    @MainActor
    func testCaptureAllCombinationsGpuFirst() throws {
        let app = XCUIApplication()
        app.launchEnvironment["CMLA_DEBUG_SNAPSHOT"] = "1"
        app.launchEnvironment["CMLA_DEBUG_RUN_LABEL"] = "allCombinationsGpuFirst"
        app.launch()

        let precisions = ["Float32", "Float16", "Int8"]
        let computeUnits = ["cpuAndGPU", "cpuOnly", "cpuAndNE", "all"] // GPU を先頭に

        var failures: [(combo: String, reason: String)] = []
        for precision in precisions {
            for computeUnit in computeUnits {
                let combo = "\(precision) × \(computeUnit)"
                XCTContext.runActivity(named: "Combination: \(combo)") { _ in
                    do {
                        try runOneCombination(app: app, precision: precision, computeUnit: computeUnit)
                    } catch {
                        failures.append((combo, "\(error)"))
                    }
                }
            }
        }

        if !failures.isEmpty {
            let summary = failures.map { "- \($0.combo): \($0.reason)" }.joined(separator: "\n")
            XCTFail("失敗した組み合わせ: \(failures.count) 件\n\(summary)")
        }
    }

    // MARK: - Helpers

    private enum UITestError: Error, CustomStringConvertible {
        case elementNotFound(String)
        case synthesisDidNotComplete(status: String)

        var description: String {
            switch self {
            case .elementNotFound(let name): return "要素が見つからない: \(name)"
            case .synthesisDidNotComplete(let status): return "合成完了に至らず (最終ステータス: \(status))"
            }
        }
    }

    /// 精度 Picker (segmented) から選択
    @MainActor
    private func selectPrecision(_ app: XCUIApplication, _ rawValue: String) throws {
        let identifier = "precision.\(rawValue)"
        let option = app.buttons[identifier]
        guard option.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound(identifier)
        }
        // 直前の合成完了直後は picker がまだ disabled な瞬間があるので有効化を待つ
        try waitUntilEnabled(option, name: identifier)
        option.tap()
    }

    /// 計算デバイス Picker (menu) を開いて選択
    @MainActor
    private func selectComputeUnit(_ app: XCUIApplication, _ rawValue: String) throws {
        let picker = app.buttons["computeUnitPicker"]
        guard picker.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("computeUnitPicker")
        }
        try waitUntilEnabled(picker, name: "computeUnitPicker")
        picker.tap()

        let identifier = "computeUnit.\(rawValue)"
        let option = app.buttons[identifier]
        guard option.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound(identifier)
        }
        option.tap()
    }

    /// 要素が isEnabled == true になるまで待つ
    @MainActor
    private func waitUntilEnabled(_ element: XCUIElement, name: String, timeout: TimeInterval = 10) throws {
        if element.isEnabled { return }
        let enabled = NSPredicate(format: "isEnabled == true")
        expectation(for: enabled, evaluatedWith: element, handler: nil)
        waitForExpectations(timeout: timeout)
        if !element.isEnabled {
            throw UITestError.elementNotFound("\(name) (enabled 待ちタイムアウト)")
        }
    }

    /// 合成実行ボタンを押して「合成完了」ステータスを待つ
    @MainActor
    private func runSynthesisAndWait(_ app: XCUIApplication, timeout: TimeInterval = 180) throws {
        let button = app.buttons["synthesizeButton"]
        guard button.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("synthesizeButton")
        }
        // Processing 中は disabled。確実に有効化されてからタップ
        let enabledPred = NSPredicate(format: "isEnabled == true")
        expectation(for: enabledPred, evaluatedWith: button, handler: nil)
        waitForExpectations(timeout: 10)
        button.tap()

        let statusLabel = app.staticTexts["synthesisStatus"]
        guard statusLabel.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("synthesisStatus")
        }

        // status は前回の "合成完了" がまだ残っている瞬間があり、ラベル監視だと
        // 合成を待たずに即マッチしてしまう。ボタンの isEnabled は isProcessing と
        // 直結しているので、disabled (合成開始) → enabled (合成終了) の遷移を見る。
        let processingPred = NSPredicate(format: "isEnabled == false")
        expectation(for: processingPred, evaluatedWith: button, handler: nil)
        waitForExpectations(timeout: 5)

        let finishedPred = NSPredicate(format: "isEnabled == true")
        expectation(for: finishedPred, evaluatedWith: button, handler: nil)
        waitForExpectations(timeout: timeout)

        let finalStatus = statusLabel.label
        if finalStatus != "合成完了" {
            throw UITestError.synthesisDidNotComplete(status: finalStatus)
        }
    }

    @MainActor
    private func runOneCombination(app: XCUIApplication, precision: String, computeUnit: String) throws {
        try selectPrecision(app, precision)
        try selectComputeUnit(app, computeUnit)
        try runSynthesisAndWait(app)
    }
}
