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
                        XCTFail("組み合わせ失敗 (\(combo)): \(error)")
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
        option.tap()
    }

    /// 計算デバイス Picker (menu) を開いて選択
    @MainActor
    private func selectComputeUnit(_ app: XCUIApplication, _ rawValue: String) throws {
        let picker = app.buttons["computeUnitPicker"]
        guard picker.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("computeUnitPicker")
        }
        picker.tap()

        let identifier = "computeUnit.\(rawValue)"
        let option = app.buttons[identifier]
        guard option.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound(identifier)
        }
        option.tap()
    }

    /// 合成実行ボタンを押して「合成完了」ステータスを待つ
    @MainActor
    private func runSynthesisAndWait(_ app: XCUIApplication, timeout: TimeInterval = 180) throws {
        let button = app.buttons["synthesizeButton"]
        guard button.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("synthesizeButton")
        }
        // Processing 中は disabled。確実に有効化されてからタップ
        let enabled = NSPredicate(format: "isEnabled == true")
        expectation(for: enabled, evaluatedWith: button, handler: nil)
        waitForExpectations(timeout: 10)
        button.tap()

        let statusLabel = app.staticTexts["synthesisStatus"]
        guard statusLabel.waitForExistence(timeout: 5) else {
            throw UITestError.elementNotFound("synthesisStatus")
        }
        // status が "合成完了" または "エラー" になるまで待つ
        let done = NSPredicate(format: "label == %@ OR label == %@", "合成完了", "エラー")
        expectation(for: done, evaluatedWith: statusLabel, handler: nil)
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
