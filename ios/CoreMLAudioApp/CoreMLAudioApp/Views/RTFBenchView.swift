import SwiftUI
import UIKit

/// 正式 RTF 測定モードで起動したときの画面（終わったらプロセスを終了する）
struct RTFBenchView: View {
    @State private var message = "RTF 測定中…"

    var body: some View {
        Text(message)
            .font(.headline)
            .padding()
            .task({
                UIApplication.shared.isIdleTimerDisabled = true
                let code = await RTFBenchRunner.runFromEnvironment()
                message = "RTF 測定終了 (\(code))"
                exit(code)
            })
    }
}
