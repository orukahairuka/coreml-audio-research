import SwiftUI
import UIKit

/// 卒論第4章の計測モードで起動したときの画面
///
/// 計測は `ThesisBenchRunner` が行い、終わったらプロセスを終了する。
/// 計測中に画面が消灯してスリープしないよう、アイドルタイマーを止めておく。
struct ThesisBenchView: View {
    @State private var message = "計測中…"

    var body: some View {
        Text(message)
            .font(.headline)
            .padding()
            .task({
                UIApplication.shared.isIdleTimerDisabled = true
                let code = await ThesisBenchRunner.runFromEnvironment()
                message = "計測終了 (\(code))"
                exit(code)
            })
    }
}
