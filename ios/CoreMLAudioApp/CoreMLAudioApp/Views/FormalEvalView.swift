import SwiftUI
import UIKit

/// 正式評価（FAD・RTF）用の一括合成モードで起動したときの画面
///
/// 合成は `FormalEvalRunner` が行い、終わったらプロセスを終了する。
/// 合成中に画面が消灯してスリープしないよう、アイドルタイマーを止めておく。
struct FormalEvalView: View {
    @State private var message = "正式評価の合成中…"

    var body: some View {
        Text(message)
            .font(.headline)
            .padding()
            .task({
                UIApplication.shared.isIdleTimerDisabled = true
                let code = await FormalEvalRunner.runFromEnvironment()
                message = "合成終了 (\(code))"
                exit(code)
            })
    }
}
