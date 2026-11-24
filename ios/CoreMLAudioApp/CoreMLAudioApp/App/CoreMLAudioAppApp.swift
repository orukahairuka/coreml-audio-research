//
//  CoreMLAudioAppApp.swift
//  CoreMLAudioApp
//
//  Created by Sakurai Erika on 2026/03/21.
//

import SwiftUI

@main
struct CoreMLAudioAppApp: App {
    var body: some Scene {
        WindowGroup {
            // 起動時の環境変数 CMLA_THESIS_BENCH があれば、UI を使わない計測モードで起動する
            if ThesisBenchConfig.isRequested {
                ThesisBenchView()
            } else if RTFBenchRunner.Config.isRequested {
                // CMLA_RTF_BENCH=1 なら、正式 RTF 測定モード（反復測定・出力は保存しない）で起動する
                RTFBenchView()
            } else if FormalEvalConfig.isRequested {
                // CMLA_FORMAL_EVAL=1 なら、正式評価（FAD・RTF）用の一括合成モードで起動する
                FormalEvalView()
            } else {
                ContentView()
            }
        }
    }
}
