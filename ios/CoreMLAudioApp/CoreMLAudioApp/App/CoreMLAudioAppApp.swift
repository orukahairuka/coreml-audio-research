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
            } else {
                ContentView()
            }
        }
    }
}
