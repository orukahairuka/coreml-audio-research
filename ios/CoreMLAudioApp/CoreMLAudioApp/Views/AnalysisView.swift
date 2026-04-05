import SwiftUI

/// 入力/出力の波形とメルスペクトログラムを4パネルで比較表示するビュー
struct AnalysisView: View {
    let result: SynthesisResult

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // 入力セクション
                GroupBox("Input") {
                    VStack(spacing: 12) {
                        WaveformView(
                            waveform: result.inputWaveform,
                            sampleRate: result.sampleRate,
                            label: "Waveform"
                        )

                        SpectrogramView(
                            melData: result.inputMelSpectrogram,
                            frameCount: result.inputFrameCount,
                            nMels: result.nMels,
                            sampleRate: result.sampleRate,
                            label: "Mel Spectrogram"
                        )
                    }
                }

                // 出力セクション
                GroupBox("Output") {
                    VStack(spacing: 12) {
                        WaveformView(
                            waveform: result.outputWaveform,
                            sampleRate: result.sampleRate,
                            label: "Waveform"
                        )

                        SpectrogramView(
                            melData: result.outputMelSpectrogram,
                            frameCount: result.outputFrameCount,
                            nMels: result.nMels,
                            sampleRate: result.sampleRate,
                            label: "Mel Spectrogram"
                        )
                    }
                }

                // デバッグセクション
                DebugStatsView(debugInfo: result.debugInfo)
            }
            .padding()
        }
        .navigationTitle("解析結果")
    }
}
