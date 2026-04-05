//
//  ContentView.swift
//  CoreMLAudioApp
//
//  Created by Sakurai Erika on 2026/03/21.
//

import SwiftUI

struct ContentView: View {
    @State private var viewModel = SynthesisViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                // ステータス表示
                GroupBox("ステータス") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(viewModel.status)
                            .font(.body)
                        if viewModel.isProcessing {
                            ProgressView(value: viewModel.progress)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // アクションボタン
                VStack(spacing: 12) {
                    Button(
                        action: { Task { await viewModel.runSynthesis() } },
                        label: {
                            Label("合成実行", systemImage: "waveform")
                                .frame(maxWidth: .infinity)
                        }
                    )
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isProcessing)

                    Button(
                        action: { viewModel.playOutput() },
                        label: {
                            Label("再生", systemImage: "play.fill")
                                .frame(maxWidth: .infinity)
                        }
                    )
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.canPlay)

                    Button(
                        action: { viewModel.stopPlayback() },
                        label: {
                            Label("停止", systemImage: "stop.fill")
                                .frame(maxWidth: .infinity)
                        }
                    )
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.isPlaying)
                }

                // 解析結果
                if let result = viewModel.synthesisResult {
                    NavigationLink(
                        destination: { AnalysisView(result: result) },
                        label: {
                            Label("解析結果を表示", systemImage: "chart.bar.xaxis")
                                .frame(maxWidth: .infinity)
                        }
                    )
                    .buttonStyle(.bordered)
                }

                // エラー表示
                if let errorMessage = viewModel.errorMessage {
                    GroupBox {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }

                Spacer()

                // 情報
                GroupBox("情報") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("モデル: PronounSE (Float16)")
                        Text("サンプルレート: 22050 Hz")
                        Text("入力: input_sample.wav (バンドル)")
                    }
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
            .navigationTitle("CoreML Audio")
        }
    }
}

#Preview {
    ContentView()
}
