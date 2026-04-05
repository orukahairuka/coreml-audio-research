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

                // 入力ソース選択
                GroupBox("入力音声") {
                    VStack(alignment: .leading, spacing: 12) {
                        // サンプル音声
                        Button(
                            action: { viewModel.audioSource = .bundledSample },
                            label: {
                                HStack {
                                    Image(systemName: viewModel.audioSource == .bundledSample ? "checkmark.circle.fill" : "circle")
                                    Text("サンプル音声")
                                    Spacer()
                                }
                            }
                        )
                        .foregroundStyle(viewModel.audioSource == .bundledSample ? .primary : .secondary)

                        // 録音ファイル一覧
                        ForEach(viewModel.recordings, id: \.absoluteString) { url in
                            Button(
                                action: { viewModel.audioSource = .recording(url) },
                                label: {
                                    HStack {
                                        let isSelected = viewModel.audioSource == .recording(url)
                                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                                        Text(url.deletingPathExtension().lastPathComponent)
                                            .lineLimit(1)
                                        Spacer()
                                    }
                                }
                            )
                            .foregroundStyle(viewModel.audioSource == .recording(url) ? .primary : .secondary)
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive, action: { viewModel.deleteRecording(at: url) }) {
                                    Label("削除", systemImage: "trash")
                                }
                            }
                        }

                        // 録音ボタン
                        if viewModel.isRecording {
                            HStack {
                                Button(
                                    action: { viewModel.stopRecording() },
                                    label: {
                                        Label("停止", systemImage: "stop.circle.fill")
                                            .foregroundStyle(.red)
                                    }
                                )
                                Spacer()
                                Text(String(format: "%.1f / %.0f 秒", viewModel.recordingTime, viewModel.maxRecordingDuration))
                                    .font(.caption)
                                    .monospacedDigit()
                            }
                        } else {
                            Button(
                                action: { viewModel.startRecording() },
                                label: {
                                    Label("録音", systemImage: "mic.fill")
                                        .frame(maxWidth: .infinity)
                                }
                            )
                            .buttonStyle(.bordered)
                            .tint(.red)
                            .disabled(viewModel.isProcessing)
                        }
                    }
                }

                // 精度選択
                Picker("精度", selection: $viewModel.selectedPrecision) {
                    ForEach(ModelPrecision.allCases) { precision in
                        Text(precision.rawValue).tag(precision)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(viewModel.isProcessing)

                // 計算デバイス選択
                Picker("計算デバイス", selection: $viewModel.selectedComputeUnit) {
                    ForEach(ComputeUnitOption.allCases) { option in
                        Text(option.displayName).tag(option)
                    }
                }
                .disabled(viewModel.isProcessing)

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
                    .disabled(viewModel.isProcessing || viewModel.isRecording)

                    Button(
                        action: { viewModel.playOutput() },
                        label: {
                            Label("再生", systemImage: "play.fill")
                                .frame(maxWidth: .infinity)
                        }
                    )
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.canPlay)
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
                        Text("モデル: PronounSE (\(viewModel.selectedPrecision.rawValue))")
                        Text("計算デバイス: \(viewModel.selectedComputeUnit.displayName)")
                        Text("サンプルレート: 22050 Hz")
                        Text("入力: \(viewModel.audioSource.displayName)")
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
