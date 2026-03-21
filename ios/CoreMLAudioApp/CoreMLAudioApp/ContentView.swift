//
//  ContentView.swift
//  CoreMLAudioApp
//
//  Created by Sakurai Erika on 2026/03/21.
//

import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var synthesizer = AudioSynthesizer()
    @State private var audioPlayer: AVAudioPlayer?
    @State private var errorMessage: String?
    @State private var outputWaveform: [Float]?

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                // ステータス表示
                GroupBox("ステータス") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(synthesizer.status)
                            .font(.body)
                        if synthesizer.isProcessing {
                            ProgressView(value: synthesizer.progress)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // アクションボタン
                VStack(spacing: 12) {
                    Button {
                        Task { await runSynthesis() }
                    } label: {
                        Label("合成実行", systemImage: "waveform")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(synthesizer.isProcessing)

                    Button {
                        playOutput()
                    } label: {
                        Label("再生", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(outputWaveform == nil || synthesizer.isProcessing)

                    Button {
                        stopPlayback()
                    } label: {
                        Label("停止", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(audioPlayer == nil || !(audioPlayer?.isPlaying ?? false))
                }

                // エラー表示
                if let errorMessage {
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

    private func runSynthesis() async {
        errorMessage = nil
        outputWaveform = nil

        do {
            try synthesizer.loadModels()

            guard let inputURL = Bundle.main.url(forResource: "input_sample", withExtension: "wav") else {
                errorMessage = "input_sample.wav がバンドルに見つかりません"
                return
            }

            let waveform = try await synthesizer.synthesize(inputURL: inputURL)
            outputWaveform = waveform
            playOutput()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func playOutput() {
        guard let waveform = outputWaveform else { return }

        // オーディオセッション設定
        #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            errorMessage = "オーディオセッションエラー: \(error.localizedDescription)"
            return
        }
        #endif

        let sampleRate: Double = AudioFeatureExtractor.sampleRate
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(waveform.count))!
        buffer.frameLength = AVAudioFrameCount(waveform.count)

        let channelData = buffer.floatChannelData![0]
        for i in 0..<waveform.count {
            channelData[i] = waveform[i]
        }

        // PCMBuffer → wav Data → AVAudioPlayer
        do {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("output.wav")
            try? FileManager.default.removeItem(at: tempURL)

            // WAV ファイルとして書き出す (Int16 PCM)
            let wavSettings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
            ]
            // スコープで閉じて書き込みを確定させる
            do {
                let outputFile = try AVAudioFile(forWriting: tempURL, settings: wavSettings, commonFormat: .pcmFormatFloat32, interleaved: false)
                try outputFile.write(from: buffer)
            }

            let player = try AVAudioPlayer(contentsOf: tempURL)
            player.volume = 1.0
            player.prepareToPlay()
            player.play()
            audioPlayer = player
        } catch {
            errorMessage = "再生エラー: \(error.localizedDescription)"
        }
    }

    private func stopPlayback() {
        audioPlayer?.stop()
    }
}

#Preview {
    ContentView()
}
