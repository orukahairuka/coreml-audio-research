import SwiftUI

/// パイプライン各ステップの統計情報を表示するデバッグビュー
struct DebugStatsView: View {
    let debugInfo: PipelineDebugInfo
    @State private var copied = false

    var body: some View {
        GroupBox("Debug Stats") {
            VStack(alignment: .leading, spacing: 12) {

                Button(action: {
                    UIPasteboard.general.string = buildCopyText()
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
                }, label: {
                    Label(copied ? "Copied" : "Copy All", systemImage: copied ? "checkmark" : "doc.on.doc")
                })
                .buttonStyle(.bordered)

                // Encoder
                statsRow(title: "Encoder output", stats: debugInfo.encoderOutput)

                Divider()

                // Decoder ステップ
                Text("Decoder steps (\(debugInfo.decoderSteps.count) recorded)")
                    .font(.headline)

                ForEach(debugInfo.decoderSteps, id: \.step) { step in
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Step \(step.step)")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundStyle(stepColor(step))
                        Text("  mel_out:    \(step.melOut.summary)")
                        Text("  postnet_out: \(step.postnetOut.summary)")
                    }
                    .font(.system(.caption, design: .monospaced))
                }

                Divider()

                // HiFi-GAN
                statsRow(title: "HiFi-GAN input", stats: debugInfo.hifiganInput)
                statsRow(title: "HiFi-GAN output", stats: debugInfo.hifiganOutput)

                Divider()

                // Waveform
                statsRow(title: "Waveform (after de-emphasis)", stats: debugInfo.waveformAfterDeemphasis)
            }
        }
    }

    private func buildCopyText() -> String {
        var lines = [String]()
        lines.append("=== Debug Stats ===")
        lines.append("")
        lines.append("[Encoder output]")
        lines.append(debugInfo.encoderOutput.summary)
        lines.append("")
        lines.append("[Decoder steps] (\(debugInfo.decoderSteps.count) recorded)")
        for step in debugInfo.decoderSteps {
            lines.append("Step \(step.step)")
            lines.append("  mel_out:     \(step.melOut.summary)")
            lines.append("  postnet_out: \(step.postnetOut.summary)")
        }
        lines.append("")
        lines.append("[HiFi-GAN input]")
        lines.append(debugInfo.hifiganInput.summary)
        lines.append("[HiFi-GAN output]")
        lines.append(debugInfo.hifiganOutput.summary)
        lines.append("")
        lines.append("[Waveform (after de-emphasis)]")
        lines.append(debugInfo.waveformAfterDeemphasis.summary)
        return lines.joined(separator: "\n")
    }

    @ViewBuilder
    private func statsRow(title: String, stats: ArrayStats) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.headline)
                .foregroundStyle(stats.hasNaN || stats.hasInf ? .red : .primary)
            Text(stats.summary)
                .font(.system(.caption, design: .monospaced))
        }
    }

    private func stepColor(_ step: DecoderStepStats) -> Color {
        if step.melOut.hasNaN || step.melOut.hasInf
            || step.postnetOut.hasNaN || step.postnetOut.hasInf {
            return .red
        }
        return .primary
    }
}
