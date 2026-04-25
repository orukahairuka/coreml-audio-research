import SwiftUI

/// メルスペクトログラムをヒートマップとして表示するビュー
struct SpectrogramView: View {
    let melData: [Float]  // [frameCount × nMels], dB スケール (-80〜0)
    let frameCount: Int
    let nMels: Int
    let sampleRate: Double
    let label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)

            if let image = MelSpectrogramRenderer.makeImage(melData: melData, frameCount: frameCount, nMels: nMels) {
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .interpolation(.medium)
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 160)
            }
        }
    }
}
