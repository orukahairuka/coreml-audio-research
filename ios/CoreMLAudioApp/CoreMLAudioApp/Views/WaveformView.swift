import SwiftUI
import Charts

/// 波形を表示するビュー (min/max envelope でダウンサンプリング)
struct WaveformView: View {
    let waveform: [Float]
    let sampleRate: Double
    let label: String

    /// ダウンサンプリングされた波形ポイント
    private var envelopePoints: [EnvelopePoint] {
        let maxPoints = 800
        let count = waveform.count
        guard count > 0 else { return [] }

        let bucketSize = max(1, count / maxPoints)
        let bucketCount = (count + bucketSize - 1) / bucketSize
        var points = [EnvelopePoint]()
        points.reserveCapacity(bucketCount * 2)

        for i in 0..<bucketCount {
            let start = i * bucketSize
            let end = min(start + bucketSize, count)
            var minVal: Float = .greatestFiniteMagnitude
            var maxVal: Float = -.greatestFiniteMagnitude
            for j in start..<end {
                let v = waveform[j]
                if v < minVal { minVal = v }
                if v > maxVal { maxVal = v }
            }
            let timeSec = Double(start + (end - start) / 2) / sampleRate
            points.append(EnvelopePoint(time: timeSec, amplitude: Double(minVal)))
            points.append(EnvelopePoint(time: timeSec, amplitude: Double(maxVal)))
        }
        return points
    }

    private var duration: Double {
        Double(waveform.count) / sampleRate
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)

            Chart(envelopePoints) { point in
                LineMark(
                    x: .value("Time", point.time),
                    y: .value("Amp", point.amplitude)
                )
                .foregroundStyle(.blue)
            }
            .chartYScale(domain: -1.0...1.0)
            .chartXScale(domain: 0...max(0.01, duration))
            .chartYAxisLabel("Amp")
            .chartXAxisLabel("sec")
            .frame(height: 80)
        }
    }
}

private struct EnvelopePoint: Identifiable {
    let id = UUID()
    let time: Double
    let amplitude: Double
}
