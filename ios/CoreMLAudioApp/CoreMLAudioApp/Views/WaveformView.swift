import SwiftUI
import Charts

/// 波形を表示するビュー (min/max envelope で AreaMark 描画)
struct WaveformView: View {
    let waveform: [Float]
    let sampleRate: Double
    let label: String

    /// バケットごとの min/max を保持
    private var envelopeBuckets: [EnvelopeBucket] {
        let maxBuckets = 800
        let count = waveform.count
        guard count > 0 else { return [] }

        let bucketSize = max(1, count / maxBuckets)
        let bucketCount = (count + bucketSize - 1) / bucketSize
        var buckets = [EnvelopeBucket]()
        buckets.reserveCapacity(bucketCount)

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
            buckets.append(EnvelopeBucket(
                time: timeSec,
                minAmplitude: Double(minVal),
                maxAmplitude: Double(maxVal)
            ))
        }
        return buckets
    }

    private var duration: Double {
        Double(waveform.count) / sampleRate
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)

            Chart(envelopeBuckets) { bucket in
                AreaMark(
                    x: .value("Time", bucket.time),
                    yStart: .value("Min", bucket.minAmplitude),
                    yEnd: .value("Max", bucket.maxAmplitude)
                )
                .foregroundStyle(.blue.opacity(0.7))
            }
            .chartYScale(domain: -1.0...1.0)
            .chartXScale(domain: 0...max(0.01, duration))
            .chartYAxisLabel("Amp")
            .chartXAxisLabel("sec")
            .frame(height: 120)
        }
    }
}

private struct EnvelopeBucket: Identifiable {
    let id = UUID()
    let time: Double
    let minAmplitude: Double
    let maxAmplitude: Double
}
