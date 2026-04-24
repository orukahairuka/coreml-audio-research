import CoreGraphics

/// メルスペクトログラムを CGImage に変換するユーティリティ
///
/// 入力は dB スケール (-80〜0) の連続配列 [frameCount × nMels]。
/// Magma カラーマップで着色した RGBA 画像を返す。
enum MelSpectrogramRenderer {

    /// dB スケールのメル配列から CGImage を生成する
    static func makeImage(melData: [Float], frameCount: Int, nMels: Int) -> CGImage? {
        guard frameCount > 0, nMels > 0,
              melData.count == frameCount * nMels else { return nil }

        let width = frameCount
        let height = nMels
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for x in 0..<width {
            for y in 0..<height {
                // Y 軸を反転 (低周波が下)
                let melIndex = nMels - 1 - y
                let dataIndex = x * nMels + melIndex
                let db = melData[dataIndex]

                // -80〜0 → 0〜1 に正規化
                let normalized = (db + 80.0) / 80.0
                let t = max(0.0, min(1.0, normalized))

                let (r, g, b) = magmaColor(t)
                let pixelIndex = (y * width + x) * 4
                pixelData[pixelIndex] = UInt8(r * 255)
                pixelData[pixelIndex + 1] = UInt8(g * 255)
                pixelData[pixelIndex + 2] = UInt8(b * 255)
                pixelData[pixelIndex + 3] = 255
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        return context.makeImage()
    }

    /// Magma カラーマップ (matplotlib 互換、主要な制御点から補間)
    private static func magmaColor(_ t: Float) -> (Float, Float, Float) {
        let stops: [(t: Float, r: Float, g: Float, b: Float)] = [
            (0.00, 0.001, 0.000, 0.014),
            (0.13, 0.082, 0.035, 0.215),
            (0.25, 0.232, 0.059, 0.437),
            (0.38, 0.390, 0.100, 0.502),
            (0.50, 0.550, 0.161, 0.506),
            (0.63, 0.716, 0.215, 0.475),
            (0.75, 0.882, 0.327, 0.408),
            (0.88, 0.978, 0.550, 0.365),
            (1.00, 0.987, 0.991, 0.750),
        ]

        if t <= stops[0].t { return (stops[0].r, stops[0].g, stops[0].b) }
        if t >= stops[stops.count - 1].t { return (stops[stops.count - 1].r, stops[stops.count - 1].g, stops[stops.count - 1].b) }

        for i in 0..<(stops.count - 1) {
            if t >= stops[i].t && t <= stops[i + 1].t {
                let frac = (t - stops[i].t) / (stops[i + 1].t - stops[i].t)
                let r = stops[i].r + frac * (stops[i + 1].r - stops[i].r)
                let g = stops[i].g + frac * (stops[i + 1].g - stops[i].g)
                let b = stops[i].b + frac * (stops[i + 1].b - stops[i].b)
                return (r, g, b)
            }
        }

        return (0, 0, 0)
    }
}
