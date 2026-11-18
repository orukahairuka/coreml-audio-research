import Foundation

/// メインスレッドが処理を受け付けられなかった時間を測る
///
/// メインキューで 2 ms ごとに動くタイマーの発火の間隔を記録する。
/// 間隔が長いほど、その間メインスレッドが他の処理で占有されて、画面の更新ができなかったことを表す。
nonisolated final class MainThreadStallProbe {

    struct Summary {
        /// 発火の間隔の最大値 (ms)
        let maxGapMs: Double
        /// 1 フレーム（60 Hz で 16.7 ms）を超えた回数
        let over16Count: Int
        /// 2 フレーム（33.3 ms）を超えた回数
        let over33Count: Int
        /// 1 フレームを超えた間隔の合計 (ms)
        let over16TotalMs: Double
    }

    private var timer: DispatchSourceTimer?
    private var last: UInt64 = 0
    private var maxGapMs = 0.0
    private var over16Count = 0
    private var over33Count = 0
    private var over16TotalMs = 0.0

    /// メインスレッドから呼ぶ
    func start() {
        last = DispatchTime.now().uptimeNanoseconds
        let source = DispatchSource.makeTimerSource(flags: .strict, queue: .main)
        source.schedule(deadline: .now(), repeating: .milliseconds(2), leeway: .nanoseconds(0))
        source.setEventHandler(handler: { [weak self] in self?.tick() })
        timer = source
        source.resume()
    }

    /// メインスレッドから呼ぶ
    func stop() -> Summary {
        tick()
        timer?.cancel()
        timer = nil
        return Summary(maxGapMs: maxGapMs, over16Count: over16Count, over33Count: over33Count, over16TotalMs: over16TotalMs)
    }

    private func tick() {
        let now = DispatchTime.now().uptimeNanoseconds
        let gapMs = Double(now - last) / 1_000_000
        last = now
        maxGapMs = max(maxGapMs, gapMs)
        if gapMs > 1000.0 / 60.0 {
            over16Count += 1
            over16TotalMs += gapMs
        }
        if gapMs > 2000.0 / 60.0 { over33Count += 1 }
    }
}
